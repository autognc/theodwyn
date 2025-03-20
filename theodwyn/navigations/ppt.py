import threading
import numpy                            as np
import cv2                              as cv
from copy                               import deepcopy
from rohan.common.base_navigations      import ThreadedNavigationBase
from rohan.common.logging               import Logger
from rohan.utils.timers                 import IntervalTimer
from typing                             import TypeVar, Optional, Tuple, Dict, List
from numpy.typing                       import NDArray

# Local Imports
from pose_utils                         import InferUtils as iut
from pose_utils                         import Bearing
from mekf_ppt                           import MEKF_ppt
from mekf_ppt                           import MEKF_ppt_Dynamics
import measurement

import pdb 

SelfMEKF    = TypeVar("SelfMEKF", bound = "MEKF")

class MEKF(ThreadedNavigationBase):
    """
    Model for a MEKF
    """
    
    # for logging
    process_name            : str                       = "MEKF (threaded) + inference (threaded)"
    frame_in                : Optional[NDArray]         = None
    mekf_prop_timer         : IntervalTimer
    mekf_meas_timer         : IntervalTimer
    # some_flag               : threading.Event            


    def __init__(
        self,
        kps3D_path          : str, 
        num_kps             : int,
        model_path          : str, 
        meas_model          : str,
        filter_dt           : float,
        meas_dt             : float,
        process_noise_std   : float,
        meas_noise_std      : float,
        max_flip_deg        : float                 = 45,
        tau                 : float                 = 1, 
        Qpsd                : float                 = 25,
        bearing_meas_std    : float                 = np.pi/180, # 1 degree
        omega0              : NDArray               = np.array([0, 0, 0]), # initial angular velocity
        alpha0              : NDArray               = np.array([0, 0, 0]), # initial angular acceleration
        tr_co2cam_cbff      : NDArray               = np.array([0,0,0]), # translation vector from camera body fixed frame to center of chase bff frame
        pose0               : NDArray               = np.array([0, 0, 25, 1, 0, 0, 0]), # initial pose
        img_in_size         : Tuple[int,int]        = (512, 512),
        imgnet_norm         : bool                  = False,
        pad_color           : Tuple[int,int,int]    = (0, 0, 0),
        box_key             : str                   = 'boxes',
        label_key           : str                   = 'labels',
        score_key           : str                   = 'scores',
        kps_key             : str                   = 'keypoints',
        kscore_key          : str                   = 'keypoint_scores',
        outkeys             : List[str]             = ['boxes', 'labels', 'scores', 'keypoints', 'keypoint_scores'],
        Kmat                : NDArray               = np.array([[1,0,0],[0,1,0],[0,0,1]]), #TODO: change to actual Kmat
        skpped_count        : int                   = 0,
        logger              : Optional[Logger]      = None,

        **config_kwargs
    ):
        ThreadedNavigationBase.__init__( 
            self,
            logger = logger
        )
        self.kps3D     = np.load(kps3D_path)[:num_kps]
        self.model_path = model_path
        self.meas_model = meas_model
        self.mekf_dt    = filter_dt
        self.meas_dt    = meas_dt
        self.Q_std      = process_noise_std
        self.R_std      = meas_noise_std
        self.omega0     = omega0
        self.alpha0     = alpha0
        self.max_flip   = max_flip_deg
        self.tau        = tau
        self.Qpsd       = Qpsd
        self.bearing_std= bearing_meas_std
        self.cam_offset = tr_co2cam_cbff
        self.pose0      = pose0
        self.img_in_size= img_in_size
        self.imgnet_norm= imgnet_norm
        self.pad_color  = pad_color
        self.box_key    = box_key
        self.label_key  = label_key
        self.score_key  = score_key
        self.kps_key    = kps_key
        self.kscore_key = kscore_key
        self.outkeys    = outkeys
        self.Kmat       = Kmat
        self.skipped    = skpped_count

        # thread.Event() works as follows: on -> .set() | off -> .clear() | check -> .is_set()
        self.is_first_meas      = threading.Event() # flag for first measurement
        self.first_meas_proc    = threading.Event() # whether the first measurement has been processed
        self.meas_ready         = threading.Event() # flag for measurement ready

        self.mekf_prop_timer    = IntervalTimer( interval = self.mekf_dt )
        self.mekf_meas_timer    = IntervalTimer( interval = self.meas_dt) 

        # initialize the threaded model
        self.add_threaded_method(target = self.spin_filter )
        self.add_threaded_method(target = self.spin_meas_model)

    def init_navigation(self):
        """ Initialize inference model + filter"""
        mod, inputs, outputs    = iut.onnx_model_setup(model_path = self.model_path)
        self.model              = mod
        self.minput_names       = inputs
        self.moutput_names      = outputs

        self.mekf       = MEKF_ppt(self.mekf_dt, self.Q_std, self.R_std, self.max_flip, self.tau, self.Qpsd)
        # measurement fcn setup
        if self.meas_model == 'direct':
            meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_direct(a, b, c, d, e) # fix this
        elif self.meas_model == 'local':
            meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_local(a, b, c, d, e)
        elif self.meas_model == 'c++':
            meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_cpp(a, b, c, d, e)
        self.meas_fcn   = meas_fh
        # example for meas_fcn 
        # start_pose, start_covar = MEKF_ppt_Dynamics.nls_cpp(pose0, azel, kps_3D, bearing_meas_std_rad, tr_co2cam_cbff)

        pass

    def deinit_navigation(self):
        """ Deinitialize inference model """
        self.model  = None #may want to revisit for cleanup with onnxruntime
        pass


    def spin_meas_model( self ) -> None:
        """
        Threaded Process: Proceses each measurement with the inference model
        """
        while not self.sigterm.is_set():
            # Inference Order of operations:
            # 1) Get frame
            # 2) Process frame
            # 3) Infer on frame with model 
            # 3.1) If successful, get keypoints, box, score, azel, image, and timestep
            # 3.2) If unsuccessful, increment skipped count
            # 4) If first measurement is not set and measurement is ready, set first measurement
            # 5) If first measurement is set, process first measurement
            # 6) update az_el value in self 
            if not self.frame_in is None:
                img_np_flt      = iut.cv2_preprocess_img_np(
                                                                self.frame_in
                                                                , resize_tuple  = self.img_in_size
                                                                , imagenet_norm = self.imgnet_norm
                                                                , pad_color     = self.pad_color
                                                            ).astype(np.float32)
                img_h, img_w    = img_np_flt.shape[:2]
                ort_output_dict = iut.ort_krcnn_inference(self.model, self.minput_names, self.moutput_names, img_np_flt, output_keys = self.outkeys) 
                try: 
                    ort_sco_max_idx = np.argmax(ort_output_dict[self.score_key])
                    ort_sco_m       = ort_output_dict[self.score_key][ort_sco_max_idx]
                    ort_box_m       = ort_output_dict[self.box_key][ort_sco_max_idx]
                    ort_kps_m       = ort_output_dict[self.kps_key][ort_sco_max_idx]
                    ort_kps_2D      = np.round(ort_kps_m[:, 0:2], 3).astype(np.float32)
                    az_el,_         = Bearing.compute_azimuth_elevation(ort_kps_2D, self.Kmat)
                    self.meas_az_el = az_el

                    if not self.first_meas_proc.is_set():
                        # if the first measurement is not set and the measurement is not ready, set the first measurement
                        self.is_first_meas.set()
                    else: 
                        self.meas_ready.set()


                    # record 2D keypoints, box, score, azel, image, and timestep

                except Exception as e:
                    self.skipped += 1
                    fail_str    = f"Failed with Exception: {e} for Image ID: total skipped count: {self.skipped}"
                    print(fail_str)
            
            self.frame_in = None # Let go of the frame after processing (or attempting to process it)
                
            pass

    
    def spin_filter( self ) -> None:    
        while not self.sigterm.is_set():
            if not self.first_meas_proc.is_set():
                if self.is_first_meas.is_set():
                    # if the first measurement is set, then process first measurement
                    start_pose, start_covar = self.meas_fcn(self.pose0, self.meas_az_el, self.kps3D, self.bearing_std, self.cam_offset)
                    posi0, quat0            = start_pose[:3], start_pose[3:]
                    # update filter initial state
                    self.mekf.set_initial_state_covar(quat0, self.omega0, self.alpha0, posi0, start_covar)
                    self.first_meas_proc.set()
                    self.is_first_meas.clear()
                
                # continue to next iteration if first measurement is set 
                continue 
            
            # reach this code after self.first_meas_proc.is_set() is True
            self.mekf_prop_timer.await_interval()
            self.mekf.time_update()

            # if measurement is ready and first measurement is processed, then measurement update
            if self.meas_ready.is_set():
                self.mekf_meas_timer.await_interval()
                pose_est        = np.concatenate([self.mekf.position_est, self.mekf.global_quat_est])
                poseJ, covarJ   = self.meas_fcn(pose_est, self.meas_az_el, self.kps3D, self.bearing_std, self.cam_offset)
                posiJ, quatJ    = poseJ[:3], poseJ[3:]
                self.mekf.measurement_update(quatJ, posiJ, covarJ)
                self.mekf.mekf_reset()
                self.meas_ready.clear()
                
            # record filter pose and diagonal of covariance matrix, record measurement pose and diagonal of covariance matrix

            pass 

    def pass_in_frame( 
        self,
        image : NDArray
    ) -> None:
        """
        Retrieves return code and frame information
        """
        self.frame_in = image
        