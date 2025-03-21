import os
import shutil
import threading
import numpy                            as np
import cv2                              as cv
from copy                               import deepcopy
from typing                             import TypeVar, Optional, Tuple, Dict, List
from numpy.typing                       import NDArray
from time                               import time, sleep, strftime
from time                               import perf_counter


# Manually Installed Package Imports
from rohan.common.base_navigations      import ThreadedNavigationBase
from rohan.common.logging               import Logger
from rohan.utils.timers                 import IntervalTimer

# Local Imports
from theodwyn.data.writers              import CSVWriter
from theodwyn.navigations.pose_utils    import InferUtils as iut
from theodwyn.navigations.pose_utils    import Bearing
from theodwyn.navigations.mekf_ppt      import MEKF_ppt
from theodwyn.navigations.mekf_ppt      import MEKF_ppt_Dynamics
import theodwyn.navigations.measurement

import pdb 

SelfMEKF    = TypeVar("SelfMEKF", bound = "MEKF")


# TIME_AR             = '{:.0f}'.format( time() )
# HOME_DIR            = os.path.expanduser('~')
# SAVE_DIR            = f'theo_calibrations'
# CSVFOLDER_PATH      = f'{HOME_DIR}/{SAVE_DIR}'
# if not os.path.exists(CSVFOLDER_PATH): 
#     os.makedirs(CSVFOLDER_PATH)
# shutil.rmtree(CSVFOLDER_PATH)
# os.makedirs(CSVFOLDER_PATH)
# CSV_MEKF_FILENAME   = f'{CSVFOLDER_PATH}/mekf_outputs_{TIME_AR}.csv'
# CSV_INFER_FILENAME  = f'{CSVFOLDER_PATH}/infer_outputs_{TIME_AR}.csv'
# MEKF_FIELDNAMES     = 
# INFER_FIELDNAMES    = 
#    self.processing_queue   = Queue(maxsize=MAX_QUEUE_SIZE)
#     self.add_threaded_method( target=self.calibration_data_writer )

# def calibration_data_writer(self):
#     """
#     Threaded process which state and input pairs
#     """
#     with CSVWriter(filename=CSV_FILENAME,fieldnames=CSV_FIELDNAMES) as csv_writer:            
#         while not self.sigterm.is_set():
#             while not self.processing_queue.empty():
#                 set, command, vicon_data = self.processing_queue.get()
#                 csv_writer.write_data( [set, *command, *vicon_data[0], *vicon_data[1] ] )
#             sleep(1)

est_csv_fn          = f"csvs/pe_ests_testing_{strftime('%Y_%m_%d_%H_%M_%S')}.csv"
meas_csv_fn         = f"csvs/pe_meas_testing_{strftime('%Y_%m_%d_%H_%M_%S')}.csv"
infer_csv_fn        = f"csvs/pe_infer_testing_{strftime('%Y_%m_%d_%H_%M_%S')}.csv"


est_csv_headers     = [
                        'timestamp'
                        ,'measurement_flag'
                        , 'dg_x', 'dg_y', 'dg_z' # delta gibbs quaternion
                        , 'w_x', 'w_y', 'w_z' # angular velocity
                        , 'a_x', 'a_y', 'a_z' # angular acceleration
                        , 'x', 'xdot', 'xddot' # position, velocity, acceleration
                        , 'y', 'ydot', 'yddot' # position, velocity, acceleration
                        , 'z', 'zdot', 'zddot' # position, velocity, acceleration
                        , 'global_quat_w', 'global_quat_x', 'global_quat_y', 'global_quat_z' # global quaternion
                        , 'covar_dgx', 'covar_dgy', 'covar_dgz' # covariance of delta gibbs quaternion
                        , 'covar_wx', 'covar_wy', 'covar_wz' # covariance of angular velocity
                        , 'covar_ax', 'covar_ay', 'covar_az' # covariance of angular acceleration
                        , 'covar_x', 'covar_xdot', 'covar_xddot' # covariance of position, velocity, acceleration
                        , 'covar_y', 'covar_ydot', 'covar_yddot' # covariance of position, velocity, acceleration
                        , 'covar_z', 'covar_zdot', 'covar_zddot' # covariance of position, velocity, acceleration
                        ]
meas_csv_headers    = [
                        'timestamp'
                        , 'p_x', 'p_y', 'p_z' # position
                        , 'q_w', 'q_x', 'q_y', 'q_z' # quaternion
                        , 'covar_theta_x', 'covar_theta_y', 'covar_theta_z' # delta theta covariance (local tangent update)
                        , 'covar_px', 'covar_py', 'covar_pz' # covariance of position
                    ]
infer_csv_headers   = [
                        'timestamp'
                        , 'img_fp', 'img_h_pix', 'img_w_pix'
                        , 'box', 'score', 'labels'
                        , 'keypoints'
                    ]

def build_infer_dict(timestamp, img_fp: str, img_h_pix: int, img_w_pix: int, box: NDArray, score: float, labels: List[str], keypoints: NDArray) -> Dict:
    """ Build an inference dictionary """
    return {
            'timestamp' : timestamp
            , 'img_fp'  : img_fp
            , 'img_h_pix': img_h_pix
            , 'img_w_pix': img_w_pix
            , 'box'     : box.tolist()
            , 'score'   : score
            , 'labels'  : labels
            , 'keypoints': keypoints.tolist()
            }
def build_meas_dict(timestamp, p: NDArray, q: NDArray, R: NDArray) -> Dict:
    """ Build a measurement dictionary """
    return {
            'timestamp' : timestamp
            , 'p_x'     : p[0]
            , 'p_y'     : p[1]
            , 'p_z'     : p[2]
            , 'q_w'     : q[0]
            , 'q_x'     : q[1]
            , 'q_y'     : q[2]
            , 'q_z'     : q[3]
            , 'covar_theta_x': R[0,0]
            , 'covar_theta_y': R[1,1]
            , 'covar_theta_z': R[2,2]
            , 'covar_px'     : R[3,3]
            , 'covar_py'     : R[4,4]
            , 'covar_pz'     : R[5,5]
            }

def build_est_dict(timestamp, meas_flag: int, state: NDArray, global_quat: NDArray, covar: NDArray) -> Dict:
    """ Build a filter estimate dictionary """
    return {
            'timestamp' : timestamp
            ,'measurement_flag' : meas_flag
            , 'dg_x' : state[0], 'dg_y' : state[1], 'dg_z' : state[2] # delta gibbs quaternion
            , 'w_x' : state[3], 'w_y' : state[4], 'w_z' : state[5] # angular velocity
            , 'a_x' : state[6], 'a_y' : state[7], 'a_z' : state[8] # angular acceleration
            , 'x' : state[9], 'xdot' : state[10], 'xddot' : state[11] # position, velocity, acceleration
            , 'y' : state[12], 'ydot' : state[13], 'yddot' : state[14] # position, velocity, acceleration
            , 'z' : state[15], 'zdot' : state[16], 'zddot' : state[17] # position, velocity, acceleration
            , 'global_quat_w' : global_quat[0], 'global_quat_x' : global_quat[1], 'global_quat_y' : global_quat[2], 'global_quat_z' : global_quat[3] # global quaternion
            , 'covar_dgx' : covar[0,0], 'covar_dgy' : covar[1,1], 'covar_dgz' : covar[2,2] # covariance of delta gibbs quaternion
            , 'covar_wx' : covar[3,3], 'covar_wy' : covar[4,4], 'covar_wz' : covar[5,5] # covariance of angular velocity
            , 'covar_ax' : covar[6,6], 'covar_ay' : covar[7,7], 'covar_az' : covar[8,8] # covariance of angular acceleration
            , 'covar_x' : covar[9,9], 'covar_xdot' : covar[10,10], 'covar_xddot' : covar[11,11] # covariance of position, velocity, acceleration
            , 'covar_y' : covar[12,12], 'covar_ydot' : covar[13,13], 'covar_yddot' : covar[14,14] # covariance of position, velocity, acceleration
            , 'covar_z' : covar[15,15], 'covar_zdot' : covar[16,16], 'covar_zddot' : covar[17,17] # covariance of position, velocity, acceleration
            }

class MEKF(ThreadedNavigationBase):
    """
    Model for a MEKF
    """
    
    # for logging
    process_name            : str                   = 'MEKF + Inference Meas Model (Threaded)'
    frame_in                : Optional[NDArray]     = None
    frame_in_fp             : Optional[str]         = None
    mekf_prop_timer         : IntervalTimer
    mekf_meas_timer         : IntervalTimer

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
        init_covar          : NDArray               = np.eye(18), # initial covariance, np.eye(18) * 1e-6
        tr_co2cam_cbff      : NDArray               = np.array([0,0,0]), # translation vector from camera body fixed frame to center of chase bff frame
        pose0               : NDArray               = np.array([0, 0, 25, 1, 0, 0, 0]), # initial pose
        img_in_size         : Tuple[int,int]        = (512, 512), # image input size for model
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
        source_mode         : str                   = "camera",
        image_dir           : Optional[str]         = None,
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
        self.omega0     = np.array(omega0) # ensure that the input is a numpy array, can be initialized from json
        self.alpha0     = np.array(alpha0) # ensure that the input is a numpy array, can be initialized from json
        self.init_covar = np.array(init_covar) # ensure that the input is a numpy array, can be initialized from json
        self.max_flip   = max_flip_deg
        self.tau        = tau
        self.Qpsd       = Qpsd
        self.bearing_std= bearing_meas_std
        self.cam_offset = np.array(tr_co2cam_cbff) # ensure that the input is a numpy array, can be initialized from json
        self.pose0      = np.array(pose0) # ensure that the input is a numpy array, can be initialized from json
        self.img_in_size= tuple(img_in_size)
        self.imgnet_norm= imgnet_norm
        self.pad_color  = tuple(pad_color)
        self.box_key    = box_key
        self.label_key  = label_key
        self.score_key  = score_key
        self.kps_key    = kps_key
        self.kscore_key = kscore_key
        self.outkeys    = outkeys
        self.Kmat       = np.array(Kmat) # ensure that the input is a numpy array, can be initialized from json
        self.source_mode= source_mode
        self.image_dir  = image_dir
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
        
        self.logger.write(f"MEKF ThreadedNavigationBase Class Initialized", process_name = self.process_name)
        self.est_csvw   = CSVWriter(filename = est_csv_fn, fieldnames = est_csv_headers)
        self.meas_csvw  = CSVWriter(filename = meas_csv_fn, fieldnames = meas_csv_headers)
        self.inf_csvw   = CSVWriter(filename = infer_csv_fn, fieldnames = infer_csv_headers)
        # open files and write the header
        self.est_csvw.open_file()
        self.meas_csvw.open_file()
        self.inf_csvw.open_file()
        self.logger.write(f"CSV logging to {est_csv_fn}, {meas_csv_fn}, and {infer_csv_fn} initialized", process_name = self.process_name)
        

    def init_navigation(self):
        """ Initialize inference (measurement) model + filter"""
        mod, inputs, outputs    = iut.onnx_model_setup(model_path = self.model_path)
        self.logger.write(f"ONNX Model Loaded: {self.model_path}", process_name = self.process_name)
        self.model              = mod
        self.minput_names       = inputs
        self.moutput_names      = outputs

        self.mekf       = MEKF_ppt(self.mekf_dt, self.Q_std, self.R_std, self.max_flip, self.tau, self.Qpsd)
        self.logger.write(f"MEKF Object Initialized", process_name = self.process_name)
        # measurement fcn setup
        if self.meas_model == 'direct':
            meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_direct(a, b, c, d, e) # fix this
        elif self.meas_model == 'local':
            meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_local(a, b, c, d, e)
        elif self.meas_model == 'c++':
            meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_cpp(a, b, c, d, e)
        self.meas_fcn   = meas_fh
        self.logger.write(f"Measurement Function Set: {self.meas_model}", process_name = self.process_name)
        # example for meas_fcn 
        # start_pose, start_covar = MEKF_ppt_Dynamics.nls_cpp(pose0, azel, kps_3D, bearing_meas_std_rad, tr_co2cam_cbff)

        pass

    def deinit_navigation(self):
        """ Deinitialize inference model """
        self.model  = None #may want to revisit for cleanup with onnxruntime
        if hasattr(self, 'est_csvw'):
            self.est_csvw.close_file()
            self.logger.write(f"MEKF Pose Estimation CSV Closed", process_name = self.process_name)
        if hasattr(self, 'meas_csvw'):
            self.meas_csvw.close_file()
            self.logger.write(f"MEKF Measurement CSV Closed", process_name = self.process_name)
        if hasattr(self, 'inf_csvw'):
            self.inf_csvw.close_file()
            self.logger.write(f"MEKF Inference CSV Closed", process_name = self.process_name)
        pass
        


    def spin_meas_model( self ) -> None:
        """
        Threaded Process: Proceses each measurement with the inference model
        """
        self.logger.write(f"Spinning up Inference Model", process_name = self.process_name)
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
                import cv2
                cv2.imwrite('/home/saa4743/agnc_repos/test320.jpg', self.frame_in)
                img_np_flt      = iut.cv2_preprocess_img_np(
                                                                self.frame_in
                                                                , resize_tuple  = self.img_in_size
                                                                , imagenet_norm = self.imgnet_norm
                                                                , pad_color     = self.pad_color
                                                            ).astype(np.float32)
                # cv2.imwrite('/home/saa4743/agnc_repos/test320_2.jpg', img_np_flt)
                
                img_h, img_w    = img_np_flt.shape[1], img_np_flt.shape[2]
                ort_output_dict = iut.ort_krcnn_inference(self.model, self.minput_names, self.moutput_names, img_np_flt, output_keys = self.outkeys) # infer on frame
                # self.logger.write(f"Frame Inference Completed on {self.frame_in_fp}", process_name = self.process_name)
                try: 
                    
                    ort_sco_max_idx = np.argmax(ort_output_dict[self.score_key])
                    ort_sco_m       = ort_output_dict[self.score_key][ort_sco_max_idx]
                    ort_box_m       = ort_output_dict[self.box_key][ort_sco_max_idx]
                    ort_kps_m       = ort_output_dict[self.kps_key][ort_sco_max_idx]
                    ort_kps_2D      = np.round(ort_kps_m[:, 0:2], 3).astype(np.float32)
                    az_el,_         = Bearing.compute_azimuth_elevation(ort_kps_2D, self.Kmat)
                    self.meas_az_el = az_el
                    self.logger.write(f"Frame Inference Accepted on {self.frame_in_fp}", process_name = self.process_name)
                    # self.logger.write(f"Box: {ort_box_m}, Score: {ort_sco_m}, AzEl: {az_el}", process_name = self.process_name)
                    self.inf_csvw.write_data({
                                                'timestamp' : time()
                                                , 'img_fp'  : self.frame_in_fp
                                                , 'img_h_pix': img_h
                                                , 'img_w_pix': img_w
                                                , 'box'     : ort_box_m.tolist()
                                                , 'score'   : ort_sco_m
                                                , 'labels'  : ort_output_dict[self.label_key]
                                                , 'keypoints': ort_kps_2D.tolist()
                                            })
                    if hasattr(self.inf_csvw, 'file') and self.inf_csvw.file:
                        self.inf_csvw.file.flush()  # flush the file to ensure that the data is written to disk
                    if not self.first_meas_proc.is_set():
                        # if the first measurement is not set and the measurement is not ready, set the first measurement
                        self.is_first_meas.set()
                    else: 
                        self.meas_ready.set()

                except Exception as e:
                    self.skipped += 1
                    fail_str    = f"Inference Failed with Exception: {e} for Image ID {self.frame_in_fp}: total skipped count: {self.skipped}"
                    self.logger.write(fail_str, process_name = self.process_name)
            
            self.frame_in   = None # Let go of the frame after processing (or attempting to process it) #TODO: check this
                
            pass

    
    def spin_filter( self ) -> None:    
        self.logger.write(f"Spinning up Filter", process_name = self.process_name)
        while not self.sigterm.is_set():
            if not self.first_meas_proc.is_set():
                if self.is_first_meas.is_set():
                    # if the first measurement is set, then process first measurement
                    start_pose, start_R     = self.meas_fcn(self.pose0, self.meas_az_el, self.kps3D, self.bearing_std, self.cam_offset)
                    # start_R would be the initial measurement noise covariance matrix but for filiter initialization, we only use starting pose
                    posi0, quat0            = start_pose[:3], start_pose[3:]
                    # update filter initial state
                    self.mekf.set_initial_state_covar(quat0, self.omega0, self.alpha0, posi0, self.init_covar)
                    self.first_meas_proc.set()
                    self.is_first_meas.clear()
                    self.logger.write(f"Filter Initialized and Ready to Run Based on First image: {self.frame_in_fp}", process_name = self.process_name)
                    self.est_csvw.write_data( build_est_dict(time(), -1, self.mekf.state_est, self.mekf.global_quat_est, self.mekf.covar_est) )
                    if hasattr(self.est_csvw, 'file') and self.est_csvw.file:
                        # flush the file to ensure that the data is written to disk
                        self.est_csvw.file.flush()
                    self.meas_csvw.write_data( build_meas_dict(time(), posi0, quat0, start_R) )
                    if hasattr(self.meas_csvw, 'file') and self.meas_csvw.file:
                        self.meas_csvw.file.flush()

                # continue to next iteration if first measurement is set 
                continue 
            
            #####################################TODO: check this
            # reach this code after self.first_meas_proc.is_set() is True
            # considerations --> sync the timeupdate when a measurement is ready?
            self.mekf_prop_timer.await_interval()
            time_update_start   = perf_counter()
            self.mekf.time_update()
            time_update_end     = perf_counter()

            # if measurement is ready and first measurement is processed, then measurement update
            if self.meas_ready.is_set():
                self.mekf_meas_timer.await_interval()
                pose_est        = np.concatenate([self.mekf.position_est, self.mekf.global_quat_est])
                poseJ, covarJ   = self.meas_fcn(pose_est, self.meas_az_el, self.kps3D, self.bearing_std, self.cam_offset)
                posiJ, quatJ    = poseJ[:3], poseJ[3:]
                meas_update_started = perf_counter()
                self.mekf.measurement_update(quatJ, posiJ, covarJ)
                meas_update_ended   = perf_counter()
                self.mekf.mekf_reset()
                self.logger.write(
                                    f"Filter Measurement Update Performed, corresponding pose {poseJ}, "
                                    f"Most Recent Time Update Start & End: {time_update_start}, {time_update_end},"
                                    f"Most Recent Measurement Update Start & End: {meas_update_started}, {meas_update_ended}", 
                                    process_name = self.process_name)
                self.meas_csvw.write_data( build_meas_dict(time(), posiJ, quatJ, covarJ) )
                if hasattr(self.meas_csvw, 'file') and self.meas_csvw.file:
                    self.meas_csvw.file.flush()
                self.meas_ready.clear()  
            
            # record filter estimate
            # in this case, is always false since we clear it after measurement update
            self.est_csvw.write_data( build_est_dict(time(), self.meas_ready.is_set(), self.mekf.state_est, self.mekf.global_quat_est, self.mekf.covar_est) )
            if hasattr(self.est_csvw, 'file') and self.est_csvw.file:
                self.est_csvw.file.flush()

            #####################################TODO: check this
            pass 

    def pass_in_frame( 
        self,
        image : NDArray,
        img_path : Optional[str] = None
    ) -> None:
        """
        Retrieves return code and frame information
        """
        self.frame_in       = image
        self.frame_in_fp    = img_path
        # self.logger.write(f"Frame Acquired: {img_path}", process_name=self.process_name)
        