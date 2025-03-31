# import os
# import shutil
import threading
import numpy                            as np
import cv2                              as cv
from copy                               import copy
from typing                             import TypeVar, Optional, Tuple, Dict, List
from numpy.typing                       import NDArray
from time                               import perf_counter, sleep
from queue                              import Queue

# Manually Installed Package Imports
from rohan.common.base_navigations      import ThreadedNavigationBase
from rohan.common.logging               import Logger
from rohan.utils.timers                 import IntervalTimer

# Local Imports
from theodwyn.data.writers              import CSVWriter
from theodwyn.navigations.pose_utils    import InferUtils as iut
from theodwyn.navigations.pose_utils    import Bearing
from theodwyn.navigations.pose_utils    import Camera
from theodwyn.navigations.pose_utils    import PnP
from theodwyn.navigations.pose_utils    import Projection
from theodwyn.navigations.mekf_ppt      import MEKF_ppt
from theodwyn.navigations.mekf_ppt      import MEKF_ppt_Dynamics

# import pdb
# import traceback


# TIME_AR             = '{:.0f}'.format( perf_counter() )
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

MAX_CSV_SAVING_QUEUE    = 1000
MAX_PROJ_SAVING_QUEUE   = 100
CSV_EST_ID              = '0'
CSV_MEAS_ID             = '1'
CSV_INF_ID              = '2'

ONNX_WARMUPS            = 10

# project inferences if projection path is set
INF_PROJ_COLOR_BGR      = (0, 0, 255) # red
PNP_PROJ_COLOR_BGR      = (57, 21, 57) # deep purple 
NLS_PROJ_COLOR_BGR      = (128, 0, 128) # purple
ORIGIN_PROJ_COLOR_BGR   = (255, 255, 255) # white

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
                        , 'img_id', 'img_input_h_pix', 'img_input_w_pix', 'img_inf_h_pix', 'img_inf_w_pix'
                        , 'box', 'score', 'labels'
                        , 'keypoints', 'az_el_radians'
                    ]

def build_infer_dict(
                        timestamp, img_id: str
                        , img_input_h_pix: int, img_input_w_pix: int, img_inf_h_pix : int, img_inf_w_pix : int, 
                        box: NDArray, score: float, labels: List[str], keypoints: NDArray, az_el: NDArray
                        ) -> Dict:
    """ Build an inference dictionary """
    return {
            'timestamp' : timestamp
            , 'img_id'  : img_id
            , 'img_input_h_pix': img_input_h_pix
            , 'img_input_w_pix': img_input_w_pix
            , 'img_inf_h_pix': img_inf_h_pix
            , 'img_inf_w_pix': img_inf_w_pix
            , 'box'     : box.tolist()
            , 'score'   : score
            , 'labels'  : labels
            , 'keypoints': keypoints.tolist()
            , 'az_el_radians' : az_el.tolist()
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

SelfMEKF    = TypeVar("SelfMEKF", bound = "MEKF")
class MEKF(ThreadedNavigationBase):
    """
    Model for a MEKF
    """
    
    # for logging
    process_name            : str                   = 'MEKF + Inference Meas Model (Threaded)'
    frame_in                : Optional[NDArray]     = None
    frame_id                : int
    mekf_prop_timer         : IntervalTimer
    mekf_meas_timer         : IntervalTimer
    est_csvw                : Optional[CSVWriter]   = None
    meas_csvw               : Optional[CSVWriter]   = None
    inf_csvw                : Optional[CSVWriter]   = None
    csv_saving_queue        : Optional[Queue]       = None
    proj_saving_queue       : Optional[Queue]       = None 

    def __init__(
        self,
        kps3D_path          : str, 
        kps_scale           : float,
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
        Kmat                : NDArray               = None, # always check
        fl_mm               : float                 = None, # focal length in mm
        sw_mm               : float                 = None, # sensor width in mm
        sh_mm               : float                 = None, # sensor height in mm
        pnp_flag            : int                   = 1, # whether to use pnp to refine azimuth and elevation measurements
        rnd_dig             : int                   = 3, # round digits 
        skpped_count        : int                   = 0,
        source_mode         : str                   = "camera",
        proj_path           : Optional[str]         = None,
        image_dir           : Optional[str]         = None,
        est_csv_fn          : Optional[str]         = None,
        meas_csv_fn         : Optional[str]         = None,
        infer_csv_fn        : Optional[str]         = None,
        logger              : Optional[Logger]      = None,
        **config_kwargs
    ):
        ThreadedNavigationBase.__init__( 
            self,
            logger = logger
        )
        # self.kps3D     = np.load(os.path.abspath(kps3D_path))[:num_kps]
        self.kps3D      = np.load(kps3D_path)[:num_kps] * kps_scale
        self.kps3D_orig = np.vstack([np.zeros((1,3)), self.kps3D])
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
        self.fl_mm      = fl_mm
        self.sw_mm      = sw_mm
        self.sh_mm      = sh_mm
        self.pnp_flag   = pnp_flag
        self.rnd_dig    = rnd_dig
        
        self.source_mode= source_mode
        self.image_dir  = image_dir
        
        self.skipped    = skpped_count
        if proj_path is not None and proj_path != "":
            self.proj               = True
            self.proj_saving_queue  = Queue(maxsize=MAX_PROJ_SAVING_QUEUE) 
        else:
            self.proj           = False
        self.proj_path          = proj_path 

        # thread.Event() works as follows: on -> .set() | off -> .clear() | check -> .is_set()
        self.is_first_meas      = threading.Event() # flag for first measurement
        self.first_meas_proc    = threading.Event() # whether the first measurement has been processed
        self.meas_ready         = threading.Event() # flag for measurement ready

        self.mekf_prop_timer    = IntervalTimer( interval = self.mekf_dt )
        self.mekf_meas_timer    = IntervalTimer( interval = self.meas_dt) 

        # initialize the threaded model
        self.add_threaded_method(target = self.spin_filter )
        self.add_threaded_method(target = self.spin_meas_model)
        if self.proj:
            self.add_threaded_method(target = self.spin_projection)
        
        if self.logger:
            self.logger.write(f"MEKF ThreadedNavigationBase Class Initialized", process_name = self.process_name)

        # >>> CSV Writers
        if est_csv_fn or meas_csv_fn or infer_csv_fn:
            if est_csv_fn   : self.est_csvw   = CSVWriter(filename = est_csv_fn, fieldnames = est_csv_headers)
            if meas_csv_fn  : self.meas_csvw  = CSVWriter(filename = meas_csv_fn, fieldnames = meas_csv_headers)
            if infer_csv_fn : self.inf_csvw   = CSVWriter(filename = infer_csv_fn, fieldnames = infer_csv_headers)
            self.csv_saving_queue   = Queue(maxsize=MAX_CSV_SAVING_QUEUE)
            self.add_threaded_method( target=self.csv_saving )

        if self.logger  : self.logger.write(f"CSV logging to {est_csv_fn}, {meas_csv_fn}, and {infer_csv_fn} initialized", process_name = self.process_name)
        

    def init_navigation(self):
        """ Initialize inference (measurement) model + filter"""
        mod, inputs, outputs    = iut.onnx_model_setup(model_path = self.model_path)
        if self.logger:
            self.logger.write(f"ONNX Model Loaded: {self.model_path}", process_name = self.process_name)
        self.model              = mod
        self.minput_names       = inputs
        self.moutput_names      = outputs

        self.mekf       = MEKF_ppt(self.mekf_dt, self.Q_std, self.R_std, self.max_flip, self.tau, self.Qpsd)
        if self.logger:
            self.logger.write(f"MEKF Object Initialized", process_name = self.process_name)
        # measurement fcn setup
        if self.meas_model == 'direct':
            # meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_direct(a, b, c, d, e) # fix this
            meas_fh = MEKF_ppt_Dynamics.nls_direct
        elif self.meas_model == 'local':
            # meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_local(a, b, c, d, e)
            meas_fh = MEKF_ppt_Dynamics.nls_local
        elif self.meas_model == 'c++':
            # meas_fh = lambda a, b, c, d, e: MEKF_ppt_Dynamics.nls_cpp(a, b, c, d, e)
            meas_fh = MEKF_ppt_Dynamics.nls_cpp
        self.meas_fcn   = meas_fh
        if self.logger:
            self.logger.write(f"Measurement Function Set: {self.meas_model}", process_name = self.process_name)
        # example for meas_fcn 
        # start_pose, start_covar = MEKF_ppt_Dynamics.nls_cpp(pose0, azel, kps_3D, bearing_meas_std_rad, tr_co2cam_cbff)


        if self.logger:
            self.logger.write(
                "Warming up onnxruntime inference. Please wait ... ",
                process_name=self.process_name
            )
            warm_up_s = perf_counter()

        for _ in range(ONNX_WARMUPS):
            _ = iut.ort_krcnn_inference(
                self.model, 
                self.minput_names, 
                self.moutput_names, 
                np.zeros( (3,*self.img_in_size) , dtype=np.float32 ), 
                output_keys = self.outkeys
            ) 

        if self.logger:
            warm_up_e = perf_counter()
            self.logger.write(
                f"Warmup for onnxruntime inference was completed ({warm_up_e-warm_up_s} s)",
                process_name=self.process_name
            )

        # open files and write the header
        if self.est_csvw    : self.est_csvw.open_file()
        if self.meas_csvw   : self.meas_csvw.open_file()
        if self.inf_csvw    : self.inf_csvw.open_file()


    def deinit_navigation(self):
        """ Deinitialize inference model """
        self.model  = None #may want to revisit for cleanup with onnxruntime
        if self.est_csvw:
            self.est_csvw.close_file()
            if self.logger: self.logger.write(f"MEKF Pose Estimation CSV Closed", process_name = self.process_name)
        if self.meas_csvw:
            self.meas_csvw.close_file()
            if self.logger: self.logger.write(f"MEKF Measurement CSV Closed", process_name = self.process_name)
        if self.inf_csvw:
            self.inf_csvw.close_file()
            if self.logger: self.logger.write(f"MEKF Inference CSV Closed", process_name = self.process_name)


    def spin_meas_model( self ) -> None:
        """
        Threaded Process: Proceses each measurement with the inference model
        """
        if self.logger:
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
            frame_proc, frame_id = None, None
            with self._instance_lock:
                if self.frame_in is not None:
                    frame_proc = self.frame_in.copy()   
                    frame_id   = copy(self.frame_id)


            if not frame_proc is None:
                
                img_input_h, img_input_w    = frame_proc.shape[0], frame_proc.shape[1]
                if self.proj:
                    img_rgb_inf, img_bgr_proj   = iut.cv2_preprocess_img_np(
                                                                            frame_proc
                                                                            , resize_tuple  = self.img_in_size
                                                                            , imagenet_norm = self.imgnet_norm
                                                                            , pad_color     = self.pad_color
                                                                            , return_bgr    = True
                                                                        )
                    img_rgb_inf     = img_rgb_inf.astype(np.float32) 
                else: 
                    img_rgb_inf     = iut.cv2_preprocess_img_np(
                                                                    frame_proc
                                                                    , resize_tuple  = self.img_in_size
                                                                    , imagenet_norm = self.imgnet_norm
                                                                    , pad_color     = self.pad_color
                                                                ).astype(np.float32)

                # latest pose estimate
                pose_est                = np.concatenate([copy(self.mekf.position_est), copy(self.mekf.global_quat_est)]) 
                # img_rgb_inf is C x H x W
                img_inf_h, img_inf_w    = img_rgb_inf.shape[1], img_rgb_inf.shape[2]
                
                infer_start             = perf_counter()
                # infer on frame
                ort_output_dict         = iut.ort_krcnn_inference(self.model, self.minput_names, self.moutput_names, img_rgb_inf, output_keys = self.outkeys)
                
                infer_end               = perf_counter()
                infer_time              = infer_end - infer_start
                if self.logger:
                    self.logger.write(f"Inferenced in {infer_time} seconds on Frame ID {frame_id}", process_name = self.process_name)
                if not self.first_meas_proc.is_set() and np.max(ort_output_dict[self.score_key]) < 0.9:
                    if self.logger:
                        self.logger.write(f"Frame Inference Rejected on {frame_id}", process_name = self.process_name)
                    with self._instance_lock: self.frame_in   = None
                    continue
                # measurement loop
                try:
                    # get highest score box prediction, index corresponding box, keypoints, and labels
                    ort_sco_max_idx = np.argmax(ort_output_dict[self.score_key])
                    ort_sco_m       = ort_output_dict[self.score_key][ort_sco_max_idx]
                    ort_box_m       = ort_output_dict[self.box_key][ort_sco_max_idx]
                    ort_kps_m       = ort_output_dict[self.kps_key][ort_sco_max_idx]
                    ort_kps_2D      = np.round(ort_kps_m[:, 0:2], self.rnd_dig).astype(np.float32)
                    ort_kps_2D_int  = np.round(ort_kps_2D).astype(np.int32)
                    inf_str         = f'Inference for {frame_id}: Box: {ort_box_m}, Score: {ort_sco_m:.4f}'
                    # recalculate camera matrix based on processed image for inference
                    Kmat_inf        = Camera.camera_matrix(img_inf_w, img_inf_h, self.sw_mm, self.sh_mm, self.fl_mm)

                    # if pnp_flag is set, use pnp to refine azimuth and elevation measurements
                    if self.pnp_flag:
                        try:
                            tr_pnp, q_pnp, _= PnP.ransac_pnp_solve(kps_3D = self.kps3D, kps_2D = ort_kps_2D, camera_matrix = Kmat_inf)
                        except Exception as e:
                            if self.logger:
                                self.logger.write(
                                    "PNP failed with the following exception: {e}",
                                    process_name=self.process_name
                                )
                            tr_pnp, q_pnp   = PnP.pnp_solve(kps_3D = self.kps3D, kps_2D = ort_kps_2D, camera_matrix = Kmat_inf)
                        self.pnp_tr = tr_pnp
                        self.pnp_q  = q_pnp
                        ort_kps_2D  = Projection.project_keypoints(q = q_pnp, r = tr_pnp, K = Kmat_inf, keypoints = self.kps3D)    
                    # calculate azimuth and elevation and store in self
                    inf_az_el, _    = Bearing.compute_azimuth_elevation(ort_kps_2D, Kmat_inf)
                    self.meas_az_el = inf_az_el

                    if self.proj:
                        proj_inf_bbox      = ort_box_m
                        proj_inf_kps_2D    = ort_kps_2D_int
                        proj_inf_img_bgr   = img_bgr_proj
                        proj_img_idx       = frame_id
                        pose_proj, _       = self.meas_fcn(pose_est, self.meas_az_el, self.kps3D, self.bearing_std, self.cam_offset)
                        q_proj, tr_proj    = pose_proj[3:], pose_proj[:3]
                        # project 3D keypoints to 2D
                        proj_nls_kps_2D    = Projection.project_keypoints(
                                                                            q = q_proj
                                                                            , r = tr_proj
                                                                            , K = Kmat_inf
                                                                            , keypoints = self.kps3D_orig
                                                                            )
                        
                        proj_pnp_kps_2D = None
                        if self.pnp_flag:
                                proj_pnp_kps_2D= Projection.project_keypoints(
                                                                                q = q_pnp
                                                                                , r = tr_pnp
                                                                                , K = Kmat_inf
                                                                                , keypoints = self.kps3D_orig
                                                                                )
                        if self.proj_saving_queue:
                            self.proj_saving_queue.put(
                                (
                                    proj_inf_img_bgr, 
                                    proj_inf_bbox, 
                                    proj_inf_kps_2D, 
                                    proj_nls_kps_2D, 
                                    proj_pnp_kps_2D, 
                                    proj_img_idx
                                )
                            )
    

                    if self.logger:
                        self.logger.write(f"Frame Inference Accepted on {frame_id}", process_name = self.process_name)
                        self.logger.write(inf_str, process_name = self.process_name)
                    if self.inf_csvw and self.csv_saving_queue: 
                        self.csv_saving_queue.put( 
                            (
                                CSV_INF_ID, 
                                {
                                    'timestamp' : perf_counter()
                                    , 'img_id'  : "image_" + str(frame_id).zfill(5)
                                    , 'img_input_h_pix': img_input_h
                                    , 'img_input_w_pix': img_input_w
                                    , 'img_inf_h_pix': img_inf_h
                                    , 'img_inf_w_pix': img_inf_w
                                    , 'box'     : np.round(ort_box_m).tolist()
                                    , 'score'   : ort_sco_m
                                    , 'labels'  : ort_output_dict[self.label_key]
                                    , 'keypoints': ort_kps_2D_int.tolist()
                                    , 'az_el_radians' : inf_az_el.tolist()
                                }
                            ) 
                        )
                        
                    if not self.first_meas_proc.is_set():
                        # if the first measurement is not set and the measurement is not ready, set the first measurement
                        self.is_first_meas.set()
                    else: 
                        self.meas_ready.set()

                except Exception as e:
                    self.skipped += 1
                    # fail_str1   = f"Inference Failed with Exception: {e}"
                    # fail_str2   = f"Failed Inference for Image ID {frame_id}: total skipped count: {self.skipped}"
                    #print(fail_str1), print(fail_str2)
                    # traceback.print_exc()
                    if self.logger:
                        self.logger.write( f"Inference Failed with Exception: {e}", process_name = self.process_name)
                        self.logger.write( f"Failed Inference for Image ID {frame_id}: total skipped count: {self.skipped}", process_name = self.process_name)
            
            if frame_proc is not None:
                with self._instance_lock: self.frame_in   = None # Let go of the frame after processing (or attempting to process it) #TODO: check this
                

    def spin_filter( self ) -> None:    
        if self.logger:
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
                    if self.logger:
                        self.logger.write(f"Filter initialized with an image", process_name = self.process_name)
                    
                    if self.est_csvw and self.csv_saving_queue: 
                        self.csv_saving_queue.put( 
                            (
                                CSV_EST_ID, 
                                build_est_dict(perf_counter(), -1, self.mekf.state_est, self.mekf.global_quat_est, self.mekf.covar_est)
                            ) 
                        )
                    if self.meas_csvw and self.csv_saving_queue: 
                        self.csv_saving_queue.put( 
                            (
                                CSV_MEAS_ID, 
                                build_meas_dict(perf_counter(), posi0, quat0, start_R)
                            ) 
                        )

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
                if self.pnp_flag:
                    self.mekf.measurement_update(copy(self.pnp_q), copy(self.pnp_tr), covarJ)
                else:
                    self.mekf.measurement_update(quatJ, posiJ, covarJ)
 
                meas_update_ended   = perf_counter()
                self.mekf.mekf_reset()
                if self.logger:
                    time_between    = meas_update_ended - time_update_end
                    self.logger.write(
                                        f"Meas updated (time between {time_between} s), pose: {poseJ}, "
                                        # f"Most Recent Time Update Start & End: {time_update_start}, {time_update_end},"
                                        # f"Most Recent Measurement Update Start & End: {meas_update_started}, {meas_update_ended}", 
                                        , process_name = self.process_name)
                if self.meas_csvw and self.csv_saving_queue: 
                    self.csv_saving_queue.put( 
                        (
                            CSV_MEAS_ID, 
                            build_meas_dict(perf_counter(), posiJ, quatJ, covarJ)
                        ) 
                    )
                self.meas_ready.clear()  
            
            # record filter estimate
            # in this case, is always false since we clear it after measurement update
            if self.est_csvw and self.csv_saving_queue: 
                self.csv_saving_queue.put( 
                    (
                        CSV_EST_ID, 
                        build_est_dict(perf_counter(), self.meas_ready.is_set(), self.mekf.state_est, self.mekf.global_quat_est, self.mekf.covar_est)
                    ) 
                )

            #####################################TODO: check this    

    def csv_saving(self):
        if self.csv_saving_queue:
            while not self.sigterm.is_set():
                while not self.csv_saving_queue.empty():
                    csv_id, data = self.csv_saving_queue.get()
                    if csv_id is CSV_EST_ID:
                        if self.est_csvw  : self.est_csvw.write_data(data=data)
                    elif csv_id is CSV_MEAS_ID:
                        if self.meas_csvw : self.meas_csvw.write_data(data=data)
                    elif csv_id is CSV_INF_ID:
                        if self.inf_csvw  : self.inf_csvw.write_data(data=data)
                    else:
                        if self.logger:
                            self.logger.write(
                                "CSV ID provided -> {csv_id}, is not a known identifier. Continuing ...",
                                process_name=self.process_name
                            )
                sleep(1) # >>> Let go of resources for a little if nothing is going on


    def spin_projection(self) -> None:
        """
        Threaded Process: Projects 3D keypoints to 2D
        """
        
        if self.proj_saving_queue and self.proj:
            while not self.sigterm.is_set():
                while not self.proj_saving_queue.empty():
                    
                    proj_inf_img_bgr, proj_inf_bbox, proj_inf_kps_2D, proj_nls_kps_2D, proj_pnp_kps_2D, proj_img_idx = self.proj_saving_queue.get()

                    proj_inf_bbox_kps       = Projection.project_bbox_kps_array_2cv2np(
                                                                                        proj_inf_img_bgr
                                                                                        , box = proj_inf_bbox
                                                                                        , keypoints = proj_inf_kps_2D
                                                                                        , keypoint_color = INF_PROJ_COLOR_BGR
                                                                                        , circle_thickness = -1
                                                                                        , circle_size = 2
                                                                                    )
                    add_nls_kps             = Projection.project_bbox_kps_array_2cv2np(
                                                                                        proj_inf_bbox_kps
                                                                                        , box = None 
                                                                                        , keypoints = proj_nls_kps_2D
                                                                                        , keypoint_color = NLS_PROJ_COLOR_BGR
                                                                                        , circle_thickness = 2
                                                                                        , circle_size = 6
                                                                                        , origin_color = ORIGIN_PROJ_COLOR_BGR
                                                                                        , origin_flag = True 
                                                                                        , origin_size = 3
                                                                                        , origin_thickness = -1
                                                                                    )
                    img_nls_fn              = f"{self.proj_path}/inf_image_{proj_img_idx}_nls_proj.png" 
                    cv.imwrite(img_nls_fn, add_nls_kps)
                    if self.logger:
                        self.logger.write(f"Projected Inference and NLS Keypoints to {img_nls_fn}", process_name = self.process_name)

                    if self.pnp_flag and (not proj_pnp_kps_2D is None):
                        add_pnp_kps             = Projection.project_bbox_kps_array_2cv2np(
                                                                                            proj_inf_bbox_kps.copy()
                                                                                            , box = None 
                                                                                            , keypoints = proj_pnp_kps_2D
                                                                                            , keypoint_color = PNP_PROJ_COLOR_BGR
                                                                                            , circle_thickness = 2
                                                                                            , circle_size = 6
                                                                                            , origin_color = ORIGIN_PROJ_COLOR_BGR
                                                                                            , origin_flag = True 
                                                                                            , origin_size = 3
                                                                                            , origin_thickness = -1
                                                                                        )
                        img_pnp_fn              = f"{self.proj_path}/inf_image_{proj_img_idx}_pnp_proj.png"
                        cv.imwrite(img_pnp_fn, add_pnp_kps)
                        if self.logger:
                            self.logger.write(f"Projected PNP Keypoints to {img_pnp_fn}", process_name = self.process_name)

                sleep(1) # >>> Let go of resources for a little if nothing is going on

    def pass_in_frame( 
        self,
        image : NDArray,
        img_cnt : int
    ) -> None:
        """
        Retrieves return code and frame information
        """
        with self._instance_lock:
            self.frame_in       = image
            self.frame_id       = img_cnt
        # if self.logger:
        #     self.logger.write(f"Frame Acquired: {self.frame_id}", process_name=self.process_name)
            
