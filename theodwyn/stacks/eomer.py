import os
from cv2                                        import imwrite
import numpy                                    as     np
from numpy.linalg                               import norm
from numpy.typing                               import NDArray
from math                                       import sqrt, pi, cos, sin, atan2, degrees
from rohan.common.base_stacks                   import ThreadedStackBase
from rohan.common.logging                       import Logger
from rohan.data.classes                         import StackConfiguration
from theodwyn.cameras.ximea                     import XIMEA
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.networks.event_signal             import SingleEventSignal
from theodwyn.controllers.viconfeedback         import ViconFeedback, VFB_Setpoints
from theodwyn.guidances.presets                 import Preset2DShapes
from theodwyn.data.writers                      import CSVWriter
from theodwyn.utils.rotations                   import Rot_123
from theodwyn.utils.data_format                 import wrap_to_pi
from theodwyn.manipulators.mechanum_wheel_model import Mechanum4Wheels
from theodwyn.navigations.mekf                  import MEKF
from typing                                     import Optional, List, Union
from time                                       import time, sleep, perf_counter
from queue                                      import Queue
from rohan.utils.timers                         import IntervalTimer
from copy                                       import deepcopy
SQRT2O2                 = sqrt(2)/2
DEBUGGING               = False
SINGLE_RUN              = True
HSI_INLOOP              = False

# TODO: PARSE THE FOLLOWING SETTINGS
# >> CALIBRATED MOTOR CONSTANTS
LX                      = 0.165
LY                      = 0.11
WRADIUS                 = 0.075
TS_CONST                = 17.224737008062963 # [rad/s]
EULER_ORIENTATION_INIT  = [ 0., 0., 0. ] # XYZ Sequence
AUTO_THROTTLE_THRES     = 0.7

# >> MANUAL CONTROL PREFERENCES
MAX_THROTTLE                = 0.50
MAX_OMEGA                   = 2*pi/5
SWITCH_COOLDOWN             = 3.
TRIGGER_HOLDTIME            = 1.

# >> VICON PREFERENCES
OBJECT_NAME                 = "eomer"
CAMERA_NAME                 = "eomer_cam"
TARGET_NAME                 = "soho"
INIT_DIST_THRESHOLD         = 0.2       # NOTE: About 8 Inches
INIT_ANGLE_THRESHOLD        = 0.2       # NOTE: About 11 degrees
INIT_CAM_ANGLE_THRESHOLD    = 0.05      # NOTE: About 11 degrees
CAMERA_OFFSET               = np.array( [ 0. ,  0.030 ,  0.010 ] )
TARGET_OFFSET               = np.array( [ 0. ,  0.000 ,  0.100 ] )

# >> DATA RECORD PREFERENCES 
RECORD_OBJECT_1   = "eomer_cam"
RECORD_OBJECT_2   = "soho"
RECORD_OBJECTS    = [RECORD_OBJECT_1, RECORD_OBJECT_2]
MAX_QUEUE_SIZE    = 100

CSV_FIELDNAMES      = [
    "ID",
    "frame_n",
    f"x_mm_{RECORD_OBJECT_1}",
    f"y_mm_{RECORD_OBJECT_1}",
    f"z_mm_{RECORD_OBJECT_1}",
    f"w_{RECORD_OBJECT_1}",
    f"i_{RECORD_OBJECT_1}",
    f"j_{RECORD_OBJECT_1}", 
    f"k_{RECORD_OBJECT_1}",
    f"x_mm_{RECORD_OBJECT_2}",
    f"y_mm_{RECORD_OBJECT_2}",
    f"z_mm_{RECORD_OBJECT_2}",
    f"w_{RECORD_OBJECT_2}",
    f"i_{RECORD_OBJECT_2}",
    f"j_{RECORD_OBJECT_2}", 
    f"k_{RECORD_OBJECT_2}"
]

# >> GUIDANCE STANDBY MODES
GSB_NOT             = "0"
GSB_WAIT_HSI        = "1"
GSB_WAIT_ALGN       = "2"
GSB_WAIT_INFR       = "3"

class EomerStack(ThreadedStackBase):
    """
    Eomer's Stack
    :param config: The stack configuration whose format can be found at .data.classes
    :param spin_intrvl: Inverse-frequency of spinning loop
    :param cntrl_factor: Factor relating controller input to servo angle changes
    """
    process_name    : str = "Eomer Stack"

    verbose         : bool
    cntl_factor     : float
    held_buttons    : List[float]
    holding_buttons : List[bool]
    control_mode    : bool              = False # (False->Manual, True->Auto)
    rotation_matrix : NDArray           = np.identity(2)
    guidance_ready  : bool              = False
    guidance_standby: int               = GSB_NOT
    processing_queue: Queue
    count           : int               = 0
    image_path      : str
    vicon_csv_path  : str 
    vicon_csv_fn    : str
    target_pnt      : Optional[NDArray] = None
    
    def __init__( 
        self, 
        config          : StackConfiguration,
        spin_intrvl     : float         = 1/60,
        cntrl_factor    : float         = 2.,
        image_path      : Optional[str] = None,
        vicon_csv_path  : Optional[str] = None,
        vicon_csv_fn    : str           = "vicon_csv.csv"
    ):
        super().__init__(
            config=config,
            spin_intrvl=spin_intrvl,
        )

        # >> MANUAL CONTROL PREFERENCES
        self.cntl_factor        = cntrl_factor # FOR MANUAL TAKEOVER
        self.held_buttons       = 1*[time(),time()]
        self.holding_buttons    = 1*[False,False] 
        self.control_switch     = IntervalTimer( interval=SWITCH_COOLDOWN )
        self._update_rotation_matrix( EULER_ORIENTATION_INIT )

        # >> INPUT MIXER
        model                   = Mechanum4Wheels(lx=LX,ly=LY,wheel_radius=WRADIUS)
        self.mechanum_ijacob    = model.get_invjacobian()

        # >> CSVs
        self.image_path     = image_path
        self.vicon_csv_path = vicon_csv_path
        self.vicon_csv_fn    = vicon_csv_fn

        # >> Log Onces
        self.logged_startframe = False

        # >> Queues 
        self.processing_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        if not DEBUGGING and (not self.image_path is None) and (not self.vicon_csv_path is None):
            self.add_threaded_method( target=self.data_saving )


    def _update_rotation_matrix( 
        self,
        vicon_orientation : List[float] 
    ):
        def _R3( yaw : float ):
            return np.array([
                [ cos(yaw), -sin(yaw) ],
                [ sin(yaw),  cos(yaw) ]
            ])
        self.rotation_matrix = _R3( vicon_orientation[2] )

    def data_saving(self):
        if not os.path.exists(self.image_path)      : os.makedirs(self.image_path,exist_ok=True)
        if not os.path.exists(self.vicon_csv_path)  : os.makedirs(self.vicon_csv_path,exist_ok=True)
        with CSVWriter(filename=f"{self.vicon_csv_path}/{self.vicon_csv_fn}",fieldnames=CSV_FIELDNAMES) as csv_writer:
            while not self.sigterm.is_set():
                while not self.processing_queue.empty():
                    frame, merged_vicon_data, frame_n = self.processing_queue.get()
                    ret_code = imwrite(
                        f"{self.image_path}/img_{str(self.count+1).zfill(5)}.jpg", 
                        frame
                    )

                    if ret_code:
                        vicon_data_out = [  
                            f"{str(self.count+1).zfill(5)}",
                            frame_n,
                            *merged_vicon_data[0],
                            *merged_vicon_data[1]
                        ]
                        csv_writer.write_data(vicon_data_out)
                        if self.count % 50 == 0:
                            ret_msg = f"{self.count+1} images captured"
                            if self.logger:
                                self.logger.write( ret_msg, process_name=self.process_name)
                        self.count +=1
                sleep(1)

    def process( 
        self, 
        network    : Optional[ List[Union[ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,ViconConnection,SingleEventSignal]] ] = None, 
        camera     : Optional[XIMEA]                                                                                            = None, 
        controller : Optional[ViconFeedback]                                                                                    = None,
        guidance   : Optional[Preset2DShapes]                                                                                   = None, 
        navigation : Optional[MEKF]                                                                                             = None,
        logger     : Optional[Logger]                                                                                           = None
    ) -> None:
        """
        > Networks
            0 : ZMQDish
            1 : ADAfruit PCA
            2 : Sabertooth 2x12
            3 : Sabertooth 2x12
            4 : ViconConnection
            5 : SingleEventSignal
        > Cameras:
              : Ximea
        > Controllers:
              : Vicon Feedback
        > Guidances:
              : CSVInterpreter/Preset2DShape
        > Navigations:
              : MEKF 
        """
        frame, camera_command, wheel_throttles = None, None, [ 0. , 0. , 0. , 0. ]

        # >>>>>>>>>>>>>>>>>>>>>>>> Relative Navigation and Camera Functionality

        if ( self.guidance_standby is GSB_WAIT_INFR ) or self.guidance_ready: 
            frame   = camera.get_frame()
            navigation.pass_in_frame( image = frame, img_cnt = self.count )

        # >>>>>>>>>>>>>>>>>>>>>>>>> Manual Control Functionality

        if network[0]:
            _, xwc_input = network[0].recv()

            if not xwc_input is None:
                # >> Spin-Down Stack from Controller
                if xwc_input[14] > 0.5:
                    if self.holding_buttons[0] is False: 
                        self.holding_buttons[0] = True
                        self.held_buttons[0]    = time()
                    if time() - self.held_buttons[0] > TRIGGER_HOLDTIME: 
                        raise KeyboardInterrupt # >>> Spin down stack through expected interrupt
                else:
                    if self.holding_buttons[0] is True:  self.holding_buttons[0] = False

                # >> Switch to from manual to openloop and visa-versa
                if xwc_input[13] > 0.5:
                    if self.holding_buttons[1] is False: 
                        self.holding_buttons[1] = True
                        self.held_buttons[1]    = time()
                    if time() - self.held_buttons[1] > TRIGGER_HOLDTIME and self.control_switch.check_interval(): 
                        self.control_mode       = not self.control_mode
                        self.guidance_ready     = False
                        self.guidance_standby   = GSB_NOT
                        if guidance: guidance.reset_guidance()
                        if logger:
                            mode = "Manual" if not self.control_mode else "Auto"
                            logger.write( f"Switching Mode to : {mode}", self.process_name )
                else:
                    if self.holding_buttons[1] is True:  self.holding_buttons[1] = False 

                # >> Read Manual Control Inputs
                if not self.control_mode:
                    
                    # >> Control camera from right-analog stick
                    r_analog  = xwc_input[3:5]
                    delta_pan_command, delta_tilt_command = 0.,0.
                    if abs(r_analog[0]) > 0.2: delta_pan_command  = self.cntl_factor * -r_analog[0]
                    if abs(r_analog[1]) > 0.2: delta_tilt_command = self.cntl_factor * r_analog[1] 
                    camera_command = [ delta_pan_command, delta_tilt_command ]
                    
                    # >> Control Base from left-analog stick
                    l_analog     = xwc_input[0:2]
                    bumper_diff  = xwc_input[10] - xwc_input[11]
                    v = np.zeros((3,1)).flatten()
                    if abs(l_analog[0]) > 0.1: v[0] = -WRADIUS * MAX_THROTTLE * l_analog[0]
                    if abs(l_analog[1]) > 0.1: v[1] = -WRADIUS * MAX_THROTTLE * l_analog[1]
                    if abs(bumper_diff) > 0.5: v[2] =  WRADIUS * MAX_OMEGA    * bumper_diff/abs(bumper_diff)
                    if norm(v,2) > WRADIUS * MAX_THROTTLE * SQRT2O2 : 
                        v *= ( WRADIUS * MAX_THROTTLE * SQRT2O2 )/norm(v,2)

                    wheel_throttles  = ( self.mechanum_ijacob @ v ).flatten()

        # >>>>>>>>>>>>>>>>>>>>>>>>> Autonomous Control Functionality

        if self.control_mode and controller and guidance:
            set_points  = VFB_Setpoints()
            
            if HSI_INLOOP and ( self.guidance_standby is GSB_WAIT_HSI ) and network[5]:
                _, data_in = network[5].recv()
                if not data_in is None:
                    self.target_pnt       = np.array([*data_in])
                    self.guidance_standby = GSB_WAIT_ALGN
                    if logger:
                        logger.write(
                            f"Friendly HSI Informed us that the target is around -> {self.target_pnt}",
                            self.process_name
                        )
                        logger.write(
                            f"Autonomous Guidance is now waiting for camera alignment",
                            self.process_name
                        )

            if ( self.guidance_standby is GSB_WAIT_INFR ) and navigation.first_meas_proc.is_set():
                self.guidance_standby   = GSB_NOT
                self.guidance_ready     = True 
                if logger:
                    logger.write(
                        f"Autonomous Guidance starting now",
                        self.process_name
                    )
            
            if not self.guidance_ready:
                if ( self.guidance_standby is GSB_NOT ):
                    guide = guidance.get_init_guidance()
                else:
                    # NOTE: Robots will be sit in standby until guidance wait time has passed
                    guide   = {} 
            else:                       
                guide = guidance.determine_guidance()
            

            if not guide is None:
                pos_xy_vicon, ang_yaw_vicon = None, None
                if 'x' in guide and 'y' in guide     : set_points.pos_xy  = np.array( [ float(guide['x'])  , float(guide['y'])   ] ).flatten()
                if 'v_x' in guide and 'v_y' in guide : set_points.vel_xy  = np.array( [ float(guide['v_x']), float(guide['v_y']) ] ).flatten()
                if 'yaw' in guide                    : set_points.ang_yaw = float(guide['yaw'])
                if 'av_z'in guide                    : set_points.avel_z  = float(guide['av_z'])
                

                if network[4]: 

                    # >>> VICON DATA PULL (1/2) -> For Platform Control
                    vicon_data = network[4].recv_pose( object_name=OBJECT_NAME, ret_quat=False )
                     
                    if frame is not None and not DEBUGGING: # make this change 
                        merged_vicon_data = len(RECORD_OBJECTS)*[None]
                        all_succeeded = True
                        
                        frame_n = -1
                        for i, object_name in enumerate(RECORD_OBJECTS):
                            vicon_data_i            = deepcopy( network[4].recv_pose( object_name=object_name ) )
                            merged_vicon_data[i]    = [ *vicon_data_i.position, *vicon_data_i.orientation_quat ]
                            frame_n                 = vicon_data_i.framenumber      
                            if not vicon_data_i.succeeded: 
                                all_succeeded = False
                                break
                        
                        if all_succeeded:
                            # frame = camera.get_frame()
                            self.processing_queue.put( (frame, merged_vicon_data, frame_n) )

                    # >>> VICON DATA PULL (2/2) -> For Camera Control
                    if (self.guidance_standby is GSB_WAIT_ALGN) or self.guidance_ready: 
                        cam_vicon_data  = network[4].recv_pose( object_name=CAMERA_NAME, ret_quat=False )

                        trgt_data_available = False
                        if HSI_INLOOP and (not self.guidance_ready):
                            if not ( self.target_pnt is None ):
                                trgt_data_available = True
                                target_pos          = self.target_pnt
                        else:
                            trgt_vicon_data = network[4].recv_pose( object_name=TARGET_NAME, ret_quat=False )
                            if trgt_vicon_data.succeeded:
                                trgt_data_available = True
                                target_pos          = trgt_vicon_data.position + Rot_123(*trgt_vicon_data.orientation_euler).T @ TARGET_OFFSET
                        

                        if cam_vicon_data.succeeded and trgt_data_available:
                            cam_rotm        = Rot_123(*cam_vicon_data.orientation_euler)
                            cam_cp_pos      = cam_vicon_data.position + cam_rotm.T @ CAMERA_OFFSET
                            cam2trgt_vff    = target_pos - cam_cp_pos
                            camz_vff        = ( cam_rotm.T @ np.array( [0.,0.,1.] ) ).flatten()
                            cam2trgt_vff_plnorm, camz_vff_plnorm = norm( cam2trgt_vff[0:2] ), norm( camz_vff[0:2] )
                            set_points.ang_dpt = np.array(
                                [
                                    wrap_to_pi( atan2( cam2trgt_vff[1], cam2trgt_vff[0] )     - atan2( camz_vff[1], camz_vff[0] ) ),
                                    wrap_to_pi( atan2( cam2trgt_vff[2], cam2trgt_vff_plnorm ) - atan2( camz_vff[2], camz_vff_plnorm ) )
                                ]
                            )
                            if self.guidance_standby is GSB_WAIT_ALGN:
                                if norm( set_points.ang_dpt, 2 ) < INIT_CAM_ANGLE_THRESHOLD:
                                    self.guidance_standby   = GSB_WAIT_INFR
                                    if logger:
                                        logger.write(
                                            f"Autonomous Guidance is now waiting for first inference",
                                            self.process_name
                                        )

                    # >>> DETERMINE GUIDANCE MODES
                    if vicon_data.succeeded: 
                        self._update_rotation_matrix( vicon_orientation=vicon_data.orientation_euler )
                        pos_xy_vicon    = 1E-3*np.array(vicon_data.position[0:2]).flatten()
                        ang_yaw_vicon   = vicon_data.orientation_euler[2]

                        if ( self.guidance_standby is GSB_NOT ) and ( not self.guidance_ready ):
                            dist_2init = norm( set_points.pos_xy - 1E-3*np.array(vicon_data.position[0:2]), 2 ) 
                            dang_2init = abs( wrap_to_pi( set_points.ang_yaw - vicon_data.orientation_euler[2] ) ) 
                            if dist_2init < INIT_DIST_THRESHOLD and dang_2init < INIT_ANGLE_THRESHOLD: 
                                if HSI_INLOOP: 
                                    self.guidance_standby = GSB_WAIT_HSI
                                    if logger:
                                        logger.write(
                                            f"Autonomous Guidance is now waiting for HSI",
                                            self.process_name
                                        )
                                else: 
                                    self.guidance_standby = GSB_WAIT_ALGN

                        if not self.logged_startframe: 
                            self.logged_startframe = True    
                            if logger: 
                                ret_msg = f"<sync> -> frame: {vicon_data.framenumber} , time: {perf_counter()}"
                                logger.write(
                                    ret_msg,
                                    self.process_name
                                )

                # >>> DETERMINE VICON FEEDBACK FROM VICON PULLS 
                v_cmd, dpt_cmd = controller.determine_control(
                    pos_xy_vicon    = pos_xy_vicon,
                    ang_yaw_vicon   = ang_yaw_vicon,
                    set_points      = set_points
                )
            
                # ----------------------------------------------------------------------------------
                # NOTE: The following swap is an artifact of the mixing matrix mechanum_ijacob
                #       being altered for the controller input. This functionality is flawed and 
                #       is a known issue that will be later addressed
                # ----------------------------------------------------------------------------------
                v_cmd[0], v_cmd[1]  = v_cmd[1], v_cmd[0]            
                v_cmd[0:2]          = ( self.rotation_matrix @ v_cmd[0:2] ).flatten()
                wheel_speeds        = ( self.mechanum_ijacob @ v_cmd ).flatten() 
                wheel_throttles     = wheel_speeds / TS_CONST
                if not dpt_cmd is None:
                    camera_command  = [ degrees(dpt_cmd[0]), -degrees(dpt_cmd[1])  ]
            
            elif guide is None:
                self.guidance_ready   = False
                self.guidance_standby = GSB_NOT
                if guidance                         : guidance.reset_guidance()
                if SINGLE_RUN                       : self.sigterm.set() 

        # >>>>>>>>>>>>>>>>>>>>>>>>> Send Throttle and Camera Commands Functionality

        if camera_command is not None and network[1]:
            network[1].send( 
                cmd = [ 
                    camera_command[0] + network[1].servokit.servo[0].angle,
                    camera_command[1] + network[1].servokit.servo[1].angle,
                ] 
            )

        if wheel_throttles is not None and network[2] and network[3]:
            for throttle in wheel_throttles:
                if abs(throttle) > AUTO_THROTTLE_THRES:
                    wheel_throttles *= AUTO_THROTTLE_THRES/abs(throttle)
                    if logger:
                        logger.write( 
                            "Commanded wheel throttle - {:2f} exceeds threshold - {:.2f}. Scaling down ...".format(throttle,AUTO_THROTTLE_THRES), 
                            process_name=self.process_name 
                        )
            network[2].send_throttles( [ wheel_throttles[0], wheel_throttles[2] ] ) 
            network[3].send_throttles( [ wheel_throttles[3], wheel_throttles[1] ] ) 

             
