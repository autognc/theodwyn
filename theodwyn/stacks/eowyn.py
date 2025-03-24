import numpy                                    as     np
from numpy.linalg                               import norm
from numpy.typing                               import NDArray
from math                                       import sqrt, pi, cos, sin
from rohan.common.base_stacks                   import ThreadedStackBase
from rohan.common.logging                       import Logger
from rohan.data.classes                         import StackConfiguration
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.controllers.viconfeedback         import ViconFeedback, VFB_Setpoints
from theodwyn.guidances.presets                 import Preset2DShapes
from theodwyn.utils.data_format                 import wrap_to_pi
from theodwyn.manipulators.mechanum_wheel_model import Mechanum4Wheels
from typing                                     import Optional, List, Union, Any
from time                                       import time, sleep
from rohan.utils.timers                         import IntervalTimer
from theodwyn.stacks.eomer                      import LX, LY, WRADIUS, TS_CONST, AUTO_THROTTLE_THRES
SQRT2O2                 = sqrt(2)/2
DEBUGGING               = False

# TODO: PARSE THE FOLLOWING SETTINGS
# >> CALIBRATED MOTOR CONSTANTS
EULER_ORIENTATION_INIT  = [ 0., 0., -pi ] # XYZ Sequence
AUTO_THROTTLE_THRES     = 0.7

# >> MANUAL CONTROL PREFERENCES
MAX_THROTTLE                = 0.50
MAX_OMEGA                   = 2*pi/5
SWITCH_COOLDOWN             = 3.
TRIGGER_HOLDTIME            = 1.

# >> VICON PREFERENCES
OBJECT_NAME                 = "eowyn"
INIT_DIST_THRESHOLD         = 0.2      # NOTE: About 8 Inches
INIT_ANGLE_THRESHOLD        = 0.2      # NOTE: About 11 degrees

class EowynStack(ThreadedStackBase):
    """
    Eomer's Stack
    :param config: The stack configuration whose format can be found at .data.classes
    :param spin_intrvl: Inverse-frequency of spinning loop
    :param cntrl_factor: Factor relating controller input to servo angle changes
    """
    process_name    : str = "Debug Image Collection Stack"

    verbose         : bool
    cntl_factor     : float
    held_buttons    : List[float]
    holding_buttons : List[bool]
    control_mode    : bool    = False # (False->Manual, True->Auto)
    rotation_matrix : NDArray = np.identity(2)
    guidance_ready  : bool    = False

    def __init__( 
        self, 
        config          : StackConfiguration,
        spin_intrvl     : float = 1/60,
        cntrl_factor    : float = 2.,
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

        # >> Log Onces
        self.logged_startframe = False


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


    def process( 
        self, 
        network    : Optional[ List[Union[ZMQDish,SabertoothSimpleSerial,ViconConnection]] ]   = None, 
        camera     : Optional[Any]                                                                              = None, 
        controller : Optional[ViconFeedback]                                                                    = None,
        guidance   : Optional[Preset2DShapes]                                                                   = None, 
        navigation : Optional[Any]                                                                              = None,
        logger     : Optional[Logger]                                                                           = None
    ) -> None:
        """
        > Networks
            0 : ZMQDish
            1 : Sabertooth 2x12
            2 : Sabertooth 2x12
            3 : ViconConnection
        > Cameras:
            None
        > Controllers:
              : Vicon Feedback
        """

        wheel_throttles = [ 0. , 0. , 0. , 0. ]

        # >>>>>>>>>>>>>>>>>>>>>>>>> Manual Control Functionality

        if network[0]:
            _, xwc_input = network[0].recv()

            if not self.control_mode and xwc_input is None:
                # >> Stop/Slow motors
                wheel_throttles = [ 0. , 0. , 0. , 0. ]

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
                        self.control_mode = not self.control_mode
                        if self.guidance_ready: 
                            self.guidance_ready = False
                            if guidance: guidance.reset_guidance()
                        if logger:
                            mode = "Manual" if not self.control_mode else "Auto"
                            logger.write( f"Switching Mode to : {mode}", self.process_name )
                else:
                    if self.holding_buttons[1] is True:  self.holding_buttons[1] = False 

                # >> Read Manual Control Inputs
                if not self.control_mode:
                    
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
            if not self.guidance_ready: guide = guidance.get_init_guidance()
            else:                       guide = guidance.determine_guidance()
                

            if not guide is None:
                if 'x' in guide and 'y' in guide     : set_points.pos_xy  = np.array( [ float(guide['x'])  , float(guide['y'])   ] ).flatten()
                if 'v_x' in guide and 'v_y' in guide : set_points.vel_xy  = np.array( [ float(guide['v_x']), float(guide['v_y']) ] ).flatten()
                if 'yaw' in guide                    : set_points.ang_yaw = float(guide['yaw'])

                if network[3]: # >>> Try to pull object data from vicon system

                    # >>> Vicon Pulls (1/2) -> For Platform Control
                    vicon_data = network[3].recv_pose( object_name=OBJECT_NAME, ret_quat=False )

                    if vicon_data.succeeded: 
                        self._update_rotation_matrix( vicon_orientation=vicon_data.orientation_euler )
                        v_cmd, _ = controller.determine_control( 
                            pos_xy_vicon    = 1E-3*np.array(vicon_data.position[0:2]).flatten(), 
                            ang_yaw_vicon   = vicon_data.orientation_euler[2],
                            set_points      = set_points 
                        )

                        if ( not self.guidance_ready ):
                            dist_2init = norm( np.array( [ guide['x'], guide['y'] ] ) - 1E-3*np.array(vicon_data.position[0:2]), 2 ) 
                            dang_2init = abs( wrap_to_pi( guide['yaw'] - vicon_data.orientation_euler[2] ) ) 
                            if dist_2init < INIT_DIST_THRESHOLD and dang_2init < INIT_ANGLE_THRESHOLD:
                                self.guidance_ready = True

                        if not self.logged_startframe: 
                            self.logged_startframe = True    
                            if logger: 
                                ret_msg = f"<sync> -> frame: {vicon_data.framenumber} , time: {guidance.get_guidance_time()}"
                                logger.write(
                                    ret_msg,
                                    self.process_name
                                )

                    else: # >> Otherwise only allows for feedforward control
                        v_cmd, _ = controller.determine_control(set_points=set_points) 
                        
                else: # >> Otherwise only allows for feedforward control
                        v_cmd, _ = controller.determine_control(set_points=set_points) 

            else: # >> Otherwise only allows for feedforward control
                v_cmd, _ = controller.determine_control(set_points=set_points)                 
            
            # ----------------------------------------------------------------------------------
            # NOTE: The following swap is an artifact of the mixing matrix mechanum_ijacob
            #       being altered for the controller input. This functionality is flawed and 
            #       is a known issue that will be later addressed
            # ----------------------------------------------------------------------------------

            v_cmd[0], v_cmd[1] = v_cmd[1], v_cmd[0]
            
            v_cmd[0:2]      = ( self.rotation_matrix @ v_cmd[0:2] ).flatten()
            wheel_speeds    = ( self.mechanum_ijacob @ v_cmd ).flatten() 
            wheel_throttles = wheel_speeds / TS_CONST

        # >>>>>>>>>>>>>>>>>>>>>>>>> Send Throttle and Camera Commands Functionality

        if wheel_throttles is not None and network[1] and network[2]:
            for throttle in wheel_throttles:
                if abs(throttle) > AUTO_THROTTLE_THRES:
                    wheel_throttles *= AUTO_THROTTLE_THRES/abs(throttle)
                    if logger:
                        logger.write( 
                            "Commanded wheel throttle - {:2f} exceeds threshold - {:.2f}. Scaling down ...".format(throttle,AUTO_THROTTLE_THRES), 
                            process_name=self.process_name 
                        )
            network[1].send_throttles( [ wheel_throttles[0], wheel_throttles[2] ] ) 
            network[2].send_throttles( [ wheel_throttles[3], wheel_throttles[1] ] ) 

             
