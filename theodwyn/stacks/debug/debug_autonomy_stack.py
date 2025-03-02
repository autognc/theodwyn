import numpy                                    as     np
from math                                       import sqrt
from rohan.common.base_stacks                   import StackBase
from rohan.data.classes                         import StackConfiguration
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.cameras.intel_realsense           import D455
from theodwyn.manipulators.mechanum_wheel_model import Mechanum4Wheels
from typing                                     import Optional, List, Union, Any
from time                                       import time

SWITCH_COOLDOWNS  = [ 1. ]
TRIGGER_HOLDTIME  = [ 2. ]
MAX_THROTTLE      = 0.50
SQRT2O2           = sqrt(2)/2
ROBOT_NAME        = "eomer"
EI_SWITCH         = int(False)
class DebugAutoStack(StackBase):
    """
    Stack used for example, testing and verification of communications
    :param config: The stack configuration whose format can be found at .data.classes
    :param spin_intrvl: Inverse-frequency of spinning loop
    :param cntrl_factor: Factor relating controller input to servo angle changes
    :param verbose: Verbosity flag indicating whether debugging task should be printed to console
    """
    process_name    : str = "Debug Communications Stack"

    cntl_factor     : float
    switches        : List[int]
    last_switches   : List[float] 
    held_buttons    : List[float]
    holding_buttons : List[bool]

    def __init__( 
        self, 
        config          : StackConfiguration,
        spin_intrvl     : float = 1/60,
        cntrl_factor    : float = 2.,
    ):
        super().__init__(
            config=config,
            spin_intrvl=spin_intrvl
        )
        self.cntl_factor    = cntrl_factor

        self.switches       = 1*[int(False)]
        self.last_switches  = 1*[time()]
        self.held_buttons   = 1*[time()]
        self.holding_buttons= 1*[False] 

        model               = Mechanum4Wheels(lx=0.165,ly=0.11,wheel_radius=0.075)
        self.mechanum_ijacob= model.get_invjacobian() * model.wheel_radius

    def process( 
        self, 
        network    : Optional[ List[Union[ViconConnection,Adafruit_PCA9685,SabertoothSimpleSerial]] ]   = None, 
        camera     : Optional[D455]                                                                     = None, 
        controller : Optional[Any]                                                                      = None
    ) -> None:

        frame_color, frame_depth = None, None
        if isinstance( camera, D455 ):
            frame_color, frame_depth  = camera.get_frame()
            
        if len(network)>0 and isinstance( network[0], ZMQDish ):
            topic, external_input = network[0].recv()

            external_input = None # TODO: something
            if external_input is not None:

                # Switch Stream Channels
                if external_input[10] > 0.5 or external_input[11] > 0.5: 
                    if time() - self.last_switches[0] > SWITCH_COOLDOWNS[0]:
                        if isinstance( camera, D455 ): camera.switch_channel()
                        self.last_switches[0] = time()

                # Spin-Down Stack from Controller
                if external_input[14] > 0.5:
                    if self.holding_buttons[0] is False: 
                        self.holding_buttons[0] = True
                        self.held_buttons[0]    = time()
                    if time() - self.held_buttons[0] > TRIGGER_HOLDTIME[0]: 
                        raise KeyboardInterrupt
                else:
                    if self.holding_buttons[0] is True:  self.holding_buttons[0] = False

                ##########################################################################
                # TODO Switch to manual commands                                         #
                ##########################################################################
                
                if not EI_SWITCH:
                    if len(network) > 2 and isinstance( network[2], Adafruit_PCA9685 ):
                        r_analog  = external_input[3:5]
                        pan_command, tilt_command = network[2].servokit.servo[0].angle , network[2].servokit.servo[1].angle
                        if abs(r_analog[0]) > 0.2: pan_command += self.cntl_factor * -r_analog[0]
                        if abs(r_analog[1]) > 0.2: tilt_command += self.cntl_factor * r_analog[1] 
                        command = [ pan_command, tilt_command ]
                        network[2].send( cmd = command )

                    if len(network)>4 and isinstance( network[3], SabertoothSimpleSerial ) and isinstance( network[4], SabertoothSimpleSerial ):
                        l_analog = external_input[0:2]
                        v = np.zeros((3,1)).flatten()
                        if abs(l_analog[0]) > 0.1: v[0] = -MAX_THROTTLE * l_analog[0]
                        if abs(l_analog[1]) > 0.1: v[1] = -MAX_THROTTLE * l_analog[1]
                        if np.linalg.norm(v[0:2],2) > MAX_THROTTLE * SQRT2O2 : 
                            v[0:2] *= ( MAX_THROTTLE * SQRT2O2 )/np.linalg.norm(v[0:2],2)
                        throttles  = ( self.mechanum_ijacob @ v ).flatten()
                        sbrth1_cmd = [ throttles[0], throttles[2] ]
                        sbrth2_cmd = [ throttles[3], throttles[1] ]
                        network[3].send_throttles( sbrth1_cmd ) 
                        network[4].send_throttles( sbrth2_cmd ) 


        if EI_SWITCH and len(network) > 1 and isinstance( network[1], ViconConnection):
            vicon_data = network[1].recv_pose( object_name=ROBOT_NAME )
            
            if vicon_data.succeeded : 
                # TODO: Process vicon data 
                pass

            if len(network) > 2 and isinstance( network[2], Adafruit_PCA9685 ):
                # TODO: Autonomous Camera Commands
                # r_analog  = external_input[3:5]
                # pan_command, tilt_command = network[2].servokit.servo[0].angle , network[2].servokit.servo[1].angle
                # if abs(r_analog[0]) > 0.2: pan_command += self.cntl_factor * -r_analog[0]
                # if abs(r_analog[1]) > 0.2: tilt_command += self.cntl_factor * r_analog[1] 
                # command = [ pan_command, tilt_command ]
                # network[2].send( cmd = command )
                pass

            if len(network)>4 and isinstance( network[3], SabertoothSimpleSerial ) and isinstance( network[4], SabertoothSimpleSerial ):
                # TODO: Autonomous Drive Commands
                # l_analog = external_input[0:2]
                # v = np.zeros((3,1)).flatten()
                # if abs(l_analog[0]) > 0.1: v[0] = -MAX_THROTTLE * l_analog[0]
                # if abs(l_analog[1]) > 0.1: v[1] = -MAX_THROTTLE * l_analog[1]
                # if np.linalg.norm(v[0:2],2) > MAX_THROTTLE * SQRT2O2 : 
                #     v[0:2] *= ( MAX_THROTTLE * SQRT2O2 )/np.linalg.norm(v[0:2],2)
                # throttles  = ( self.mechanum_ijacob @ v ).flatten()
                # sbrth1_cmd = [ throttles[0], throttles[2] ]
                # sbrth2_cmd = [ throttles[3], throttles[1] ]
                # network[3].send_throttles( sbrth1_cmd ) 
                # network[4].send_throttles( sbrth2_cmd ) 
                pass