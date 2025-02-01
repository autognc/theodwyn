from rohan.common.base_stacks           import StackBase
from rohan.data.classes                 import StackConfiguration
from theodwyn.networks.adafruit         import Adafruit_PCA9685
from theodwyn.networks.comm_prot        import ZMQDish
from theodwyn.cameras.intel_realsense   import D455
from typing                             import Optional, List, Union, Any
from time                               import time

SWITCH_COOLDOWNS  = [ 1. ]
TRIGGER_HOLDTIME  = [ 2. ]
class DebugCommStack(StackBase):
    """
    Stack used for example, testing and verification of communications
    :param config: The stack configuration whose format can be found at .data.classes
    :param spin_intrvl: Inverse-frequency of spinning loop
    :param cntrl_factor: Factor relating controller input to servo angle changes
    :param verbose: Verbosity flag indicating whether debugging task should be printed to console
    """
    process_name    : str = "Debug Communications Stack"

    verbose         : bool
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
        verbose         : bool  = False,
    ):
        super().__init__(
            config=config,
            spin_intrvl=spin_intrvl
        )
        self.cntl_factor    = cntrl_factor
        self.verbose        = verbose

        self.switches       = 1*[int(False)]
        self.last_switches  = 1*[time()]
        self.held_buttons   = 1*[time()]
        self.holding_buttons= 1*[False] 

    def process( 
        self, 
        network    : Optional[ List[Union[ZMQDish,Adafruit_PCA9685]] ]  = None, 
        camera     : Optional[D455]                                     = None, 
        controller : Optional[Any]                                      = None
    ) -> None:

        frame_color, frame_depth = None, None
        if isinstance( camera, D455 ):
            frame_color, frame_depth  = camera.get_frame()
            
        if isinstance( network[0], ZMQDish ):
            topic, control_input = network[0].recv()

            if control_input is not None:

                # Switch Stream Channels
                if control_input[10] > 0.5 or control_input[11] > 0.5: 
                    if time() - self.last_switches[0] > SWITCH_COOLDOWNS[0]:
                        camera.switch_channel()
                        self.last_switches[0] = time()

                # Spin-Down Stack from Controller
                if control_input[14] > 0.5:
                    if self.holding_buttons[0] is False: 
                        self.holding_buttons[0] = True
                        self.held_buttons[0]    = time()
                    if time() - self.held_buttons[0] > TRIGGER_HOLDTIME[0]: 
                        raise KeyboardInterrupt
                else:
                    if self.holding_buttons[0] is True:  self.holding_buttons[0] = False

                if isinstance( network[1], Adafruit_PCA9685 ):
                    r_analog  = control_input[3:5]
                    pan_command, tilt_command = network[1].servokit.servo[0].angle , network[1].servokit.servo[1].angle
                    if abs(r_analog[0]) > 0.2: pan_command += self.cntl_factor * -r_analog[0]
                    if abs(r_analog[1]) > 0.2: tilt_command += self.cntl_factor * r_analog[1] 
                    command = [ pan_command, tilt_command ]
                    network[1].send( cmd = command )

        if self.verbose : print( f"\
                -> RGB Camera      : {'Online' if frame_color is not None else 'Offline' } \n\
                -> Depth Camera    : {'Online' if frame_depth is not None else 'Offline' } \n\
                -> Wireless Topic  : {topic if isinstance( network[0], ZMQDish ) else 'Offline' } \n\
                -> Servo Commands  : {command if control_input is not None and isinstance( network[1], Adafruit_PCA9685 ) else 'Offline'} \n\
                -> Servo Angles    : {( network[1].servokit.servo[0].angle,  network[1].servokit.servo[1].angle ) if isinstance( network[1], Adafruit_PCA9685 ) else 'Offline'} \n" 
        )