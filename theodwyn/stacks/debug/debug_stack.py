from rohan.common.base_stacks           import StackBase
from rohan.data.classes                 import StackConfiguration
from rohan.common.logging               import Logger
from theodwyn.networks.adafruit         import Adafruit_PCA9685
from theodwyn.controllers.gamepad       import XboxGamePad 
from theodwyn.cameras.intel_realsense   import D455
from typing                             import Optional, Any
from time                               import sleep, time

class DebugStack(StackBase):

    """
    Stack used for example, testing and verification
    :param config: The stack configuration whose format can be found at .data.classes
    :param freq: The frequency of the control loops
    :param cntrl_factor: Factor relating controller input to servo angle changes
    :param verbose: Verbosity flag indicating whether debugging task should be printed to console
    """

    verbose         : bool
    cntl_period     : float
    cntl_factor     : float


    def __init__( 
        self, 
        config          : StackConfiguration,
        freq            : float = 60.,
        cntrl_factor    : float = 2.,
        verbose         : bool  = False,
    ):
        super().__init__(config=config)
        self.cntl_period    = 1/freq
        self.cntl_factor    = cntrl_factor
        self.verbose        = verbose

    def process( 
        self, 
        network    : Optional[Adafruit_PCA9685]   = None, 
        camera     : Optional[D455]               = None, 
        controller : Optional[XboxGamePad]        = None,
        guidance   : Optional[Any]                = None, 
        navigation : Optional[Any]                = None,
        logger     : Optional[Logger]             = None
    ) -> None:
        
        frame_color, frame_depth = None, None

        if isinstance( camera, D455 ):
            frame_color, frame_depth  = camera.get_frame()

        if isinstance( controller, XboxGamePad ):
            control_input     = controller.determine_control()
            
            if isinstance( network, Adafruit_PCA9685 ):
                r_analog  = control_input[3:5]
                pan_command, tilt_command = network.servokit.servo[0].angle , network.servokit.servo[1].angle
                if abs(r_analog[0]) > 0.2: pan_command += self.cntl_factor * -r_analog[0]
                if abs(r_analog[1]) > 0.2: tilt_command += self.cntl_factor * r_analog[1] 
                command = [ pan_command, tilt_command ]
                network.send( cmd = command )

        if self.verbose : print( f"\
                -> RGB Camera      : {'Online' if frame_color is not None else 'Offline' } \n\
                -> Depth Camera    : {'Online' if frame_depth is not None else 'Offline' } \n\
                -> Control input   : {control_input if isinstance( controller, XboxGamePad ) else 'Offline'} \n\
                -> Servo Commands  : {command if isinstance( controller, XboxGamePad ) and isinstance( network, Adafruit_PCA9685 ) else 'Offline'} \n\
                -> Servo Angles    : {( network.servokit.servo[0].angle,  network.servokit.servo[1].angle ) if isinstance( network, Adafruit_PCA9685 ) else 'Offline'} \n"
        )