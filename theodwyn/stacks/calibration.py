import os
import numpy                                    as     np
from math                                       import sqrt, pi
from rohan.common.base_stacks                   import ThreadedStackBase
from rohan.common.logging                       import Logger
from rohan.data.classes                         import StackConfiguration
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.data.writers                      import CSVWriter
from theodwyn.manipulators.mechanum_wheel_model import Mechanum4Wheels
from typing                                     import Optional, List, Union, Any
from time                                       import time, sleep
from queue                                      import Queue
from rohan.utils.timers                         import IntervalTimer
SQRT2O2                     = sqrt(2)/2

# TODO: PARSE THE FOLLOWING SETTINGS
# >> SAVE PREFERENCES
TIME_AR             = "{:.0f}".format( time() )
HOME_DIR            = os.path.expanduser("~")
SAVE_DIR            = f"theo_calibrations"
CSVFOLDER_PATH      = f"{HOME_DIR}/{SAVE_DIR}"
if not os.path.exists(CSVFOLDER_PATH): 
    os.makedirs(CSVFOLDER_PATH,exist_ok=True)
CSV_FILENAME        = f"{CSVFOLDER_PATH}/calibration_{TIME_AR}.csv" 
CSV_FIELDNAMES      = ["set","cmd_0","cmd_1","cmd_2","cmd_3","x","y","z","w","i", "j","k"]

# >> CALIBRATION PREFERENCES
CALIBRATION_THROTTLES       = [0.10,-0.10,0.25,-0.25]
CALIBRATION_SETTIME         = 5.

# >> MANUAL CONTROL PREFERENCES
MAX_THROTTLE                = 0.50
MAX_OMEGA                   = 2*pi/5
SWITCH_COOLDOWN             = 1.
TRIGGER_HOLDTIME            = 2.

# >> VICON AND DATA LOG PREFERENCES
OBJECT_NAME                 = "eomer"
MAX_QUEUE_SIZE              = 1000

class CalibrationStack(ThreadedStackBase):
    """
    Stack used for collecting data for calibrating the Eomer system parameters
    :param config: The stack configuration whose format can be found at .data.classes
    :param spin_intrvl: Inverse-frequency of spinning loop
    :param cntrl_factor: Factor relating controller input to servo angle changes
    """
    process_name    : str = "Debug Image Collection Stack"

    verbose         : bool
    cntl_factor     : float
    held_buttons    : List[float]
    holding_buttons : List[bool]
    control_mode    : bool  = False # (False->Manual, True->Auto)

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

        self.calibration_set    = -1
        self.calibration_timer  = IntervalTimer( interval=CALIBRATION_SETTIME )


        # >> MANUAL CONTROL PREFERENCES
        self.cntl_factor        = cntrl_factor # FOR MANUAL TAKEOVER
        self.held_buttons       = 1*[time(),time()]
        self.holding_buttons    = 1*[False,False] 
        self.control_switch     = IntervalTimer( interval=SWITCH_COOLDOWN )

        # >> INPUT MIXER
        model                   = Mechanum4Wheels(lx=0.165,ly=0.11,wheel_radius=0.075)
        self.mechanum_ijacob    = model.get_invjacobian() * model.wheel_radius

        # >> PROCESS SAVING CALIBRATION DATA
        self.processing_queue   = Queue(maxsize=MAX_QUEUE_SIZE)
        self.add_threaded_method( target=self.calibration_data_writer )


    def calibration_data_writer(self):
        """
        Threaded process which state and input pairs
        """
        with CSVWriter(filename=CSV_FILENAME,fieldnames=CSV_FIELDNAMES) as csv_writer:            
            while not self.sigterm.is_set():
                while not self.processing_queue.empty():
                    set, command, vicon_data = self.processing_queue.get()
                    csv_writer.write_data( [set, *command, *vicon_data[0], *vicon_data[1] ] )
                sleep(1)


    def process( 
        self, 
        network    : Optional[ List[Union[ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,ViconConnection]] ]   = None, 
        camera     : Optional[Any]                                                                              = None, 
        controller : Optional[Any]                                                                              = None,
        guidance   : Optional[Any]                                                                              = None, 
        navigation : Optional[Any]                                                                              = None,
        logger     : Optional[Logger]                                                                           = None
    ) -> None:
        """
        > Networks
            0 : ZMQDish
            1 : ADAfruit PCA
            2 : Sabertooth 2x12
            3 : Sabertooth 2x12
            4 : ViconConnection
        > Cameras:
        > Controllers:
        """

        camera_command, wheel_throttles = None, None

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
                        self.sigterm.set()
                else:
                    if self.holding_buttons[0] is True:  self.holding_buttons[0] = False

                # >> Switch to from manual to openloop and visa-versa
                if xwc_input[13] > 0.5:
                    if self.holding_buttons[1] is False: 
                        self.holding_buttons[1] = True
                        self.held_buttons[1]    = time()
                    if time() - self.held_buttons[1] > TRIGGER_HOLDTIME and self.control_switch.check_interval(): 
                        self.control_mode = not self.control_mode
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
                    if abs(l_analog[0]) > 0.1: v[0] = -MAX_THROTTLE * l_analog[0]
                    if abs(l_analog[1]) > 0.1: v[1] = -MAX_THROTTLE * l_analog[1]
                    if abs(bumper_diff) > 0.5: v[2] =  MAX_OMEGA    * bumper_diff/abs(bumper_diff)
                    if np.linalg.norm(v,2) > MAX_THROTTLE * SQRT2O2 : 
                        v *= ( MAX_THROTTLE * SQRT2O2 )/np.linalg.norm(v,2)
                    wheel_throttles  = ( self.mechanum_ijacob @ v ).flatten()
    
        if self.control_mode:
            if self.calibration_set < len(CALIBRATION_THROTTLES):
                if not self.calibration_timer.check_interval():
                    wheel_throttles = [ 
                        -CALIBRATION_THROTTLES[self.calibration_set],
                         CALIBRATION_THROTTLES[self.calibration_set],
                         CALIBRATION_THROTTLES[self.calibration_set],
                        -CALIBRATION_THROTTLES[self.calibration_set]
                    ]                        
                else: 
                    self.calibration_set += 1
            else:
                wheel_throttles = 4*[0.]
                self.sigterm.set() # >> Spindown stack

        # >> Send Camera Command
        if camera_command is not None and network[1]:
            network[1].send( 
                cmd = [ 
                    camera_command[0] + network[1].servokit.servo[0].angle,
                    camera_command[1] + network[1].servokit.servo[1].angle,
                ] 
            )

        # >> Send Throttle Command
        if wheel_throttles is not None and network[2] and network[3]:
            network[2].send_throttles( [ wheel_throttles[0], wheel_throttles[2] ] ) 
            network[3].send_throttles( [ wheel_throttles[3], wheel_throttles[1] ] ) 

            if network[4]:
                vicon_data = network[4].recv_pose( object_name=OBJECT_NAME )

                if vicon_data.succeeded:
                    vicon_position      = vicon_data.position
                    vicon_orientation   = vicon_data.orientation_quat
                    vicon_data          = (vicon_position, vicon_orientation)
                    c_set               = self.calibration_set if self.control_mode else "M"
                    self.processing_queue.put( (c_set, wheel_throttles, vicon_data) )                    
