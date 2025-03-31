import threading
import cv2
import os
import sys
import numpy                                    as     np
from math                                       import sqrt, pi
from rohan.common.base_stacks                   import ThreadedStackBase
from rohan.common.logging                       import Logger
from rohan.data.classes                         import StackConfiguration
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.cameras.ximea                     import XIMEA
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.data.writers                      import CSVWriter
from theodwyn.manipulators.mechanum_wheel_model import Mechanum4Wheels
from typing                                     import Optional, List, Union, Any
from time                                       import time, strftime
from queue                                      import Queue
from rohan.utils.timers                         import IntervalTimer
from copy                                       import deepcopy

TIME_AR     = strftime("%Y-%m-%d_%H-%M-%S")
HOME_DIR    = os.path.expanduser("~")
SAVE_DIR    = f"eomer_usb/run_{TIME_AR}"
IMAGE_PATH1 = f"{HOME_DIR}/{SAVE_DIR}/MC"
IMAGE_PATH2 = f"{HOME_DIR}/{SAVE_DIR}/SC"
VICON_PATH1 = f"{HOME_DIR}/{SAVE_DIR}/MC"
VICON_PATH2 = f"{HOME_DIR}/{SAVE_DIR}/SC"

SWITCH_COOLDOWNS  = [ 1. ]
TRIGGER_HOLDTIME  = [ 2. ]
MAX_THROTTLE      = 0.50
MAX_OMEGA         = 2*pi/5 # rad/s
SQRT2O2           = sqrt(2)/2
VICON_OBJ1      = "eomer_cam"
VICON_OBJ2      = "soho"
VICON_OBJ = [VICON_OBJ1, VICON_OBJ2]
MAX_QUEUE_SIZE    = 100

# DATA Collection constants
MULTI_IMAGE_FOLDER  = f"{IMAGE_PATH1}/MC_Img_Data"
SINGLE_IMAGE_FOLDER = f"{IMAGE_PATH2}/SC_Img_Data"
MULTI_VICON_FOLDER  = f"{VICON_PATH1}/MC_Vicon_Data"
SINGLE_VICON_FOLDER = f"{VICON_PATH2}/SC_Vicon_Data"
if not os.path.exists(MULTI_IMAGE_FOLDER):  os.makedirs(MULTI_IMAGE_FOLDER,exist_ok=True)
if not os.path.exists(SINGLE_IMAGE_FOLDER): os.makedirs(SINGLE_IMAGE_FOLDER,exist_ok=True)
if not os.path.exists(MULTI_VICON_FOLDER):  os.makedirs(MULTI_VICON_FOLDER,exist_ok=True)
if not os.path.exists(SINGLE_VICON_FOLDER): os.makedirs(SINGLE_VICON_FOLDER,exist_ok=True)
CSV_FILENAME_MC    = f"{MULTI_VICON_FOLDER}/vicon_mc_{TIME_AR}.csv" 
CSV_FILENAME_SC    = f"{SINGLE_VICON_FOLDER}/vicon_sc_{TIME_AR}.csv"
CSV_FIELDNAMES_MC  = [
    "Set",
    "ID",
    f"x_mm_{VICON_OBJ1}",
    f"y_mm_{VICON_OBJ1}",
    f"z_mm_{VICON_OBJ1}",
    f"w_{VICON_OBJ1}",
    f"i_{VICON_OBJ1}",
    f"j_{VICON_OBJ1}", 
    f"k_{VICON_OBJ1}",
    f"x_mm_{VICON_OBJ2}",
    f"y_mm_{VICON_OBJ2}",
    f"z_mm_{VICON_OBJ2}",
    f"w_{VICON_OBJ2}",
    f"i_{VICON_OBJ2}",
    f"j_{VICON_OBJ2}", 
    f"k_{VICON_OBJ2}"
]
CSV_FIELDNAMES_SC  = CSV_FIELDNAMES_MC[1:]


class DebugImColl(ThreadedStackBase):
    """
    Stack used for example, testing and verification of image collection
    :param config: The stack configuration whose format can be found at .data.classes
    :param spin_intrvl: Inverse-frequency of spinning loop
    :param cntrl_factor: Factor relating controller input to servo angle changes
    :param verbose: Verbosity flag indicating whether debugging task should be printed to console
    """
    process_name    : str = "Debug Image Collection Stack"

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
            spin_intrvl=spin_intrvl,
        )
        self.cntl_factor    = cntrl_factor
        self.verbose        = verbose     
        self.switches       = 1*[int(False)]
        self.last_switches  = 1*[time()]
        self.held_buttons   = 1*[time()]
        self.holding_buttons= 1*[False] 

        model               = Mechanum4Wheels(lx=0.165,ly=0.11,wheel_radius=0.075)
        self.mechanum_ijacob= model.get_invjacobian() * model.wheel_radius

        self.process_switch   = IntervalTimer( interval=SWITCH_COOLDOWNS[0] )
        self.capture_flag     = threading.Event()
        self.capture_switch   = threading.Event()
        self.processing_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.add_threaded_method( target=self.data_saving )


    def data_saving(self):
        with    CSVWriter(filename=CSV_FILENAME_MC,fieldnames=CSV_FIELDNAMES_MC) as csv_writer_mc, \
                CSVWriter(filename=CSV_FILENAME_SC,fieldnames=CSV_FIELDNAMES_SC) as csv_writer_sc:
            
            count_single, count_multi, set = 0, 0, 0
            capture_flag_switched_on  = False
            while not self.sigterm.is_set():
                while not self.processing_queue.empty():
                    frame, merged_vicon_data = self.processing_queue.get()

                    if not capture_flag_switched_on and self.capture_flag.is_set():
                        capture_flag_switched_on = True
                        set+=1
                        count_multi = 0
                    elif capture_flag_switched_on and not self.capture_flag.is_set():
                        capture_flag_switched_on = False

                    if self.capture_flag.is_set():

                        if not self.capture_switch.is_set():
                            # SAVE IMAGE AND CSV
                            multi_image_captured = cv2.imwrite(f"{MULTI_IMAGE_FOLDER}/set_{str(set).zfill(5)}_ximea_mc_{str(count_multi).zfill(5)}.png", frame)
                            vicon_data_out = [  
                                str(set).zfill(5),
                                f"{str(count_multi).zfill(5)}",
                                *merged_vicon_data[0],
                                *merged_vicon_data[1]
                            ]
                            csv_writer_mc.write_data(vicon_data_out)

                            # Log Confirmation
                            if multi_image_captured:  
                                ret_msg = f"Image ({set},{count_multi}) captured"
                            else:
                                ret_msg = f"Image set not captured"
                            self.logger.write( ret_msg, process_name=self.process_name)

                            count_multi +=1
                            
                        else:
                            self.capture_switch.clear() 
                            self.capture_flag.clear()

                            single_image_captured = cv2.imwrite(f"{SINGLE_IMAGE_FOLDER}/ximea_sc_{str(count_single).zfill(5)}.png", frame)
                            vicon_data_out = [  
                                f"{str(count_single).zfill(5)}",
                                *merged_vicon_data[0],
                                *merged_vicon_data[1]
                            ]
                            csv_writer_sc.write_data(vicon_data_out)

                            if single_image_captured:  
                                ret_msg = f"Image {count_single} captured"
                            else:
                                ret_msg = f"Image set not captured"
                            self.logger.write( ret_msg, process_name=self.process_name)
                            count_single +=1

    def process( 
        self, 
        network    : Optional[ List[Union[ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,ViconConnection]] ]   = None, 
        camera     : Optional[XIMEA]                                                                            = None, 
        controller : Optional[Any]                                                                              = None,
        guidance   : Optional[Any]                                                                              = None, 
        navigation : Optional[Any]                                                                              = None,
        logger     : Optional[Logger]                                                                           = None
    ) -> None:
        
        def get_vicondata() -> None:
            merged_data = len(VICON_OBJ)*[None]
            succeeded = True
            for i,object_name in enumerate(VICON_OBJ):
                vicon_data      = deepcopy( network[4].recv_pose( object_name=object_name ) )
                merged_data[i]  = [ *vicon_data.position, *vicon_data.orientation_quat ]
                if succeeded and not vicon_data.succeeded: 
                    succeeded = False
                    break
            return merged_data, succeeded


        if self.capture_flag.is_set() and camera: # and isinstance( camera, XIMEA ):
            frame = camera.get_frame()

            if len(network)>4 and network[4]: # isinstance( network[4], ViconConnection ):
                merged_vicon_data, succeeded = get_vicondata()
                if succeeded: 
                    self.processing_queue.put( (frame, merged_vicon_data) )                    

        if len(network)>0 and network[0]: # isinstance( network[0], ZMQDish ):
            _, control_input = network[0].recv()

            if control_input is None:
                # NOTE: Need to stop the motors if network connection is lost
                if len(network)>2 and isinstance( network[2], SabertoothSimpleSerial ):
                    network[2].send( int(0).to_bytes(1,byteorder=sys.byteorder) ) 
                if len(network)>3 and isinstance( network[3], SabertoothSimpleSerial ):
                    network[3].send( int(0).to_bytes(1,byteorder=sys.byteorder) ) 

            else:

                # Spin-Down Stack from Controller
                if control_input[14] > 0.5:
                    if self.holding_buttons[0] is False: 
                        self.holding_buttons[0] = True
                        self.held_buttons[0]    = time()
                    if time() - self.held_buttons[0] > TRIGGER_HOLDTIME[0]: 
                        self.sigterm.set()
                else:
                    if self.holding_buttons[0] is True:  self.holding_buttons[0] = False

                # Multi-Image Capture
                if control_input[5] > 0.5 and self.process_switch.check_interval():
                    self.capture_flag.clear() if self.capture_flag.is_set() else self.capture_flag.set()

                # Single Image Capture
                if control_input[2] > 0.5 and self.process_switch.check_interval():
                    self.capture_flag.set()
                    self.capture_switch.set()
                
                # Controller
                if len(network)>1 and network[1]: # isinstance( network[1], Adafruit_PCA9685 ):
                    r_analog  = control_input[3:5]
                    pan_command, tilt_command = network[1].servokit.servo[0].angle , network[1].servokit.servo[1].angle
                    if abs(r_analog[0]) > 0.2: pan_command += self.cntl_factor * -r_analog[0]
                    if abs(r_analog[1]) > 0.2: tilt_command += self.cntl_factor * r_analog[1] 
                    command = [ pan_command, tilt_command ]
                    network[1].send( cmd = command )

                if len(network)>3 and network[2] and network[3]: # isinstance( network[2], SabertoothSimpleSerial ) and isinstance( network[3], SabertoothSimpleSerial ):
                    l_analog     = control_input[0:2]
                    bumper_diff  = control_input[10] - control_input[11]
                    v = np.zeros((3,1)).flatten()
                    if abs(l_analog[0]) > 0.1: v[0] = -MAX_THROTTLE * l_analog[0]
                    if abs(l_analog[1]) > 0.1: v[1] = -MAX_THROTTLE * l_analog[1]
                    if abs(bumper_diff) > 0.5: v[2] =  MAX_OMEGA    * bumper_diff/abs(bumper_diff)
                    if np.linalg.norm(v,2) > MAX_THROTTLE * SQRT2O2 : 
                        v *= ( MAX_THROTTLE * SQRT2O2 )/np.linalg.norm(v,2)
                    throttles  = ( self.mechanum_ijacob @ v ).flatten()
                    sbrth1_cmd = [ throttles[0], throttles[2] ]
                    sbrth2_cmd = [ throttles[3], throttles[1] ]
                    network[2].send_throttles( sbrth1_cmd ) 
                    network[3].send_throttles( sbrth2_cmd ) 
