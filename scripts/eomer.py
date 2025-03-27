import shutil
import os
import board
import json
from numpy                                      import array, diag, identity
from math                                       import pi
from theodwyn.cameras.ximea                     import XIMEA
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.networks.event_signal             import SingleEventSignal
from theodwyn.controllers.viconfeedback         import ViconFeedback
# from theodwyn.guidances.presets                 import Preset2DShapes, PRESET_CIRCLE
from theodwyn.guidances.file_interpreters       import CSVInterpreter
from theodwyn.navigations.mekf                  import MEKF
from theodwyn.stacks.eomer                      import EomerStack
from rohan.data.classes                         import StackConfiguration
from time import sleep, strftime
DEBUGGING   = True

if __name__ == "__main__":
    TIME_AR             = strftime('%Y_%m_%d_%H_%M_%S')
    HOME_DIR            = os.path.expanduser("~")
    USB_DIR             = f"{HOME_DIR}/eomer_usb"
    SAVE_DIR            = f"{USB_DIR}/run_{TIME_AR}"
    IMAGE_FOLDER        = f"{SAVE_DIR}/images"
    VICON_FOLDER        = f"{SAVE_DIR}"

    if not os.path.exists(f"{SAVE_DIR}")    : os.makedirs( f"{SAVE_DIR}"     ,exist_ok=True )
    if not os.path.exists(f"{IMAGE_FOLDER}"): os.makedirs( f"{IMAGE_FOLDER}" ,exist_ok=True )
    if not os.path.exists(f"{VICON_FOLDER}"): os.makedirs( f"{VICON_FOLDER}" ,exist_ok=True )
    if not os.path.exists(f"{SAVE_DIR}/proj"): os.makedirs( f"{SAVE_DIR}/proj" ,exist_ok=True )

    if DEBUGGING:
        shutil.copyfile( "./config/eomer.json", f"{SAVE_DIR}/eomer.json" )
        shutil.copyfile( "../theodwyn/navigations/mekf.py", f"{SAVE_DIR}/mekf.py" )

    with open("./config/eomer.json") as file:
        json_data = json.load(file)

    config                                      = StackConfiguration()
    config.log_filename                         = f"{SAVE_DIR}/log.txt"
    config.camera_configs                       = json_data["camera"]
    config.network_configs                      = json_data["network"]
    config.network_configs[1]["SDA"]            = board.SDA
    config.network_configs[1]["SCL"]            = board.SCL
    # config.guidance_configs                     = { "shape" : PRESET_CIRCLE, "shape_params"  : { "r" : 1.0, "freq"  : 2 * pi / 45. } }
    config.guidance_configs                     = { "csv_trajfilename" : "./trajs/trajectory_redim_t05.csv" }
    config.controller_configs["p_gain"]         = 1.50 * diag([0.5,0.5])
    config.controller_configs["a_gain"]         = 0.50
    config.controller_configs["c_gain"]         = 0.40
    config.navigation_configs                   = json_data["navigation"]
    config.navigation_configs["kps3D_path"]     = "/home/eomer/soho_128.npy"
    config.navigation_configs["model_path"]     = "/home/eomer/models/dropout_custom_krcnn_mbnv3_fpn_v2_best_fine_tine_no_j_add_o_real_synth_soho_40kps_epochs_50_lr5_bs87.onnx"
    config.navigation_configs["est_csv_fn"]     = f"{SAVE_DIR}/pe_ests_testing_{TIME_AR}.csv"
    config.navigation_configs["meas_csv_fn"]    = f"{SAVE_DIR}/pe_meas_testing_{TIME_AR}.csv"
    config.navigation_configs["infer_csv_fn"]   = f"{SAVE_DIR}/pe_infer_testing_{TIME_AR}.csv"
    config.navigation_configs["proj_path"]      = f"{SAVE_DIR}/proj"

    config.camera_classes               = XIMEA
    config.network_classes              = [ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,SabertoothSimpleSerial,ViconConnection,SingleEventSignal]
    config.controller_classes           = ViconFeedback
    # config.guidance_classes             = Preset2DShapes
    config.guidance_classes             = CSVInterpreter
    config.navigation_classes           = MEKF
    
    with EomerStack( 
        config=config, 
        image_path=IMAGE_FOLDER, 
        vicon_csv_path=VICON_FOLDER, 
        vicon_csv_fn=f"vicon_sc_{TIME_AR}.csv" 
    ) as theo_stack:
        try:
            while not theo_stack.sigterm.is_set() : sleep(5)
        except KeyboardInterrupt: 
            print("---> Eomer Session Ended <---")
