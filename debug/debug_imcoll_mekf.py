import board
import json
from numpy                                      import array, diag, identity
from math                                       import pi
from theodwyn.cameras.ximea                     import XIMEA
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.controllers.viconfeedback         import ViconFeedback
from theodwyn.guidances.presets                 import Preset2DShapes, PRESET_CIRCLE
from theodwyn.navigations.mekf                  import MEKF
from theodwyn.stacks.debug.debug_imcoll_mekf    import CollectionStack
from rohan.data.classes                         import StackConfiguration
from time import sleep, strftime

if __name__ == "__main__":

    with open("./config/debug_imcoll_mekf.json") as file:
        json_data = json.load(file)

    config                              = StackConfiguration()
    # config.log_filename                 = ".logs/log.txt"
    config.camera_configs               = json_data["camera"]
    config.network_configs              = json_data["network"]
    config.network_configs[1]["SDA"]    = board.SDA
    config.network_configs[1]["SCL"]    = board.SCL
    config.guidance_configs             = {
        "shape"         : PRESET_CIRCLE,
        "shape_params"  : {
            "r"     : 0.75,
            "freq"  : 2 * pi / 45.
        }
    }
    config.controller_configs["p_gain"]         = 1.50 * diag([0.5,0.5])
    config.controller_configs["a_gain"]         = 0.50
    config.controller_configs["c_gain"]         = 0.40
    
    config.navigation_configs                   = json_data["navigation"]
    config.navigation_configs["kps3D_path"]     =  "/home/eomer/soho_128.npy"
    config.navigation_configs["model_path"]     = "/home/eomer/dropout_custom_krcnn_mbnv3_fpn_v2_last_epoch_test_010_soho_40kps.onnx"
    # config.navigation_configs["image_dir"]  = "home/eomer/soho_trajectories/subset_midLighting_jpsmooth2_trajectory"
    config.navigation_configs["est_csv_fn"]     = f"~/eomer_usb/pe_ests_testing_{strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    config.navigation_configs["meas_csv_fn"]    = f"~/eomer_usb/pe_meas_testing_{strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    config.navigation_configs["infer_csv_fn"]   = f"~/eomer_usb/pe_infer_testing_{strftime('%Y_%m_%d_%H_%M_%S')}.csv"
    config.navigation_configs["proj_path"]      = f"~/eomer_usb/proj"

    config.camera_classes               = XIMEA
    config.network_classes              = [ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,SabertoothSimpleSerial,ViconConnection]
    config.controller_classes           = ViconFeedback
    config.guidance_classes             = Preset2DShapes
    config.navigation_classes           = MEKF
    
    with CollectionStack( config=config ) as theo_stack:
        try:
            while True : sleep(30)
        except KeyboardInterrupt: 
            print("---> Calibration Session Ended <---")
