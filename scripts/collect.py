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
from theodwyn.guidances.file_interpreters       import CSVInterpreter
from theodwyn.stacks.collection                 import CollectionStack
from rohan.data.classes                         import StackConfiguration
from time import sleep 

if __name__ == "__main__":

    with open("./config/collect.json") as file:
        json_data = json.load(file)

    config                              = StackConfiguration()
    config.camera_configs               = json_data["camera"]
    config.network_configs              = json_data["network"]
    config.network_configs[1]["SDA"]    = board.SDA
    config.network_configs[1]["SCL"]    = board.SCL
    # config.guidance_configs             = { "shape" : PRESET_CIRCLE, "shape_params": { "r" : 1.0, "freq" : 2 * pi / 120 } }
    config.guidance_configs             = { "csv_trajfilename" : "./trajs/trajectory_redim_t05.csv" }
    config.controller_configs["p_gain"] = 1.50 * diag([0.5,0.5])
    config.controller_configs["a_gain"] = 0.50
    config.controller_configs["c_gain"] = 0.4

    config.camera_classes               = XIMEA
    config.network_classes              = [ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,SabertoothSimpleSerial,ViconConnection]
    config.controller_classes           = ViconFeedback
    # config.guidance_classes             = Preset2DShapes
    config.guidance_classes             = CSVInterpreter
    
    with CollectionStack( config=config ) as theo_stack:
        try:
            while True : sleep(30)
        except KeyboardInterrupt: 
            print("---> Calibration Session Ended <---")
