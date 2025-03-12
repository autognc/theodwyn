import board
import json
from numpy                                      import array, diag, identity
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.controllers.viconfeedback         import ViconFeedback
from theodwyn.stacks.eomer                      import EomerStack
from rohan.data.classes                         import StackConfiguration

if __name__ == "__main__":

    with open("./config/eomer.json") as file:
        json_data = json.load(file)

    config                              = StackConfiguration()
    config.network_configs              = json_data["network"]
    config.network_configs[1]["SDA"]    = board.SDA
    config.network_configs[1]["SCL"]    = board.SCL
    config.controller_configs           = json_data["controller"]
    config.controller_configs["p_gain"] = diag([0.75,0.5])

    config.network_classes              = [ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,SabertoothSimpleSerial,ViconConnection]
    config.controller_classes           = ViconFeedback
    
    theo_stack = EomerStack( config=config ) 
    theo_stack.spin()
    print("---> Calibration Session Ended <---")