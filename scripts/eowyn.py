import os
import board
import json
from numpy                                      import array, diag, identity
from math                                       import pi
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.controllers.viconfeedback         import ViconFeedback
from theodwyn.guidances.presets                 import Preset2DShapes, PRESET_CIRCLE
from theodwyn.stacks.eowyn                      import EowynStack
from rohan.data.classes                         import StackConfiguration
from time                                       import sleep

if __name__ == "__main__":

    with open("./config/eowyn.json") as file:
        json_data = json.load(file)

    config                                      = StackConfiguration()
    config.network_configs                      = json_data["network"]
    config.network_configs[1]["SDA"]            = board.SDA
    config.network_configs[1]["SCL"]            = board.SCL
    config.guidance_configs                     = { 
        "shape" : PRESET_CIRCLE, 
        "shape_params"  : { 
            "r"     : 0, 
            "freq"  : 2 * pi / 45.,
            "c_pnt" : [1.00,0.25]
        } 
    }
    config.controller_configs["p_gain"]         = 1.50 * diag([0.5,0.5])
    config.controller_configs["a_gain"]         = 0.50
    config.controller_configs["c_gain"]         = 0.40


    config.network_classes              = [ZMQDish,SabertoothSimpleSerial,SabertoothSimpleSerial,ViconConnection]
    config.controller_classes           = ViconFeedback
    config.guidance_classes             = Preset2DShapes
    
    with EowynStack( config=config ) as theo_stack:
        try:
            while not theo_stack.sigterm.is_set() : sleep(5)
        except KeyboardInterrupt: 
            print("---> Eowyn Session Ended <---")
