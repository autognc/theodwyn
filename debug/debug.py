import board
import json
from theodwyn.cameras.intel_realsense   import D455
from theodwyn.controllers.gamepad       import XboxGamePad
from theodwyn.networks.adafruit         import Adafruit_PCA9685
from theodwyn.stacks.debug.debug_stack  import DebugStack
from rohan.data.classes                 import StackConfiguration
from time                               import time

if __name__ == "__main__":

    with open("./config/debug_config.json") as file:
        json_data = json.load(file)

    config = StackConfiguration()
    config.log_filename = "./logs/log_{:.2f}.txt".format( time() )
    config.camera_configs = json_data["camera"]
    config.controller_configs = json_data["controller"]
    config.network_configs = json_data["network"]
    config.network_configs["SDA"] = board.SDA
    config.network_configs["SCL"] = board.SCL

    config.camera_classes = D455
    config.controller_classes = XboxGamePad
    config.network_classes = Adafruit_PCA9685

    stack = DebugStack( config=config, verbose=True )
    stack.spin()
    print("---> Debugging Session Ended <---")

