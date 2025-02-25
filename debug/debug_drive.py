import board
import json
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.cameras.intel_realsense           import D455
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.stacks.debug.debug_comm_stack     import DebugCommStack
from rohan.data.classes                         import StackConfiguration
from time                                       import time

if __name__ == "__main__":

    with open("./config/debug_wireless_drive_config.json") as file:
        json_data = json.load(file)

    config = StackConfiguration()
    config.log_filename = "./logs/log_{:.2f}.txt".format( time() )
    config.camera_configs = json_data["camera"]
    config.controller_configs = json_data["controller"]

    config.network_configs = json_data["network"]
    config.network_configs[1]["SDA"] = board.SDA
    config.network_configs[1]["SCL"] = board.SCL

    # config.camera_classes = D455
    config.network_classes = [ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,SabertoothSimpleSerial]

    stack = DebugCommStack( config=config, verbose=False )
    stack.spin()
    print("---> Debugging Session Ended <---")