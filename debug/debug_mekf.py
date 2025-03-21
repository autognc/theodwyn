import board
import json
from theodwyn.navigations.mekf                  import MEKF  
from theodwyn.cameras.ximea                     import XIMEA
from theodwyn.stacks.debug.debug_mekf_stack     import DebugMEKF
from rohan.data.classes                         import StackConfiguration
from time                                       import time, sleep

import pdb
if __name__ == "__main__":

    with open("./config/debug_mekf.json") as file:
        json_data   = json.load(file)

    config              = StackConfiguration()
    config.log_filename = "./logs/log_{:.2f}.txt".format( time() )

    config.navigation_classes   = MEKF
    config.navigation_configs   = json_data["navigation"]
    # if config.navigation_configs["source_mode"] == "camera":
    #     config.camera_classes       = XIMEA
    #     config.camera_configs       = json_data["camera"]

    with DebugMEKF( config = config ) as theo_stack:
        try:
            while True:
                sleep(10)
        except KeyboardInterrupt:
            theo_stack.sigterm.set()

    print("---> Debugging Session Ended <---")