import board
import os, shutil, glob
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

    clear_flag  = True 
    # clear_flag  = False
    if clear_flag:
        directories     = ['logs/', 'data/', 'csvs/']
        # get all items in the directories (flattening the list)
        paths           = sum([glob.glob(os.path.join(d, '*')) for d in directories], [])
        # filter out items that are named '.gitignore'
        paths_to_remove = list(filter(lambda p: os.path.basename(p) != '.gitignore', paths))
        # remove each item (removes directories recursively and files)
        list(map(lambda p: shutil.rmtree(p) if os.path.isdir(p) else os.remove(p), paths_to_remove)) 

    config              = StackConfiguration()
    config.log_filename = "./logs/log_{:.2f}.txt".format( time() )

    config.navigation_classes   = MEKF
    config.navigation_configs   = json_data["navigation"]

    with DebugMEKF( config = config ) as theo_stack:
        try:
            while not theo_stack.sigterm.is_set():
                sleep(10)
        except KeyboardInterrupt:
            theo_stack.sigterm.set()

    print("---> Debugging Session Ended <---")