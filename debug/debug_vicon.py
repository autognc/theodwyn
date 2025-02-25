import json
from rohan.common.logging        import Logger
from theodwyn.networks.vicon     import ViconConnection

OBJECT_NAME = "eomer"
if __name__ == "__main__":

    with open("./config/debug_vicon.json") as file:
        json_data = json.load(file)

    with Logger() as logger:
        with ViconConnection(**json_data,logger=logger) as network:
            while True: 
                print( network.recv_pose( OBJECT_NAME ) )
        