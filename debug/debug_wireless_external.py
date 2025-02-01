import json
from theodwyn.externals.gamepad     import ExternalXboxGamePad
from rohan.common.logging           import Logger

if __name__ == "__main__":

    with open("./config/debug_wireless_config.json") as file:
        json_data = json.load(file)
    network_configs = json_data["network"]

    try:
        with Logger() as logger:
            with ExternalXboxGamePad( 
                    addr=network_configs[0]["addr"], 
                    data_format=network_configs[0]["data_format"], 
                    topic=network_configs[0]["topic"],
                    logger=logger
            ) as controller_comm:
                while True: pass
    except KeyboardInterrupt as e:
        print("Exiting ... ")