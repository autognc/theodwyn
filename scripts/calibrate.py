import board
import json
from theodwyn.networks.sabertooth               import SabertoothSimpleSerial
from theodwyn.networks.adafruit                 import Adafruit_PCA9685
from theodwyn.networks.comm_prot                import ZMQDish
from theodwyn.networks.vicon                    import ViconConnection
from theodwyn.stacks.calibration                import CalibrationStack
from rohan.data.classes                         import StackConfiguration
from time                                       import sleep

if __name__ == "__main__":

    with open("./config/calibrate.json") as file:
        json_data = json.load(file)

    config                              = StackConfiguration()
    config.network_configs              = json_data["network"]
    config.network_configs[1]["SDA"]    = board.SDA
    config.network_configs[1]["SCL"]    = board.SCL
    config.network_classes              = [ZMQDish,Adafruit_PCA9685,SabertoothSimpleSerial,SabertoothSimpleSerial,ViconConnection]

    with CalibrationStack( config=config ) as theo_stack:
        while True:
            sleep(10)
            if theo_stack.sigterm.is_set():
                break
    print("---> Calibration Session Ended <---")