import json
from time                               import time
from rohan.common.logging               import Logger
from theodwyn.externals.vicon_collector import ViconCollector

if __name__ == "__main__":

    with open("./config/debug_viconcollector.json") as file:
        json_data = json.load(file)

    data_filename = "./data/data_{:.2f}.csv".format( time() )
    with Logger() as logger:
        with ViconCollector(**json_data,filename=data_filename, logger=logger) as network:
            try:
                while True: 
                    pass # NOTE: It's collecting data in the background on a seperate thread when you enter the context
            except KeyboardInterrupt:
                print("exitting ...")
        