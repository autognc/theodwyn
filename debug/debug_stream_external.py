import json
import cv2                          as cv 
from theodwyn.externals.gamepad     import ExternalXboxGamePad, TriggerQConfig
from rohan.common.logging           import Logger
from typing                         import Optional
from time                           import time

class XboxControllerOut(ExternalXboxGamePad):

    process_name = "Outgoing Xbox Controller"

    def __init__(
        self,
        addr             : str,
        data_format      : str,
        topic            : str              = "Noname",
        gstream_pipeline : Optional[str]    = None,
        trigger_qconfig  : TriggerQConfig   = TriggerQConfig(),
        logger           : Optional[Logger] = None,
        **kwargs
    ):
        super().__init__(
            addr             = addr,
            data_format      = data_format,
            topic            = topic,
            gstream_pipeline = gstream_pipeline,
            trigger_qconfig  = trigger_qconfig,
            logger           = logger,
            **kwargs 
        )

    def processLT(
        self, 
        frame : cv.typing.MatLike
    ) -> None:
        if isinstance(self.logger,Logger):
            self.logger.write( "Left Trigger Pulled", self.process_name )
        # cv.imwrite("./images/test_{:.5f}.jpg".format(time()), frame )

    def processRT(
        self, 
        frame : cv.typing.MatLike
    ) -> None:
        if isinstance(self.logger,Logger):
            self.logger.write( "Right Trigger was pulled ... Saving Current Image", self.process_name )
        cv.imwrite("./images/image_{:.5f}.jpg".format(time()), frame )

if __name__ == "__main__":
    
    with open("./config/debug_stream_config.json") as file:
        json_data = json.load(file)
    network_configs = json_data["network"]
    camera_configs  = json_data["camera"]
    gstream_config  = camera_configs["gstream_config"]

    gstream_pipeline = (
        'udpsrc address={} port={} '
        'caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264" ! '
        'rtph264depay ! '
        'avdec_h264 ! '
        'videoconvert ! '
        'appsink drop=1'
    ).format( gstream_config["sink_ip"], gstream_config["sink_port"] )

    trigger_qconfig = TriggerQConfig(
        max_qLsize  =100,
        max_qRsize  =100,
        qL_freq     =10,
        qR_freq     =10
    )

    try:
        with Logger() as logger:
            with XboxControllerOut( 
                    addr=network_configs[0]["addr"], 
                    data_format=network_configs[0]["data_format"], 
                    topic=network_configs[0]["topic"],
                    gstream_pipeline=gstream_pipeline,
                    trigger_qconfig=trigger_qconfig,
                    logger=logger
            ) as controller_comm:
                while True: pass
    except KeyboardInterrupt as e:
        print("Exiting ... ")
