import threading
import struct
import cv2                              as cv
from queue                              import Queue, Full
from theodwyn.networks.comm_prot        import ZMQRadio
from theodwyn.controllers.gamepad       import XboxGamePad
from rohan.common.logging               import Logger
from rohan.utils.timers                 import IntervalTimer
from rohan.common.type_aliases          import Resolution
from typing                             import Optional
from dataclasses                        import dataclass

@dataclass
class TriggerQConfig:
    max_qLsize : int = 100
    max_qRsize : int = 100
    qL_freq    : int = 10
    qR_freq    : int = 10

class ExternalXboxGamePad(ZMQRadio):

    """
    Object for external Xbox Controller which sends commands to stack 
    :param addr: IP-address of microcomputer running stack, to communicate with
    :param data_format: dataformat of output message according to struct package
    :param topic: group name to broadcast data
    :param gstream_pipeline: gstreamer pipeline for recieving videostream
    :param thread_intrvl: Inverse-frequency of thread loop
    :param trigger_qconfig: configuration for trigger-action queues
    :param logger: rohan logging instance 
    """

    gstream_pipeline : Optional[str]             = None
    stream_obj       : Optional[cv.VideoCapture] = None

    trigger_qconfig  : TriggerQConfig
    
    triggerL_queue   : Queue
    triggerR_queue   : Queue
    send_qL_event    : threading.Event
    send_qR_event    : threading.Event

    procL_interval   : float
    procR_interval   : float

    def __init__(
        self,
        addr                : str,
        data_format         : str,
        topic               : str              = "Noname",
        gstream_pipeline    : Optional[str]    = None,
        display_resolution  : Resolution       = (2560,1440),
        thread_intrvl       : float            = 1/60,
        trigger_qconfig     : TriggerQConfig   = TriggerQConfig(),
        logger              : Optional[Logger] = None,
        **kwargs
    ):
        super().__init__(
            addr        = addr,
            data_format = data_format,
            topic       = topic,
            logger      = logger
        )
        self.load( 
            gstream_pipeline=gstream_pipeline,
            **kwargs
        )
        self.thread_interval    = thread_intrvl
        self.triggerL_queue     = Queue(maxsize=trigger_qconfig.max_qLsize)
        self.triggerR_queue     = Queue(maxsize=trigger_qconfig.max_qRsize)
        self.procL_interval     = 1/trigger_qconfig.qL_freq
        self.procR_interval     = 1/trigger_qconfig.qR_freq
        self.display_resolution = display_resolution
        
        self.add_threaded_method( target=self.spin )
        self.add_threaded_method( target=self.stream_spin )
        self.add_threaded_method( target=self.triggerL_spin )
        self.add_threaded_method( target=self.triggerR_spin )

        self.send_qL_event = threading.Event()
        self.send_qR_event = threading.Event()

    def __enter__(self):
        if self.gstream_pipeline is not None:
            if isinstance( self.stream_obj, cv.VideoCapture ):
                self.stream_obj.release()
            self.stream_obj = cv.VideoCapture( 
                self.gstream_pipeline, 
                cv.CAP_GSTREAMER
            )            
            if not self.stream_obj.isOpened():
                if isinstance(self.logger,Logger): 
                    self.logger.write(
                        f'f"Unable to open camera stream with provided pipe: {self.gstream_pipeline}',
                        process_name=self.process_name
                    )
        return super().__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        super().__exit__(exception_type, exception_value, traceback)
        if isinstance( self.stream_obj, cv.VideoCapture ):
            self.stream_obj.release()
        self.stream_obj = None

    def spin( self ) -> None:
        """
        Primary Threaded Process (1/4) : Sends packets of encoded xbox-controller input
        """
        thread_timer    = IntervalTimer(interval=self.thread_interval)
        procL_timer     = IntervalTimer(interval=self.procL_interval)
        procR_timer     = IntervalTimer(interval=self.procR_interval)
        with XboxGamePad() as controller:
            while not self.sigterm.is_set():
                thread_timer.await_interval()
                
                # >> Determine control from controller
                control_inpt = controller.determine_control() 

                # >> Add frame to processing queue on RT and LT, respectively
                if control_inpt[2] > 0.5 :
                    if procL_timer.check_interval():
                        if not self.send_qL_event.is_set() : self.send_qL_event.set()

                if control_inpt[5] > 0.5 :
                    if procR_timer.check_interval():
                        if not self.send_qR_event.is_set() : self.send_qR_event.set()

                self.socket.send(
                    struct.pack(
                        self.publisher_format,
                        self.topic, " ".encode('ascii'),
                        *control_inpt
                    ), group=self.topic
                )

    def stream_spin( self ) -> None:
        """
        Threaded Process (2/4) : Reads stream and passes images to processing queues
        """
        while not self.sigterm.is_set():
            if isinstance( self.stream_obj, cv.VideoCapture ):
                frame = None
                if self.stream_obj.isOpened():
                    _, frame = self.stream_obj.read()
                    cv.imshow("Camera Stream", mat=cv.resize( frame, self.display_resolution ) )
                    cv.waitKey(1)

                if not frame is None:
                    if self.send_qL_event.is_set():
                        try:
                            self.triggerL_queue.put( frame, block=False )
                        except Full:
                            if isinstance(self.logger,Logger):
                                self.logger.write("Left-trigger queue is full please wait", self.process_name)
                        self.send_qL_event.clear()

                    if self.send_qR_event.is_set():
                        try:
                            self.triggerR_queue.put( frame, block=False )
                        except Full:
                            if isinstance(self.logger,Logger):
                                self.logger.write("Right-trigger queue is full please wait", self.process_name)
                        self.send_qR_event.clear()

    def triggerL_spin( self ) -> None:
        """
        Threaded Process (3/4) : Processes images from Left-Trigger queue
        """
        warned_once = False
        while not self.sigterm.is_set():
            while not self.triggerL_queue.empty(): 
                if self.sigterm.is_set() and not warned_once: 
                    if isinstance(self.logger,Logger):
                        self.logger.write( '[WARNING] WAIT! finishing processing on Left-Trigger queue', self.process_name )
                    warned_once = True
                self.processLT( frame=self.triggerL_queue.get() )


    def triggerR_spin( self ) -> None:
        """
        Threaded Process (3/4) : Processes images from Right-Trigger queue
        """
        warned_once = False
        while not self.sigterm.is_set():
            while not self.triggerR_queue.empty(): 
                if self.sigterm.is_set() and not warned_once:
                    if isinstance(self.logger,Logger): 
                        self.logger.write( '[WARNING] WAIT! finishing processing on Right-Trigger queue', self.process_name )
                    warned_once = True
                self.processRT( frame=self.triggerR_queue.get() )

    def processLT( 
        self, 
        frame : cv.typing.MatLike 
    ) -> None:
        """
        Process incurred on images queue from Left-Trigger Pull
        """
    
    def processRT( 
        self, 
        frame : cv.typing.MatLike 
    ) -> None:
        """
        Process incurred on images queue from Right-Trigger Pull
        """
