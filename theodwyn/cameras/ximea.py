import numpy                            as np
import cv2                              as cv
from ximea                              import xiapi
from copy                               import deepcopy
from rohan.common.base_cameras          import ThreadedCameraBase
from rohan.common.logging               import Logger
from rohan.common.type_aliases          import Config, Resolution
from rohan.utils.timers                 import IntervalTimer
from typing                             import TypeVar, Optional, Tuple, Dict
from numpy.typing                       import NDArray

SelfXIMEA = TypeVar("SelfXIMEA", bound="XIMEA" )
class XIMEA(ThreadedCameraBase):
    """
    Model for a Ximea Camera.
    :param resolution: The resolution of the camera's rgb channels
    :param lidar_resolution: The resolution of the camera's depth channel
    :param fps: The fps of the camera's channels
    :param stream_resolution: Resolution of stream -> Defaults to resolution
    :param gstream_config: Gstreamer configuration options
    :param logger: rohan Logger() instance
    """
    
    # For Logging
    process_name     : str                      = "XIMEA xiC camera (threaded)"
    
    # Capture object used to connect to ximea I/O
    capture_obj      : Optional[xiapi.Camera]       = None
    aeg_settings     : Optional[Dict[str,int]]      = None
    awb_settings     : Optional[Dict[str,int]]      = None
    
    # Stream object used to stream camera data
    stream_obj          : Optional[cv.VideoWriter] = None
    gstream_pipeline    : Optional[str]            = None
    stream_channel      : int                      = 0
    stream_resolution   : Optional[Resolution]     = None
    frame_instance      : xiapi.Image
    frame_data          : Optional[NDArray]
    stream_timer        : IntervalTimer


    def __init__(
        self,
        resolution          : Resolution            = (1280,800),
        fps                 : int                   = 30,
        aeg_settings        : Dict[str,int]         = None,
        awb_settings        : Dict[str,int]         = None,
        stream_resolution   : Optional[Resolution]  = None,
        gstream_config      : Config                = None,
        logger              : Optional[Logger]      = None,
        **config_kwargs
    ):
        ThreadedCameraBase.__init__( 
            self,
            resolution=resolution,  
            fps=fps,
            logger=logger
        )
        self.aeg_settings           = aeg_settings
        self.awb_settings           = awb_settings
        self.stream_resolution      = stream_resolution if not stream_resolution is None else resolution
        self.frame_instance         = xiapi.Image()
        self.frame_data             = np.zeros( resolution )
        self.stream_timer           = IntervalTimer( interval=1/fps )
        self.load( 
            gstream_config=gstream_config,
            **config_kwargs 
        )
        self.add_threaded_method( target=self.spin )
        if not self.gstream_pipeline is None:
            self.add_threaded_method( target=self.spin_stream )

    def load( 
        self, 
        **kwargs
    ) -> None:
        """
        Loads (or reloads) the configuration parameters for the XIMEA xiC camera
        """
        def build_gstreamer_pipeline(
            sink_ip     : str,
            sink_port   : str,
        ) -> str:
            return (
                'appsrc ! '
                'video/x-raw, format=BGR, width={}, height={}, framerate={}/1 ! '
                'videoconvert ! '
                'x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! '
                'h264parse ! '
                'rtph264pay ! '
                'udpsink port={} host={} auto-multicast=true'

            ).format(
                self.stream_resolution[0],
                self.stream_resolution[1],
                self.fps,
                sink_port,
                sink_ip
            )

        if 'gstream_config' in kwargs:
            self.gstream_pipeline = build_gstreamer_pipeline( **kwargs['gstream_config'] ) if isinstance(kwargs['gstream_config'],Dict) else None
        ThreadedCameraBase.load(self,**kwargs)


    def connect( self ) -> None:
        """
        Connect to the CV video stream
        """

        if isinstance( self.capture_obj, xiapi.Camera ):
            self.capture_obj.close_device()
        self.capture_obj = xiapi.Camera()
        self.capture_obj.open_device()

        self.capture_obj.set_framerate( self.fps            )
        self.capture_obj.set_width(     self.resolution[0]  )
        self.capture_obj.set_height(    self.resolution[1]  )
        self.capture_obj.set_imgdataformat('XI_RGB24')

        if self.aeg_settings is not None:
            if 'ae_max_limit' in self.aeg_settings and 'ag_max_limit' in self.aeg_settings:
                self.capture_obj.enable_aeag()
                self.capture_obj.set_ae_max_limit( self.aeg_settings['ae_max_limit'] )
                self.capture_obj.set_ag_max_limit( self.aeg_settings['ag_max_limit'] )
            elif 'exposure' in self.aeg_settings and 'gain' in self.aeg_settings:
                self.capture_obj.disable_aeag()
                self.capture_obj.set_exposure(  self.aeg_settings['exposure'] )
                self.capture_obj.set_gain(      self.aeg_settings['gain']     )
            else:        
                if isinstance(self.logger,Logger): 
                    self.logger.write(
                        "Exposure/Gain setting not defined in configuration file",
                        process_name=self.process_name
                    )    
        else:        
            if isinstance(self.logger,Logger): 
                self.logger.write(
                    "Exposure/Gain setting not defined in configuration file",
                    process_name=self.process_name
                )    

        if self.awb_settings is not None:
            if 'wb_coef_red' in self.aeg_settings and 'wb_coef_green' in self.aeg_settings and 'wb_coef_blue' in self.aeg_settings:
                self.capture_obj.disable_auto_wb()
                self.capture_obj.set_wb_kr( self.aeg_settings['wb_coef_red']    )
                self.capture_obj.set_wb_kg( self.aeg_settings['wb_coef_green']  )
                self.capture_obj.set_wb_kb( self.aeg_settings['wb_coef_blue']   )
            else:
                self.capture_obj.enable_auto_wb()
        else:
            self.capture_obj.enable_auto_wb()

        self.capture_obj.start_acquisition()

        if not self.gstream_pipeline is None:
            if isinstance( self.stream_obj, cv.VideoWriter ):
                self.stream_obj.release()
            self.stream_obj = cv.VideoWriter( 
                self.gstream_pipeline, 
                cv.CAP_GSTREAMER, 
                0, 
                self.fps, 
                self.stream_resolution 
            )            
            if not self.stream_obj.isOpened():
                if isinstance(self.logger,Logger): 
                    self.logger.write(
                        f"Unable to open camera stream with provided pipe:\n      {self.gstream_pipeline}",
                        process_name=self.process_name
                    )    


    def disconnect( self ) -> None:
        """
        Disconnect from the CV video stream
        """
        if isinstance( self.stream_obj, cv.VideoWriter ):
            self.stream_obj.release()
        self.stream_obj = None
        if isinstance( self.capture_obj, xiapi.Camera ):
            self.capture_obj.stop_acquisition()
            self.capture_obj.close_device()
        self.capture_obj = None


    def spin( self ):
        """
        Threaded Process: Reads camera data and streams it according to provided gstreamer config
        """
        while not self.sigterm.is_set():
            try:
                if self.capture_obj:
                    self.capture_obj.get_image(self.frame_instance)
                    numpy_data = self.frame_instance.get_image_data_numpy()
                    with self._instance_lock:
                        self.frame_data = numpy_data
                        
            except Exception as e:
                if self.logger:
                    self.logger.write( f"Failed with Exception: {e}", process_name=self.process_name )

    
    def spin_stream( self ):    
        while not self.sigterm.is_set():
            if self.stream_obj.isOpened(): 
                self.stream_timer.await_interval()
                with self._instance_lock: 
                    frame_data = self.frame_data.copy()
                self.stream_obj.write( image=cv.resize( frame_data, self.stream_resolution) )


    def get_frame( self ) -> NDArray:
        """
        Retrieves return code and frame information
        """
        with self._instance_lock:
            frame_data = self.frame_data.copy()
        return frame_data