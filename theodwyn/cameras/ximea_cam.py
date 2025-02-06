import numpy                            as np
import cv2                              as cv
from ximea                              import xiapi
from copy                               import deepcopy
from rohan.common.base_cameras          import ThreadedCameraBase
from rohan.common.logging               import Logger
from rohan.common.type_aliases          import Config, Resolution
from typing                             import TypeVar, Optional, Tuple, Dict
from numpy.typing                       import NDArray
import yaml
import os 



SelfXIMEA = TypeVar("SelfXIMEA", bound="XIMEA" )
class XIMEA(ThreadedCameraBase):
    """
    Model for an Intel D455 Realsense Camera.
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
    capture_obj      : Optional[xiapi.Camera]    = None
    
    # Stream object used to stream camera data
    stream_obj       : Optional[cv.VideoWriter] = None
    gstream_pipeline : Optional[str]            = None
    stream_channel   : int                      = 0
    stream_resolution: Optional[Resolution]     = None
    frame_color      : Optional[NDArray]        = None
    frame_depth      : Optional[NDArray]        = None
    frame_data       : Optional[NDArray]        = np.zeros((1280,800))

    def __init__(
        self,
        resolution          : Resolution            = (1280,800),
        fps                 : int                   = 30,
        stream_resolution   : Optional[Resolution]  = None,
        exposure            : int                   = 10000,
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
        self.stream_resolution = stream_resolution if not stream_resolution is None else resolution
        self.exposure = exposure
        self.load( 
            gstream_config=gstream_config,
            **config_kwargs 
        )
        self.add_threaded_method( target=self.spin )


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
                'udpsink port={} host={}'

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

        """Load Configuration File"""
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(os.path.dirname(current_dir))

        config_path = os.path.join(parent_dir,'debug', 'config', 'ximea_config.yml')

        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        cam_config = config.get('Camera Settings',{})
        resolution = cam_config.get('resolution')

        if isinstance( self.capture_obj, xiapi.Camera ):
            self.capture_obj.close_device()
        self.capture_obj    = xiapi.Camera()
        self.capture_obj.open_device()
        

        """Defining Camera Settings from Configuration File"""
        self.capture_obj.set_framerate(cam_config.get('framerate', self.fps))
        self.capture_obj.set_height(resolution.get('height', self.stream_resolution[0]))
        self.capture_obj.set_width(resolution.get('width', self.stream_resolution[1]))
        #self.capture_obj.set_imgdataformat(cam_config.get('imgdataformat'))
        try:
            if cam_config.get('aeg') == True:
                self.capture_obj.enable_aeag()
                self.capture_obj.set_ae_max_limit(cam_config.get('ae_max_limit'))
                self.capture_obj.set_ag_max_limit(cam_config.get('ag_max_limit'))
            else:
                self.capture_obj.disable_aeag()
                self.capture_obj.set_exposure(cam_config.get('exposure', self.exposure))
                self.capture_obj.set_gain(cam_config.get('gain'))
        except NameError:
            "Exposure/Gain setting not defined in configuration file"

        try:
            if cam_config.get('awb') == True:
                self.capture_obj.enable_auto_wb()
            else:
                self.capture_obj.disable_auto_wb()
                self.capture_obj.set_wb_kr(cam_config.get('wb_coef_red'))
                self.capture_obj.set_wb_kg(cam_config.get('wb_coef_green'))
                self.capture_obj.set_wb_kb(cam_config.get('wb_coef_blue'))
        except NameError:
            "White Balance setting not defined in configuration file"

        self.capture_obj.start_acquisition()


        if self.gstream_pipeline is not None:
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
            frame = None
            try:
                if isinstance( self.capture_obj, xiapi.Camera ):
                    frame = xiapi.Image()
                    self.capture_obj.get_image(frame)
                    with self._instance_lock:
                        self.frame_data    = frame.get_image_data_numpy()
                        
            except Exception as e:
                if isinstance(self.logger,Logger):
                    self.logger.write( f"Failed with Exception: {e}", process_name=self.process_name )
                frame   = None
        
            if frame is not None and isinstance( self.stream_obj, cv.VideoWriter ):
                if self.stream_obj.isOpened():
                    with self._instance_lock:
                            self.stream_obj.write( image=cv.resize( self.frame_data, self.stream_resolution) )

    def get_frame( self ) -> Tuple[ cv.typing.MatLike, cv.typing.MatLike ]:
        """
        Retrieves return code and frame information
        """
        with self._instance_lock:
            return deepcopy( self.frame_data )