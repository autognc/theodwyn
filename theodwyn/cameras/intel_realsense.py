import numpy                            as np
import cv2                              as cv
import pyrealsense2                     as rs
from copy                               import deepcopy
from rohan.common.base_cameras          import ThreadedLidarCameraBase
from rohan.common.logging               import Logger
from rohan.common.type_aliases          import Config, Resolution
from typing                             import TypeVar, Optional, Tuple, Dict
from numpy.typing                       import NDArray

SelfD455 = TypeVar("SelfD455", bound="D455" )
class D455(ThreadedLidarCameraBase):
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
    process_name     : str                      = "Intel RS camera (threaded)"
    
    # Capture object used to connect to Intel RS I/O
    capture_obj      : Optional[rs.pipeline]    = None
    
    # Stream object used to stream camera data
    stream_obj       : Optional[cv.VideoWriter] = None
    gstream_pipeline : Optional[str]            = None
    stream_channel   : int                      = 0
    stream_resolution: Optional[Resolution]     = None
    frame_color      : Optional[NDArray]        = None
    frame_depth      : Optional[NDArray]        = None

    def __init__(
        self,
        resolution          : Resolution            = (1280,800),
        lidar_resolution    : Resolution            = (1280,720),
        fps                 : int                   = 30,
        stream_resolution   : Optional[Resolution]  = None,
        gstream_config      : Config                = None,
        logger              : Optional[Logger]      = None,
        **config_kwargs
    ):
        ThreadedLidarCameraBase.__init__( 
            self,
            resolution=resolution, 
            lidar_resolution=lidar_resolution, 
            fps=fps, 
            lidar_fps=fps,
            logger=logger
        )
        self.stream_resolution = stream_resolution if not stream_resolution is None else resolution
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
        Loads (or reloads) the configuration parameters for the Realsense D455 camera
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
        ThreadedLidarCameraBase.load(self,**kwargs)


    def connect( self ) -> None:
        """
        Connect to the CV video stream
        """
        if isinstance( self.capture_obj, rs.pipeline ):
            self.capture_obj.stop()
        pipleine_cfg        = rs.config()
        pipleine_cfg.enable_stream( rs.stream.color, *self.resolution, rs.format.bgr8, self.fps )
        pipleine_cfg.enable_stream( rs.stream.depth, *self.lidar_resolution, rs.format.z16, self.lidar_fps )
        self.capture_obj    = rs.pipeline()
        self.capture_obj.start( pipleine_cfg )  

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
        if isinstance( self.capture_obj, rs.pipeline ):
            self.capture_obj.stop()
        self.capture_obj = None


    def spin( self ):
        """
        Threaded Process: Reads camera data and streams it according to provided gstreamer config
        """
        while not self.sigterm.is_set():
            frame = None
            try:
                if isinstance( self.capture_obj, rs.pipeline ):
                    frame               = self.capture_obj.wait_for_frames()
                    with self._instance_lock:
                        self.frame_color    = np.asanyarray( frame.get_color_frame().get_data() )
                        self.frame_depth    = np.asanyarray( frame.get_depth_frame().get_data() ) 
            except Exception as e:
                if isinstance(self.logger,Logger):
                    self.logger.write( f"Failed with Exception: {e}", process_name=self.process_name )
                frame   = None
        
            if frame is not None and isinstance( self.stream_obj, cv.VideoWriter ):
                if self.stream_obj.isOpened():
                    with self._instance_lock:
                        if   self.stream_channel == 0 : 
                            self.stream_obj.write( image=cv.resize( self.frame_color, self.stream_resolution ) )
                        elif self.stream_channel == 1 : 
                            depth_heatmap = cv.applyColorMap( 
                                src=cv.convertScaleAbs( self.frame_depth, alpha = 0.5 ),
                                colormap=cv.COLORMAP_JET
                            )
                            self.stream_obj.write( image=cv.resize( depth_heatmap, self.stream_resolution  ) )


    def switch_channel( self ):
        """
        Switches streamed channel from RGB to Depth and Visa-Versa
        """
        with self._instance_lock:
            self.stream_channel = int( not bool(self.stream_channel) )


    def get_frame( self ) -> Tuple[ NDArray, NDArray ]:
        """
        Retrieves return code and frame information
        """
        with self._instance_lock:
            return deepcopy( self.frame_color ), deepcopy( self.frame_depth )