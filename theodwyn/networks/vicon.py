import pyvicon_datastream          as pv
from pyvicon_datastream.tools      import ObjectTracker
from rohan.common.base_networks    import NetworkBase
from rohan.common.logging          import Logger
from typing                        import Optional, List
from dataclasses                   import dataclass, field

# TODO: Add Additional Output Data Options
@dataclass
class ViconData:
    succeeded           : bool              = False
    latency             : Optional[int]     = None
    framenumber         : Optional[int]     = None
    position            : Optional[List]    = field(default_factory=list)
    orientation_quat    : Optional[List]    = field(default_factory=list)
    orientation_euler   : Optional[List]    = field(default_factory=list)

class _ViconObjectTracker( ObjectTracker ):
    def __init__( self, ip ):
        self.ip = ip
        self.is_connected = False
        self.vicon_client = pv.PyViconDatastream()

    def connect(self, ip=None):
        if ip is not None: # set
            print(f"Changing IP of Vicon Host to: {ip}")
            self.ip = ip
        
        if self.is_connected : self.vicon_client.disconnect()
        retcode = self.vicon_client.connect(self.ip)

        if retcode != pv.Result.Success:
            self.is_connected = False
            return f"Connection to {self.ip} failed"
        else:
            self.is_connected = True
            self.vicon_client.enable_segment_data()
            self.vicon_client.set_stream_mode(pv.StreamMode.ServerPush)
            self.vicon_client.set_axis_mapping(pv.Direction.Forward, pv.Direction.Left, pv.Direction.Up) 
            return f"Connection to {self.ip} successful"    
        
    def disconnect(self):
        if self.is_connected : 
            self.vicon_client.disconnect()
            self.is_connected = False

    def is_connected(self) : return True if self.is_connected else False


class ViconConnection( NetworkBase ):
    """
    Serial Connection Handler
    """

    host_addr   : str
    vicon_ds    : _ViconObjectTracker

    def __init__(
        self,
        host_addr       : str,
        logger          : Optional[Logger]  = None,
    ):
        super().__init__(
            logger=logger
        )
        self.load(
            host_addr   = host_addr,
        )
        self.vicon_ds   = _ViconObjectTracker( ip=host_addr )

    def connect(self):
        """
        Connect to vicon system
        """
        ret_msg = self.vicon_ds.connect()
        if isinstance(self.logger,Logger): 
            self.logger.write(
                ret_msg,
                process_name=self.process_name
            )

    def disconnect(self):
        """
        Disconnect from vicon system
        """
        ret_msg = self.vicon_ds.disconnect()
        if isinstance(self.logger,Logger): 
            self.logger.write(
                ret_msg,
                process_name=self.process_name
            )

    def recv_pose(
      self,
      object_name : str,
      ret_quat : bool = True
    ) -> ViconData:
        """
        Query pose of specified object from vicon system
        """
        data = ViconData()
        if self.vicon_ds.is_connected:
            out = self.vicon_ds.get_pose( object_name, ret_quat=ret_quat )
            if not ( out == False) and len(out[2])>0:
                (latency, framenumber, pose_coll)= out
                pose = pose_coll[0][2:]
                data.succeeded              = True
                data.latency                = latency
                data.framenumber            = framenumber
                data.position               = list( pose[0:3] )

                if ret_quat:
                    data.orientation_quat   = list( pose[3:7] )
                else:
                    data.orientation_euler  = list( pose[3:6] )
        return data