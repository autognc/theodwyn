import  numpy                                  as np
from    rohan.common.logging                   import Logger
from    rohan.common.base_controllers          import ControllerBase
from    theodwyn.data.readers                  import CSVReader
from    numpy.typing                           import NDArray
from    typing                                 import Union, TypeVar, List, Optional
from    time                                   import perf_counter

SelfViconFeedback = TypeVar("SelfViconFeedback", bound="ViconFeedback" )
class ViconFeedback(ControllerBase):
    """
    Class for handling Vicon Feedback Control
    :param logger: rohan Logger() context
    """

    process_name : str = "ViconFeedback Controller"
    
    p_gain              : NDArray
    csv_handler         : Optional[CSVReader]  = None # TODO: Fix this 
    traj_start_time     : Optional[float]      = None

    def __init__(
        self,
        p_gain              : NDArray          = np.zeros((2,2)),
        csv_trajfilename    : Optional[str]    = None,
        logger              : Optional[Logger] = None, 
        **kwargs,
    ):
        super().__init__(
            logger=logger,
        )
        if not csv_trajfilename is None: self.csv_handler = CSVReader(filename=csv_trajfilename)
        self.p_gain = p_gain
        self.load( **kwargs )
    
    def init_controller( self ):
        if not self.csv_handler is None: 
            self.csv_handler.open_file()

    def deinit_controller( self ):
        if not self.csv_handler is None: 
            self.csv_handler.close_file()

    def get_control_time(self):
        if not self.traj_start_time is None:
            return perf_counter() - self.traj_start_time
        return None

    def determine_control( 
        self, 
        pos_xy_vicon : Optional[NDArray] = None 
    ) -> Union[ List[float], NDArray ]:
    
        v_out = np.zeros((3,))
        csv_data_t = self.csv_handler.read_nextrow()
        if csv_data_t is None: 
            return v_out

        if self.traj_start_time is None: self.traj_start_time = perf_counter()
        if 'time' in csv_data_t: 
            curr_time = perf_counter() - self.traj_start_time
            while float(csv_data_t['time']) < curr_time:
                csv_data_t = self.csv_handler.read_nextrow()
                if csv_data_t is None: 
                    return v_out
            
            # >>> Feedforward Input
            # TODO: Add angular velocity support
            v_out = np.array( [ float(csv_data_t['v_x']), float(csv_data_t['v_y']), 0. ] ).flatten()

            # >>> Feedback Input
            if not pos_xy_vicon is None:
                p_err       = np.array( [ float(csv_data_t['x']), float(csv_data_t['y']) ] ).flatten() - pos_xy_vicon.flatten()
                v_out[0:2] += ( self.p_gain @ p_err ).flatten()
        
        return v_out