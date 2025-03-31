from    rohan.common.logging                   import Logger
from    rohan.common.base_guidances            import GuidanceBase
from    theodwyn.data.readers                  import CSVReader
from    numpy.typing                           import NDArray
from    typing                                 import Union, TypeVar, List, Optional
from    time                                   import perf_counter

SelfCSVInterpreter = TypeVar("SelfCSVInterpreter", bound="CSVInterpreter" )
class CSVInterpreter( GuidanceBase ):
    process_name : str = "CSV Interpreter Guidance "
    
    csv_handler         : Optional[CSVReader]  = None # TODO: Fix this 
    traj_start_time     : Optional[float]      = None
    csv_trajfilename    : str 
    time_keyname        : str

    def __init__(
        self,
        csv_trajfilename    : Optional[str]     = None,
        time_keyname        : Union[str,int]    = "time",
        logger              : Optional[Logger]  = None, 
        **kwargs,
    ):
        super().__init__(
            logger=logger,
        )
        if not csv_trajfilename is None: 
            self.csv_handler = CSVReader(filename=csv_trajfilename)
            self.csv_trajfilename = csv_trajfilename
        self.time_keyname = time_keyname
        self.load( **kwargs )
    
    def init_guidance( self ):
        if not self.csv_handler is None: 
            self.csv_handler.open_file()

    def deinit_guidance( self ):
        if not self.csv_handler is None: 
            self.csv_handler.close_file()

    def get_guidance_time(self):
        if not self.traj_start_time is None:
            return perf_counter() - self.traj_start_time
        return None
    
    def reset_guidance(self):
        if not self.csv_handler is None: 
            self.csv_handler.reset_iterator()
        self.traj_start_time = None

    def get_init_guidance(self):
        if not self.csv_trajfilename is None: 
            tmp = CSVReader(filename=self.csv_trajfilename)
            tmp.open_file()             # >> 1. Open file with tmp handler
            data = tmp.read_nextrow()   # >> 2. Read first row
            tmp.close_file()            # >> 3. Close file with tmp handler
            return data
        return None

    def determine_guidance( 
        self,
    ) -> Optional[ Union[ List[float], NDArray ] ]:
    
        csv_data_t = self.csv_handler.read_nextrow()
        if csv_data_t is None: 
            return None 

        if self.traj_start_time is None: self.traj_start_time = perf_counter()

        if self.time_keyname in csv_data_t: 
            curr_time = perf_counter() - self.traj_start_time
            while float(csv_data_t[self.time_keyname]) < curr_time:
                csv_data_t = self.csv_handler.read_nextrow()
                if csv_data_t is None: 
                    return None
                
            return csv_data_t
        else:
            return None