import csv
from theodwyn.networks.vicon    import ViconConnection, ViconData
from rohan.common.base          import _RohanThreading
from io                         import TextIOWrapper
from rohan.common.logging       import Logger
from rohan.utils.timers         import IntervalTimer
from typing                     import Optional, Any

ASSOCIATED_KEYS = [ 
    "frame", 
    "pos_x"  , "pos_y"  , "pos_z",
    "euler_x", "euler_y", "euler_z"
]
class ViconCollector(_RohanThreading):
    """
    Object for external collection of vicon data 
    :param host_addr: address of machine hosting vicon hardware and Tracker software
    :param object_name: name of vicon Tracker object to collect data of
    :param filename: csv filename to save to
    :param min_interval: minimum interval between vicon collections
    :param logger: rohan logging instance 
    """

    process_name: str                      = "Vicon Collector"

    vicon_cnnct : ViconConnection
    object_name : str
    itrvl_timer : IntervalTimer
    filename    : str
    file        : Optional[TextIOWrapper]  = None
    csv_writer  : Optional[csv.DictWriter] = None 
    logger      : Optional[Logger]         = None

    def __init__(
        self,
        host_addr       : str,
        object_name     : str,
        filename        : str,
        min_interval    : int               = 0,
        logger          : Optional[Logger]  = None
    ):
        super().__init__()
        self.vicon_cnnct    = ViconConnection(host_addr=host_addr)
        self.object_name    = object_name
        self.filename       = filename
        self.itrvl_timer    = IntervalTimer( interval= min_interval )
        self.logger         = logger
        self.add_threaded_method( self.spin )


    def open_file( self ) : 
        """
        Open file specified by name in initializer
        """
        if isinstance(self.file, TextIOWrapper):
            self.close_file()
        try:
            self.file       = open(self.filename,"w",newline='')
            self.csv_writer = csv.DictWriter( self.file, fieldnames=ASSOCIATED_KEYS )
            self.csv_writer.writeheader()
        except Exception as e:
            self.file = None
            if isinstance(self.logger,Logger): 
                self.logger.write(
                    f'Failed to open {self.filename} with exception {e}',
                    process_name=self.process_name
                )


    def close_file( self ):
        """
        Close file specified by name in initializer if open
        """
        self.csv_writer = None
        if isinstance(self.file,TextIOWrapper):
            self.file.close()
        self.file = None


    def __enter__( self ):
        self.open_file()
        ret_msg = self.vicon_cnnct.connect()
        if isinstance(self.logger,Logger): 
            self.logger.write(
                ret_msg,
                process_name=self.process_name
            )
        self.start_spin()
        return self


    def __exit__( self, exception_type, exception_value, traceback ):
        self.stop_spin()
        ret_msg = self.vicon_cnnct.disconnect()
        if isinstance(self.logger,Logger): 
            self.logger.write(
                ret_msg,
                process_name=self.process_name
            )    
        self.close_file()
       

    def pull_data( self ):
        """
        Pull data from the vicon system associated with object specified by name in the initializer
        """
        return self.vicon_cnnct.recv_pose(object_name=self.object_name)
    

    def write_to_file( self, data : ViconData ):
        """
        Write vicon data to csv file specified by name in initializer
        """
        data_out = [ data.framenumber, *data.position, *data.orientation_euler ]
        self.csv_writer.writerow(  dict( zip(ASSOCIATED_KEYS,data_out) ) )


    def spin( self ):
        while not self.sigterm.is_set():
            self.itrvl_timer.await_interval()
            data = self.pull_data()
            if data.succeeded: 
                self.write_to_file( data=data )

