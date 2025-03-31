import zmq
import struct
from zmq.sugar.socket                   import Socket
from rohan.common.base_networks         import ThreadedNetworkBase
from rohan.common.logging               import Logger
from typing                             import Optional, List, Tuple

class _ZMQThreadedConnection( ThreadedNetworkBase ):
    """
    Class handling zmq socket connections
    :param socket_type: ZMQ Socket type
    :param addr: Address of socket to connect or bind
    :param data_format: Format of packets in accordance with python struct package
    :param topic: Topic/group by which socket is communicating
    :param logger: rohan Logger() instance
    """

    process_name : str = "ZMQ Connection (threaded)"

    context     : Optional[zmq.Context] = None
    socket      : Optional[Socket] = None
    socket_type : int
    addr        : str
    data_format : str
    topic       : str

    def __init__(
        self,
        socket_type : int,
        addr        : str,
        data_format : str,
        topic       : str = "unnameddtopic",
        logger      : Optional[Logger] = None
    ):
        ThreadedNetworkBase.__init__(
            self,
            logger=logger
        )
        self.socket_type = socket_type
        self.addr        = addr
        self.topic       = topic.encode('ascii')
        self.data_format = data_format

    def disconnect(self):
        """
        Closes ZMQ socket and terminates associated contexts
        """
        if isinstance( self.socket, Socket ) : self.socket.close()
        self.socket     = None
        if isinstance( self.context, zmq.Context ) : self.context.term()
        self.context    = None
        if isinstance(self.logger,Logger): 
            self.logger.write(
                f'Disconnected',
                process_name=self.process_name
            )

"""
=========================================================================================
"""

class ZMQDish( _ZMQThreadedConnection ):
    """
    ZMQ Dish Socket handler
    :param socket_type: ZMQ Socket type
    :param addr: Address of socket to connect or bind
    :param data_format: Format of packets in accordance with python struct package
    :param topic: Topic/group by which socket is communicating
    :param logger: rohan Logger() instance
    """

    process_name : str                           = "ZMQ Dish (threaded)"
    data         : Optional[List[float]]         = None
    btopic       : Optional[List[float]]         = None
    timeo        : int                           = 100
    log_timeo    : bool                          = True

    def __init__(
        self,
        addr        : str,
        data_format : str,
        timeo       : int               = 1000,
        topic       : str               = "unnameddrtopic",
        log_timeo   : bool              = True,
        logger      : Optional[Logger]  = None
    ):
        super().__init__(
            socket_type = zmq.DISH,
            addr        = addr,
            data_format = data_format,
            topic       = topic,
            logger      = logger
        ) 
        self.process_name   = f"{self.process_name} to {topic}"
        self.timeo          = timeo
        self.log_timeo      = log_timeo
        self.add_threaded_method( target=self.spin )

    def connect(self):
        """
        Connects to ZMQ socket at provided address on provided topic
        """
        self.context = zmq.Context()
        self.socket  = self.context.socket( self.socket_type )
        self.socket.setsockopt(zmq.LINGER,0)
        self.socket.setsockopt(zmq.CONFLATE,1)
        self.socket.bind( addr = self.addr )
        self.socket.join( group = self.topic )
        if isinstance(self.logger,Logger): 
            self.logger.write(
                f'Connected to {self.addr} in group {self.topic}',
                process_name=self.process_name
            )

    def spin( self ):
        """
        Threaded Process: Collects incoming data from ZMQ socket at provided address on provided topic
        """
        while not self.sigterm.is_set():
            if self.socket.poll( timeout=self.timeo, flags=zmq.POLLIN ):
                binary_topic, data_buffer = self.socket.recv(copy=False).bytes.split(b' ',1)
                decoded_btopic = binary_topic.decode(encoding = 'ascii')
                with self._instance_lock:
                    self.btopic = decoded_btopic 
                    self.data   = struct.unpack( self.data_format, data_buffer )
            else:
                if self.log_timeo and self.logger: 
                    self.logger.write(
                        f'Timed out with no message recieved ... continuing',
                        process_name=self.process_name
                    )
                with self._instance_lock:
                    self.btopic = None 
                    self.data   = None

    def recv( self ) -> Tuple[Optional[str],Optional[List[float]]]:
        """
        Provided user with incoming data being read from ZMQ socket at provided address on provided topic
        """
        with self._instance_lock:
            return self.btopic, self.data

    
class ZMQRadio( _ZMQThreadedConnection ):
    """
    ZMQ Radio Socket handler
    :param socket_type: ZMQ Socket type
    :param addr: Address of socket to connect or bind
    :param data_format: Format of packets in accordance with python struct package
    :param topic: Topic/group by which socket is communicating
    :param logger: rohan Logger() instance
    """
    process_name : str = "ZMQ Radio (threaded)"

    publisher_format : str

    def __init__(
        self,
        addr        : str,
        data_format : str,
        topic       : str = "unnameddrtopic",
        logger      : Optional[Logger] = None
    ):
        super().__init__(
            socket_type = zmq.RADIO,
            addr        = addr,
            data_format = data_format,
            topic       = topic,
            logger      = logger
        ) 
        self.publisher_format = f"<{len(self.topic)}ss{self.data_format}"
        self.process_name = f"{self.process_name} to {topic}"
        
    def connect(self):
        """
        Connects to ZMQ socket at provided address on provided topic
        """
        self.context = zmq.Context.instance()
        self.socket  = self.context.socket( self.socket_type )
        self.socket.setsockopt(zmq.LINGER,0)
        self.socket.setsockopt(zmq.CONFLATE,1)
        self.socket.connect( self.addr  )
        if isinstance(self.logger,Logger): 
            self.logger.write(
                f'Connected to {self.addr}',
                process_name=self.process_name
            )
