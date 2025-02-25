from serial                         import Serial
from rohan.common.base_networks     import NetworkBase
from rohan.utils.timers             import IntervalTimer
from rohan.common.logging           import Logger
from typing                         import Optional, Any, Dict
from collections.abc                import ByteString

class SerialConnection( NetworkBase ):
    """
    Serial Connection Handler
    """

    ser         : Optional[Serial]
    port        : str
    port_config : Dict[str,Any]
    waiter      : IntervalTimer

    def __init__(
        self,
        port            : str,
        port_config     : Dict[str,Any],
        min_interval    : float             = 0.,
        logger          : Optional[Logger]  = None,
    ):
        super().__init__(
            logger=logger
        )
        self.load(
            port        = port,
            port_config = port_config
        )
        self.ser        = None
        self.waiter     = IntervalTimer( interval=min_interval )


    def connect(self):
        """
        Connect to serial port
        """
        self.ser    = Serial( port=self.port, **self.port_config )


    def disconnect(self):
        """
        Disconnect from serial port
        """
        self.ser.close()
        self.ser    = None


    def send(
        self,
        msg : ByteString
    ):
        """
        Send message through specified serial connection
        """
        if not self.ser is None and self.ser.is_open:
            self.waiter.await_interval()
            self.ser.write( msg )

    def recv(
      self      
    ):
        """
        Recieve message through specified serial connection
        """
        if isinstance( self.ser, Serial ) and self.ser.is_open:
            out = ''
            while self.ser.in_waiting > 0:
                out += self.ser.read()
            return out