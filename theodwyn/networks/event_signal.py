import threading
from theodwyn.networks.comm_prot    import ZMQDish
from rohan.common.logging           import Logger
from typing                         import Optional, Tuple, List

class EventSignal(ZMQDish):

    process_name : str                           = "Event Signal"
    event_flag   : threading.Event

    def __init__(
        self,
        addr        : str,
        data_format : str,
        timeo       : int               = 1000,
        topic       : str               = "unnameddrtopic",
        logger      : Optional[Logger]  = None
    ):
        super().__init__(
            addr        = addr,
            data_format = data_format,
            timeo       = timeo,
            topic       = topic,
            log_timeo   = False,
            logger      = logger
        )
        self.event_flag = threading.Event()


class SingleEventSignal(EventSignal):

    process_name : str                           = "Single Event Signal"
    event_flag   : threading.Event

    def __init__(
        self,
        addr        : str,
        data_format : str,
        timeo       : int               = 1000,
        topic       : str               = "unnameddrtopic",
        logger      : Optional[Logger]  = None
    ):
        super().__init__(
            addr        = addr,
            data_format = data_format,
            timeo       = timeo,
            topic       = topic,
            logger      = logger
        )

    def recv( self ) -> Tuple[Optional[str],Optional[List[float]]]:
        """
        Provided user with incoming data being read from ZMQ socket at provided address on provided topic
        """
        if not self.sigterm.is_set():
            with self._instance_lock:
                topic, data = self.btopic, self.data
            
            if not data is None:
                self.sigterm.set()
                if self.logger:
                    self.logger.write(
                        "Event data recieved. Passing data then closing",
                        process_name=self.process_name
                    )
                return topic, data
            else:
                return None, None
        else:
            if self.logger:
                self.logger.write(
                    "New event data was requested after replied ONCE. Replying with None",
                    process_name=self.process_name
                )
            return None, None
