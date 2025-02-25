import sys
import json
import numpy                    as     np
from rohan.common.logging       import Logger
from theodwyn.networks.serial   import SerialConnection
from typing                     import Dict, Optional, Any, List

SSSERIAL_C0_CMDS = range(1,128)
SSSERIAL_C1_CMDS = range(128,256)
THROTTLE_C0_CMDS = np.linspace( -1., 1., num=len(SSSERIAL_C0_CMDS) )
THROTTLE_C1_CMDS = np.linspace( -1., 1., num=len(SSSERIAL_C1_CMDS) )

class SabertoothSimpleSerial( SerialConnection ):
    def __init__(
        self,
        port            : str,
        port_config     : Dict[str,Any],
        min_interval    : float             = 0.,
        logger          : Optional[Logger]  = None,
    ):
        super().__init__(
            port=port,
            port_config=port_config,
            min_interval=min_interval,
            logger=logger,
        )

    def connect( self ):
        super().connect()
        self.send( int(0).to_bytes(1,byteorder=sys.byteorder) ) # Stops all motors
    
    def disconnect( self ):
        self.send( int(0).to_bytes(1,byteorder=sys.byteorder) )  # Stops all motors
        return super().disconnect()
    

    def send_c0_throttle(
        self,
        cmd : float
    ):
        self.send( 
            int(
                SSSERIAL_C0_CMDS[
                    np.digitize(cmd,THROTTLE_C0_CMDS)-1
                ]
            ).to_bytes(1,byteorder=sys.byteorder)
        )


    def send_c1_throttle(
        self,
        cmd : float
    ):
        self.send( 
            int(
                SSSERIAL_C1_CMDS[
                    np.digitize(cmd,THROTTLE_C1_CMDS)-1
                ]
            ).to_bytes(1,byteorder=sys.byteorder)
        )       


    def send_throttles(
        self,
        cmds    : List[float]
    ):
        self.send_c0_throttle( cmds[0] )
        self.send_c1_throttle( cmds[1] )