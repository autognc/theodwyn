import sys
import json
from rohan.common.logging        import Logger
from theodwyn.networks.serial    import SerialConnection
from time                        import sleep

class Serial_4MW( SerialConnection ):
    def __init__(
        self, 
        port, 
        port_config, 
        min_interval,
        logger = None
    ):
        super().__init__(
            port=port,
            port_config=port_config,
            min_interval=min_interval,
            logger=logger,
        )

    def connect( self ):
        super().connect()
        self.send( int(0).to_bytes(1,byteorder=sys.byteorder) )     
    
    def disconnect( self ):
        self.send( int(0).to_bytes(1,byteorder=sys.byteorder) ) 
        return super().disconnect()

if __name__ == "__main__":

    with open("./config/debug_serial.json") as file:
        json_data = json.load(file)

    with Logger() as logger:
        with Serial_4MW(**json_data[0],logger=logger,min_interval=50E-6) as ser_port1, \
             Serial_4MW(**json_data[1],logger=logger,min_interval=50E-6) as ser_port2:            


            def send_command( wheel : int, direction : int ):
                """
                Sends command with 50% throttle
                """
                def i2bs( a_int : int ):
                    return int(a_int).to_bytes(1,byteorder=sys.byteorder)

                if wheel == 0:
                    if direction != 0:
                        ser_port1.send( i2bs(32) if direction>0 else i2bs(95) )
                    else:
                        ser_port1.send( i2bs(64) )
                elif wheel == 1:
                    if direction != 0:
                        ser_port2.send( i2bs(32) if direction>0 else i2bs(95) )
                    else:
                        ser_port2.send( i2bs(64) )
                elif wheel == 2:
                    if direction != 0:
                        ser_port1.send( i2bs(159) if direction>0 else i2bs(223) )
                    else:
                        ser_port1.send( i2bs(192) )
                else:
                    if direction != 0:
                        ser_port2.send( i2bs(159) if direction>0 else i2bs(223) )
                    else:
                        ser_port2.send( i2bs(192) )


            # Spin all the wheels at 50% throttle
            send_command( wheel=0, direction=1 )
            sleep(0.5)
            send_command( wheel=0, direction=0 )
            sleep(0.5)
            send_command( wheel=0, direction=-1 )
            sleep(0.5)
            send_command( wheel=0, direction=0 )
            sleep(0.5)

            send_command( wheel=1, direction=1 )
            sleep(0.5)
            send_command( wheel=1, direction=0 )
            sleep(0.5)
            send_command( wheel=1, direction=-1 )
            sleep(0.5)
            send_command( wheel=1, direction=0 )
            sleep(0.5)

            send_command( wheel=2, direction=1 )
            sleep(0.5)
            send_command( wheel=2, direction=0 )
            sleep(0.5)
            send_command( wheel=2, direction=-1 )
            sleep(0.5)
            send_command( wheel=2, direction=0 )
            sleep(0.5)

            send_command( wheel=3, direction=1 )
            sleep(0.5)
            send_command( wheel=3, direction=0 )
            sleep(0.5)
            send_command( wheel=3, direction=-1 )
            sleep(0.5)
            send_command( wheel=3, direction=0 )
            sleep(0.5)