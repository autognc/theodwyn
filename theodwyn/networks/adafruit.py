import busio
from rohan.common.base_networks         import NetworkBase
from rohan.common.logging               import Logger
from adafruit_servokit                  import ServoKit
from typing                             import List, Tuple, Optional

try:
    import threading
except ImportError:
    threading = None

class Adafruit_PCA9685( NetworkBase ):
    """
    Network Class for Adafruit PCA9685
    :param n_channels: Number of channels of the PCA9685
    :param SDA: Identifier for the connected SDA pin
    :param SCL: Identifier for the connected SCL pin
    :param init_cmd: Initial command to send via the send method (in many cases this must be called prior to reading data)
    :param actuation_range: A list of actuation ranges for the connected servos
    :param min_max_pwr: A list of tuples containing the (min,max) pulse widths for a respective servo
    :param safety_bounds: An optional list of safety bounds to restrict the actuation of servos (not used for determining pwm signals)
    :param logger: rohan Logger() instance
    """
    process_name    : str = "Adafruit PCA9685"
    
    SDA             : int
    SCL             : int
    freq            : int = 100000
    i2c_bus         : busio.I2C 
    channels        : int 
    servokit        : Optional[ServoKit] = None
    init_cmd        : Optional[List[float]]
    actuation_range : Optional[List[float]]
    min_max_pwr     : Optional[List[Tuple[int,int]]]
    safety_bounds   : Optional[List[Optional[Tuple[float,float]]]]

    def __init__( 
        self,
        channels        : int,
        SDA             : int,
        SCL             : int, 
        init_cmd        : Optional[List[float]] = None,
        actuation_range : Optional[List[float]] = None,
        min_max_pwr     : Optional[List[Tuple[int,int]]] = None,
        safety_bounds   : Optional[List[Optional[Tuple[float,float]]]] = None,
        logger          : Optional[Logger] = None,
        **config_kwargs
    ):
        super().__init__(
            logger=logger
        )
        self.load( 
            channels        = channels,
            SDA             = SDA,
            SCL             = SCL,
            init_cmd        = init_cmd,
            actuation_range = actuation_range,
            min_max_pwr     = min_max_pwr,
            safety_bounds   = safety_bounds,
            **config_kwargs 
        )        
        

    def connect( self ) -> None: 
        """
        Connect network interfaces
        """
        self.i2c_bus = busio.I2C( scl=self.SCL, sda=self.SDA, frequency=self.freq )
        if threading is not None:
            self.i2c_bus._lock.acquire()
        self.servokit = ServoKit( channels = self.channels , i2c = self.i2c_bus )

        if self.actuation_range is not None:
            for i, actuation_range_i in enumerate(self.actuation_range):
                self.servokit.servo[i].actuation_range = actuation_range_i

        if self.min_max_pwr is not None:
            for i, min_max_i in enumerate(self.min_max_pwr):
                self.servokit.servo[i].set_pulse_width_range(min_pulse=min_max_i[0],max_pulse=min_max_i[1])
        
        self.send( self.channels*[0.] ) if self.init_cmd is None else self.send( self.init_cmd )

        

    def disconnect( self ) -> None:
        """
        Disconnect network interfaces
        """
        self.send( self.channels*[0.] ) if self.init_cmd is None else self.send( self.init_cmd )
        if threading is not None:
            self.i2c_bus._lock.release()
        self.i2c_bus.deinit()

        self.servokit = None


    def send(
        self, 
        cmd : List[float], 
        **kwargs
    ) -> None:
        """
        Send command through network
        :param cmd: Command to be sent
        """

        if self.servokit is None : raise RuntimeError('Connection to IIC has not been established')

        def check_safe( 
            safety_bound : Optional[Tuple[float,float]], 
            cmd : float
        ) -> None:
            """
            Checks whether fulfilling a given command will violate saftey contraints
            """
            if safety_bound is not None:
                if cmd_i < safety_bound[0] or cmd_i > safety_bound[1]:
                    raise ValueError('Angle is out of prescribed safety bound')
            return

        for i, cmd_i in enumerate( cmd ):
            try:
                if self.safety_bounds is not None: check_safe( safety_bound=self.safety_bounds[i], cmd=cmd_i )
                self.servokit.servo[i].angle = cmd_i
            except ValueError as e:
                if isinstance(self.logger,Logger): 
                    self.logger.write(
                        f'[network][cmd] Warning from network as follows -> Value Error:{e}',
                        process_name=self.process_name
                    )
