import pygame as pg
from rohan.common.logging                   import Logger
from rohan.common.base_controllers          import ControllerBase
from numpy.typing                           import NDArray
from typing                                 import Union, TypeVar, List, Optional

SelfXboxGamePad = TypeVar("SelfXboxGamePad", bound="XboxGamePad" )
class XboxGamePad(ControllerBase):
    """
    Class for handling inputs from connected Xbox-Controller (NOTE: Mapping are 
    specific to xbox-controller and thing is not a general gamepad class)
    :param logger: rohan Logger() context
    """

    process_name : str = "Xbox Controller"
    
    _player_id   : int

    def __init__(
        self,
        player_id   : int              = 0,
        logger      : Optional[Logger] = None,
        **config_kwargs
    ):
        super().__init__(
            logger=logger
        )
        self._player_id = player_id
        self.load( **config_kwargs )
    
    def init_controller( self ):
        """
        Initialize XBOX GAMEPAD by initializeing pygame and respective joystick
        """
        pg.init()
        pg.joystick.init()
        if pg.joystick.get_count() > self._player_id:
            self.jstick = pg.joystick.Joystick( self._player_id )
            if isinstance(self.logger,Logger): 
                self.logger.write(
                    f'Connected to {self.jstick.get_name()}',
                    process_name=self.process_name
                )
        else:
            raise RuntimeError('Unable to connect to a controller ... Exiting')

    def deinit_controller( self ):
        """
        Cleans up pygame artifacts openned resulting from connecting XBOX GAMEPAD
        """
        pg.quit()

    def determine_control( self ) -> Union[ List[float], NDArray ]:
        """
        Reads Input from connected Xbox Gamepad. Mapping found below:
        ----------------------------------------- 
        Controller Axies in JOYAXISMOTION event   
        ----------------------------------------- 
         0 :  0., # Left-Analog Stick Horizontal  
         1 :  0., # Left-Analog Stick Vertical    
         2 : -1., # Left Trigger                  
         3 :  0., # Right-Analog Stick Horizontal 
         4 :  0., # Right-Analog Stick Vertical    
         5 : -1., # Right Trigger                 
        ----------------------------------------- 
        Controller Buttons in JOYAXISMOTION event 
        ----------------------------------------- 
         6  | 0 :  0., A-Button                   
         7  | 1 :  0., B-Button                   
         8  | 2 :  0., X-Button                   
         9  | 3 :  0., Y-Button                   
         10 | 4 :  0., Left-Bumper                
         11 | 5 :  0., Right-Bumper               
         12 | 6 :  0., Left-Select                
         13 | 7 :  0., Right-Select               
         14 | 8 :  0., Middle-Select              
         15 | 9 :  0., Left-Stick                 
         16 | 10:  0., Right-Stick                
        ----------------------------------------- 
        """
        pg.event.get() 
        return [ self.jstick.get_axis(i) for i in range( self.jstick.get_numaxes() ) ] \
             + [ float(self.jstick.get_button(i)) for i in range( self.jstick.get_numbuttons() ) ]
        

        