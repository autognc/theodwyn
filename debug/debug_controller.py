from theodwyn.controllers.gamepad   import XboxGamePad
from time                           import sleep

with XboxGamePad() as controller:
    while True:
        print( controller.determine_control() )
