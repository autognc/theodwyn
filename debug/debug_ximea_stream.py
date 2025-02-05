
import os
import sys
import cv2
from time import sleep

from rohan.common.logging import Logger
# os.system('cat /sys/module/usbcore/parameters/usbfs_memory_mb')
# os.system("/bin/bash -c 'tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0'")
# os.system('cat /sys/module/usbcore/parameters/usbfs_memory_mb')

from theodwynbyebye.cameras.ximea_cam                 import XIMEA

if __name__ == "__main__":
    with Logger() as logger:
        with XIMEA(logger=logger) as camera:
            while True: 
                cv2.imshow("test",camera.get_frame())
                cv2.waitKey(1)
            
        
