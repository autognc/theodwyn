import cv2
from rohan.common.logging                       import Logger
from theodwyn.cameras.ximea_cam                 import XIMEA

if __name__ == "__main__":
    with Logger() as logger:
        with XIMEA(logger=logger) as camera:
            while True: 
                cv2.imshow("XIMEA Stream",camera.get_frame())
                cv2.waitKey(1)
                # Check if the window is closed
                if cv2.getWindowProperty('XIMEA Stream', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
        
