import cv2
import json
from rohan.common.logging        import Logger
from theodwyn.cameras.ximea      import XIMEA

if __name__ == "__main__":

    with open("./config/debug_ximea.json") as file:
        json_data = json.load(file)

    with Logger() as logger:
        with XIMEA(**json_data,logger=logger) as camera:
            while True: 
                cv2.imshow("XIMEA Stream",camera.get_frame())
                cv2.waitKey(1)
                # Check if the window is closed
            #     if cv2.getWindowProperty('XIMEA Stream', cv2.WND_PROP_VISIBLE) < 1:
            #         break
            # cv2.destroyAllWindows()
        
