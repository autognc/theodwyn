import threading
import cv2
import os
import sys
import glob
import numpy                                    as     np
from math                                       import sqrt, pi
from rohan.common.base_stacks                   import ThreadedStackBase
from rohan.common.logging                       import Logger
from rohan.data.classes                         import StackConfiguration
from theodwyn.data.writers                      import CSVWriter
from theodwyn.navigations.mekf                  import MEKF
from theodwyn.cameras.ximea                     import XIMEA
from typing                                     import Optional, List, Union, Any
from time                                       import time, strftime, sleep
from rohan.utils.timers                         import IntervalTimer
from copy                                       import deepcopy, copy
# from queue                                      import Queue

import pdb

class DebugMEKF(ThreadedStackBase):
    """ DebugMEKF is a class that provides the functionality to debug the Pose Prediction Terrier (PPT) stack """
    process_name    = "DebugMEKF"

    def __init__(self, config: StackConfiguration):
        super().__init__(config = config)
        self.image_index    = 0
        self.image_files    = []
        self.source_mode    = config.navigation_configs['source_mode']
        
        if self.source_mode == "camera":
            # initialize the camera using the provided camera configuration.
            raise ValueError("Camera mode not supported yet")
        elif self.source_mode == "offline":
            image_prefix    = config.navigation_configs["image_prefix"]
            image_dir       = config.navigation_configs["image_dir"]
            if not image_dir:
                raise ValueError("Offline mode selected but no 'image_dir' provided in configuration.")
            # build a sorted list of image file paths.
            self.image_files    = sorted(
                                            glob.glob(os.path.join(image_dir, image_prefix + "*.jpg")) +
                                            glob.glob(os.path.join(image_dir, image_prefix +  "*.png"))
                                )
            self.image_index = 0
        else:
            raise ValueError(f"Invalid source_mode specified: {self.source_mode}")

        # set measurement dt from config
        self.meas_dt    = config.navigation_configs["meas_dt"]

        # this attribute will later hold the navigation (MEKF) instance
        self.navigation: Optional[MEKF] = None

        # add our long-running image processing loop ?
        self.add_threaded_method(target = self.process_stack)

    def acquire_image(self) -> Optional[np.ndarray]:
        """
        Acquire an image either from a live camera or an offline directory

        Outputs:
        image (np.ndarray or None): the acquired image, or None if no image is available
        """
        if self.source_mode == "camera":
            # Call the XIMEA camera's get_frame() method.
            return self.camera.get_frame()
        elif self.source_mode == "offline":
            if self.image_index >= len(self.image_files):
                self.sigterm.set()
                return None, None
            image_path          = self.image_files[self.image_index]
            self.image_index    += 1
            image               = cv2.imread(image_path)
            return image, copy(self.image_index)
        return None, None

    def process_stack(self) -> None:
        """
        Long-running loop for image acquisition.
          - Acquires an image.
          - If the navigation (MEKF) instance is available, passes the image to it via pass_in_frame().
          - Sleeps for meas_dt seconds between acquisitions.
        """
        while not self.sigterm.is_set():
            image, img_index = self.acquire_image()
            if image is not None and self.navigation is not None:
                # Hand off the image to MEKF 
                # MEKF.spin_meas_model will pick up this frame, perform preprocessing/inference,
                # and handle first-measurement synchronization 
                self.navigation.pass_in_frame(image, img_cnt = img_index)
            sleep(self.meas_dt)

    def process(self,
                network: Optional[Any] = None,
                camera: Optional[Any] = None,
                controller: Optional[Any] = None,
                guidance: Optional[Any] = None,
                navigation: Optional[MEKF] = None,
                logger: Optional[Logger] = None) -> None:
        """
        This process() method is called repeatedly by the base class's spin() loop.
        Here, we simply ensure that the navigation (MEKF) instance is stored.
        Since image acquisition is already handled in process_stack() (spawned in __init__),
        no additional repeated processing is needed.
        """
        if self.navigation is None and navigation is not None:
            self.navigation = navigation