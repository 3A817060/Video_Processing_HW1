# Get two image from Dataset & plot Current frame with Reference frame attached the motion field
# ================================================================

import numpy as np
import cv2

class DATA():
    def __init__(self, path, CurrentFrame_N, ReferenceFrame_N):
         self.path = path
         self.CFN = CurrentFrame_N
         self.RFN = ReferenceFrame_N
         self.shape = (None, None)

    def read_frames(self):
        """ Read current frame and target frame, turn them into gray scale """
        try:
            current = cv2.imread(self.path+f"im{self.CFN}.png", 0)
            target = cv2.imread(self.path+f"im{self.RFN}.png", 0)
        except:
            print("Error in Path or Frame Count")
            exit()
        self.shape = current.shape[:2]
        self.frame_inp = [self.CFN, current, self.RFN, target]
        print("[INFO] Video Import Completed")

    def visualize(self, colorspace, block2color):
        """ plot current frame, colorspace, reference frame, color field """

