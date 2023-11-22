# Get two image from Dataset & plot Current frame with Reference frame attached the motion field
# ================================================================

import numpy as np
import cv2

class DATA():
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    def __init__(self, path, CurrentFrame_N, ReferenceFrame_N):
         self.path = path
         self.CFN = CurrentFrame_N
         self.RFN = ReferenceFrame_N
         self.shape = (None, None)

    def read_frames(self):
        """ Read current frame and reference frame, turn them into gray scale """
        try:
            current = cv2.imread(self.path+f"im{self.CFN}.png", 1)
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            reference = cv2.imread(self.path+f"im{self.RFN}.png", 1)
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        except:
            print("Error in Path or Frame Count")
            exit()
        self.shape = current.shape[:2]
        self.frame_inp = [self.CFN, current_gray, self.RFN, reference_gray, current, reference]
        print("[INFO] Video Import Completed")

    def visualize(self, motionField, current_P):
        """Put 4 frames together to show gui."""

        h = 70 ; w = 10
        H,W = self.shape
        HH,WW = h+2*H+20, 2*(W+w)
        frame_gray = np.ones((HH,WW), dtype="uint8")*255
        frame = np.ones((HH,WW,3), dtype="uint8")*255

        cv2.putText(frame, f"anchor-im{self.CFN:03d}", (w, h-4), DATA.FONT, 0.4, 0, 1)
        cv2.putText(frame, f"target-im{self.RFN:03d}", (w+W, h-4), DATA.FONT, 0.4, 0, 1)
        cv2.putText(frame_gray, "motion field", (w, h+2*H+10), DATA.FONT, 0.4, 0, 1)
        cv2.putText(frame, "predicted anchor", (w+W, h+2*H+10), DATA.FONT, 0.4, 0, 1)

        frame[h:h+H, w:w+W] = self.frame_inp[4] 
        frame[h:h+H, w+W:w+2*W] = self.frame_inp[5] 
        frame_gray[h+H:h+2*H, w:w+W] = motionField 
        frame[h+H:h+2*H, w+W:w+2*W] = current_P

        return frame, motionField