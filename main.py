from Dataset import DATA
from BMA_RDO import EBMA_RDO
from tqdm import tqdm_gui
import numpy as np
import cv2
import time
import math

# Blocking Matching Parameters with RDO
# reference1 https://ieeexplore.ieee.org/document/733497
# reference2 https://ieeexplore.ieee.org/document/733495
# reference_code https://github.com/001honi/video-processing

# Adjustable Parameters
# ================================================================
# there are four dfd that we can use 0:MAD 1:MSE 2:SAD 3:SSD
# block_size will be block_size/1, block_size/2, block_size/4
dfd=0 ; block_size=64 ; Lagrange_Multipler = 2
# Lagrange_Multipler = math.sqrt(math.sqrt(0.8*math.pow(25, 2)))


# Import Data Parameters from Dataset
# ================================================================
path = "./0005/"
CurrentFrame_N = 1
ReferenceFrame_N = 4

data = DATA(path, CurrentFrame_N, ReferenceFrame_N)
data.read_frames()
(H, W) = data.shape
colorspace = cv2.imread("./Color_Space/color_space2.png",1)

# Calculate all of blocks of motion vector 
# ================================================================
bm = EBMA_RDO(dfd=dfd, 
              blockSize_o=block_size, 
              LM=Lagrange_Multipler, 
              motionIntensity=False)
if dfd==0:
    print("Distortion models is MAD")
else:
    print("Distortion models is MSE")
# elif dfd==2:
#     print("Distortion models is SAD")
# else:
#     print("Distortion models is SSD")
start_time  = time.time()

# wait for restore
bm.step(data.frame_inp)  
currentP_color = bm.prediction_color
currentP = bm.prediction
motionField = bm.motionField

elapsed_time = time.time()-start_time

# Visualize the results and print PSNR
# ================================================================
out, motionfield = data.visualize(motionField, currentP_color)
out_color = bm.motionField2color(colorspace, block_size)
cv2.imshow("HW1_Demo", out)
cv2.imshow("HW1_MotionField", motionfield)
cv2.imshow("HW1_MotionField_color", out_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imshow("HW1_")
print(f"PSNR for GroundTruch: {bm.getPSNR(data.frame_inp[1], data.frame_inp[3])}dB")
print(f"PSNR for Predicted Frame: {bm.getPSNR(currentP, data.frame_inp[3])}dB")
print(f"Elapsed time: {elapsed_time:.3f} seconds")

