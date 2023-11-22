# Author: Selahaddin HONI | 001honi@github
# March, 2021
# ============================================================================================
from utils import MAD, MSE 
import copy
import numpy as np
import itertools
import cv2

class Block():
    min = None
    max = None
    max_mv_amp = 0

    def __init__(self,x,y,w,h):
        self.coord   = (x,y,w,h)
        self.center  = (x+w//2,y+h//2)
        self.mv      = (0,0)
        self.mv_amp  = 0

    def check_inside_frame(self,x,y):
        """check if the searched box inside the target frame"""
        check = True
        if x<Block.min[0] or x>Block.max[0] \
            or y<Block.min[1] or y>Block.max[1]:
            check = False
        return check

    def calculate_mv_amp(self):
        """calculate L2 norm of motion-vector"""
        amp = (self.mv[0]**2 + self.mv[1]**2)**0.5
        if amp > Block.max_mv_amp:
            Block.max_mv_amp = amp
        self.mv_amp = amp


class EBMA():

    def __init__(self,dfd,blockSize_o,searchMethod,searchRange,motionIntensity=True):
        """
        dfd : {0:MAD, 1:MSE} Displaced frame difference 
        blockSize : {(sizeH,sizeW)}
        searchMethod : {0:Exhaustive, 1:Three-Step}
        searchRange : (int) +/- pixelwise range 
        """
        # given parameters
        self.mse = dfd
        self.blockSize = blockSize_o
        self.searchMethod = searchMethod
        self.searchRange = searchRange
        self.motionIntensity = motionIntensity

        # given frames
        self.anchor = None
        self.target = None
        self.shape  = None

        self.anchor_color=None
        self.target_color=None

        # blocks in anchor
        self.blocks = None

        # return frames
        self.prediction     = None
        self.motionField = None
    
    def step(self,anchor,target, anchor_c, target_c):
        """One-step run for given frame pair."""

        self.anchor = anchor
        self.target = target
        self.anchor_color=anchor_c
        self.target_color=target_c
        self.shape  = anchor.shape

        self.frame2blocks()

        if self.searchMethod == 0:
            self.EBMA()
        elif self.searchMethod == 1:
            self.ThreeStepSearch()
        else:
            print("Search Method does not exist!")

        self.plot_motionField()
        self.blocks2frame()  
        self.blocks2frame_color()   

    def frame2blocks(self):
        """Divides the frame matrix into block objects."""

        (H,W) = self.shape 
        (sizeH,sizeW) = (self.blockSize, self.blockSize)

        self.blocks = []
        for h in range(H//sizeH):
            for w in range(W//sizeW):
                # initialize Block() objects with 
                # upper-left coordinates and block size
                x = w*sizeW ; y = h*sizeH
                self.blocks.append(Block(x,y,sizeW,sizeH))

        # store the upper-left and bottom-right block coordinates
        # for future check if the searched block inside the frame
        Block.min = self.blocks[0].coord
        Block.max = self.blocks[-1].coord

    def plot_motionField(self):
        """Construct the motion field from motion-vectors"""
        frame = np.zeros(self.shape,dtype="uint8")           

        for block in self.blocks:
            intensity = round(255 * block.mv_amp/Block.max_mv_amp) if self.motionIntensity else 255
            intensity = 100 if intensity<100 else intensity
            (x2,y2) = block.mv[0]+block.center[0], block.mv[1]+block.center[1]
            cv2.arrowedLine(frame, block.center, (x2,y2), intensity, 1, tipLength=0.3)
        
        self.motionField = frame

    def EBMA(self):
        """Exhaustive Search Algorithm"""

        dx = dy = [i for i in range(-self.searchRange,self.searchRange+1)]
        searchArea = [r for r in itertools.product(dx,dy)]
        
        for block in self.blocks:
            print(block.min, block.max)
            # get block coordinates for anchor frame
            (x,y,w,h) = block.coord
            # extract the block from anchor frame
            block_a = self.anchor[y:y+h, x:x+w]

            # displaced frame difference := initially infinity
            dfd_norm_min = np.Inf

            # search the matched block in target frame in search area
            for (dx,dy) in searchArea:
                (x,y,w,h) = block.coord
                # check if the searched box inside the target frame
                if not block.check_inside_frame(x+dx,y+dy):
                    continue
                x = x+dx ; y = y+dy
                
                # extract the block from target frame
                block_t = self.target[y:y+h, x:x+w]

                # calculate displaced frame distance
                if self.mse:
                    dfd_norm = MSE(block_a,block_t)
                else:
                    dfd_norm = MAD(block_a,block_t)

                if dfd_norm < dfd_norm_min:
                    # set the difference as motion-vector
                    block.mv = (dx,dy)
                    block.calculate_mv_amp()
                    dfd_norm_min = dfd_norm

    def plot_motionField(self):
        """Construct the motion field from motion-vectors"""
        frame = np.zeros(self.shape,dtype="uint8")           

        for block in self.blocks:
            intensity = round(255 * block.mv_amp/Block.max_mv_amp) if self.motionIntensity else 255
            intensity = 100 if intensity<100 else intensity
            (x2,y2) = block.mv[0]+block.center[0], block.mv[1]+block.center[1]
            cv2.arrowedLine(frame, block.center, (x2,y2), intensity, 1, tipLength=0.3)
        
        self.motionField = frame

    def blocks2frame(self):
        """Construct the predicted"""
        # frame = np.zeros(self.shape,dtype="uint8")
        target_frame = copy.deepcopy(self.target)
        for block in self.blocks:
            # get block coordinates for current frame
            (x,y,w,h) = block.coord
            # extract the block from current frame
            block_c = self.anchor[y:y+h, x:x+w]
            # append motion-vector to prediction coordinates 
            (x,y) = x+block.mv[0], y+block.mv[1]
            # shift the block to new coordinates
            target_frame[y:y+h, x:x+w] = block_c

        self.prediction = target_frame

    def motionField2color(self, ColorSpace, blocksize):
        frame = np.zeros((256, 448 , 3),dtype="uint8")  
        ColorSpace = cv2.resize(ColorSpace, (blocksize,blocksize), interpolation=cv2.INTER_AREA)
        cs_center = (blocksize//2, blocksize//2)
        for block in self.blocks:
            (x,y,w,h) = block.coord
            (x_mv,y_mv) = (block.mv[0], block.mv[1])
            color_x = cs_center[0] + x_mv
            color_y = cs_center[1] + y_mv
            if color_x>64:
                color_x=63
            if color_y>64:
                color_y=63
            frame[y:y+h, x:x+w,:] = ColorSpace[color_y,color_x,:]
        return frame

    def getPSNR(self, img1, img2):
        """ Print the PSNR from predicted frame """
        mse = np.mean((img1 - img2) ** 2)
        max_intensity = 255
        return 20 * np.log10(max_intensity) - 10 * np.log10(mse)

    def blocks2frame_color(self):
        """Construct the predicted"""
        # frame = np.zeros(self.shape,dtype="uint8")
        target_frame = copy.deepcopy(self.target_color)
        for block in self.blocks:
            # get block coordinates for current frame
            (x,y,w,h) = block.coord
            # extract the block from current frame
            block_c = self.anchor_color[y:y+h, x:x+w]
            # append motion-vector to prediction coordinates 
            (x,y) = x+block.mv[0], y+block.mv[1]
            # shift the block to new coordinates
            target_frame[y:y+h, x:x+w] = block_c

        self.prediction_color = target_frame
