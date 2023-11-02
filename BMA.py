# Block_Matching_Algorithm with Rate-Distortion-Optimization
# ================================================================
import utils
import numpy as np
import itertools 
import cv2
import math

class Block():
    min = None
    max = None 
    max_mp_amp = 0

    def __init__(self,x,y,w,h,N):
        self.coord   = (x,y,w,h)
        self.center  = (x+w//2,y+h//2)
        self.number  = N   # Designation
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
        if amp > Block.max_mp_amp:
            Block.max_mp_amp = amp
        self.mv_amp = amp
    

class EBMA_RDO():
    def __init__(self, dfd, blockSize_o, LM):
        """
            dfd: {0: MAD, 1: MSE, 2: SAD, 3: SSD}
            BlockSize: 64*64, 32*32, 16*16
            searchMethod: Exhaustive
            searchRange: (int) +- BlockSize*2
        """
        # given parameters
        self.dfd = dfd
        self.blockSize = blockSize_o

        # given frames 
        self.current = None
        self.reference = None
        self.shape = None

        # blocks in current
        self.blocks = []      # turnout blocks
        self.subblocks = None   # sub_blocks for comparison
        self.LM = LM            # Lagrange Multipler

        # return frames 
        self.prediction = None
        self.motionField = None
    
    def step(self, frame_inp):
        self.current = frame_inp[1]     # Current_frame
        self.reference = frame_inp[3]   # Reference_frame
        self.shape = self.current.shape

        # Split the entire frame into 64*64 blocks
        self.frame2blocks(None, self.blockSize)

        # Exhaustive Block Matching Algo by recursive way
        for block in self.blocks:
            self.EBMA_RDO(block=block, block_size=self.blockSize)


        self.plot_Blocks()
        self.plot_motionField()
    
    def frame2blocks(self, blocks, size):
        """ Split the entire frame matrix into blocks by parapeter_size """
        # Initial Configuration
        if blocks is None:
            cnt = 0         # count the sequence of big_block
            (H,W) = self.shape
            # Debug
            # print(f"shape = {self.shape}")
            for h in range(H//size):
                for w in range(W//size):
                    x=w*size ; y=h*size
                    self.blocks.append(Block(x, y, size, size, cnt))
                    # Debug
                    print(f"x = {x}, y={y}")
                    cnt +=1
            # for future check if the searched block inside the frame
            Block.min = self.blocks[0].coord
            Block.max = self.blocks[-1].coord
            #Debug
            print(f"[min, max] = {Block.min, Block.max}")
            # print(f"len(blocks)={len(self.blocks)}, len(counter)={self.cnt}")
        # Split the designated block into 4 sub_blocks
        else:
            (X,Y,H,W) = (blocks.coor[2], blocks.coor[3])
            for h in range(H//size):
                for w in range(W//size):
                    x=X+w*size ; y=Y+h*size
                    self.subblocks.append(Block(x, y, size, size))

    def EBMA_RDO(self, block, block_size):
        """ Exhaustive Search Algorithm with Rate Distortion Optimized by recursive """

        searchRange = block_size
        dx = dy = [i for i in range(-searchRange,searchRange+1)]
        searchArea = [r for r in itertools.product(dx,dy)]
        # print(searchArea)
        # get block coordinates for current frame
        (x, y, w, h) = block.coord
        # extract the block from current frame
        block_c = self.current[y:y+h, x:x+w]

        # displaced frame difference := initially infinity
        dfd_norm_min = np.Inf

        # search the matched block in reference frame in search area
        for (dx, dy) in searchArea:
            (x, y, w, h) = block.coord
            if not block.check_inside_frame(x+dx, y+dy):
                continue
            x = x+dx ; y = y+dy

            # extract the block from reference frame
            block_r = self.reference[y:y+h, x:x+w]

            #calculate Displaced Frame Difference
            if self.dfd == 0:
                dfd_norm = utils.MAD(block_c, block_r)
            elif self.dfd == 1:
                dfd_norm = utils.MSE(block_c, block_r)
            elif self.dfd == 2:
                dfd_norm = utils.SAD(block_c, block_r)
            else:
                dfd_norm = utils.SSD(block_c, block_r)

            if dfd_norm < dfd_norm_min:
                block.mv = (dx, dy)
                block.calculate_mv_amp()
                dfd_norm_min = dfd_norm


        # self.LM   # Lagrangian parameter
        # bitrate = math.pow(self.blockSize,2)*8  # according to block_size block_size^2 * 8(bits)

    def plot_Blocks(self):
        """ Visualized All of blocks included bigsize mediansize smallsize """

    def plot_motionField(self):
        """ Construct the motion field from motion-vectors """

    def Blocks2Color(self, colorspace):
        """ Turn all of motion-vectors from motion field into color by color_space2"""

    def getPSNR(self):
        """ Print the PSNR from predicted frame """


        