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

    def __init__(self,x,y,w,h):
        self.coord   = (x,y,w,h)
        self.center  = (x+w//2,y+h//2)
        # self.number  = N   #  Designation
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
        self.blocks = []        # The total of 64*64 blocks
        self.subblocks = []     # sub_blocks for comparison
        self.result_blocks = [] # The splited blocks calculated by Rate-Ditortion-Optimization
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
            self.EBMA_RDO(block=block, block_size=self.blockSize)  # 64*64

        # for block in self.result_blocks:
        #     print(f"The coordinates of block: {block.coord}")


        self.plot_Blocks()
        self.plot_motionField()
    
    def frame2blocks(self, blocks, size):
        """ Split the entire frame matrix into blocks by parapeter_size """
        # Initial Configuration
        if blocks is None:
            # cnt = 0         # count the sequence of big_block
            (H,W) = self.shape
            # Debug
            # print(f"shape = {self.shape}")
            for h in range(H//size):
                for w in range(W//size):
                    x=w*size ; y=h*size
                    self.blocks.append(Block(x, y, size, size))
                    # Debug
                    print(f"x = {x}, y={y}")
                    # cnt +=1
            # for future check if the searched block inside the frame
            Block.min = self.blocks[0].coord
            Block.max = self.blocks[-1].coord
            #Debug
            print(f"[min, max] = {Block.min, Block.max}")
            # print(f"len(blocks)={len(self.blocks)}, len(counter)={self.cnt}")
        # Split the designated block into 4 sub_blocks
        else:
            (X,Y,H,W) = blocks.coord
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

        # Defined a variable for Displaced Frame Difference & cost for comparison
        dfd_norm = 0 ; cost = 0

        # search the matched block in reference frame in search area
        for (dx, dy) in searchArea:
            (x, y, w, h) = block.coord
            if not block.check_inside_frame(x+dx, y+dy):
                continue
            x = x+dx ; y = y+dy

            # extract the block from reference frame
            block_r = self.reference[y:y+h, x:x+w]

            # calculate Displaced Frame Difference
            if self.dfd == 0:
                dfd_norm = utils.MAD(block_c, block_r)
            else:
                dfd_norm = utils.MSE(block_c, block_r)
            # elif self.dfd == 2:
            #     dfd_norm = utils.SAD(block_c, block_r)
            # else:
            #     dfd_norm = utils.SSD(block_c, block_r)

            if dfd_norm < dfd_norm_min:  # replace dfd_norm make it minimize
                block.mv = (dx, dy)
                block.calculate_mv_amp()
                dfd_norm_min = dfd_norm

        # Implement Rate Detortion Optimization to get cost by inputing block_size and dfd_norm
        cost = self.calculate_cost(block_size, dfd_norm)

        # Debug
        # print(f"block[{block.number}]dfd_norm = {dfd_norm}")
        # print(f"self.blocks.index(block)={self.blocks.index(block)}")
        # Debug
        print(f"The block coordinates: {block.coord}, block_size={block_size}, cost={cost}")
        s_block_cost = 0                # variable for Mixing all dfd_norm from 4 sub-block
        if block_size/2 >= 16:
            self.frame2blocks(block, block_size//2) 
            for s_block in self.subblocks:
                s_block_cost += self.EBMA_RDO(s_block, block_size//2)
                #Debug
                # print(f"The subblock: {s_block.coord}, block_size={block_size}, cost={s_block_cost}")
            if s_block_cost < cost:
                print(f"s_block_cost={s_block_cost},cost={cost}")
                if not self.result_blocks:
                    for sb in self.subblocks:
                        self.result_blocks.append(sb)
                else:
                    b_position = self.result_blocks.index(block)               # Get the position of the bigger block from BlockList
                    self.result_blocks.pop(b_position)                         # delete the bigger Block
                    self.result_blocks[b_position:b_position] = self.subblocks # append 4 sub_blocks into the rare of the position of the bigger block
            else:
                self.result_blocks.append(block)    # append the bigger block
                #Debug
                print("(Bebug)")
                for debug in self.result_blocks:
                    print(f"block_coordinate={debug.coord}", end="->")
                print()
                self.subblocks.clear()
        
        return cost
    
    def calculate_cost(self, block_size, dfd_norm):
        """Calculates the number of bits used to encode the motion vector and residual of an image block.

        Args:
            search_range: The search range of the motion vector.
            quantization_factor: The quantization factor.
            image_block_size: The size of the image block.

        Returns:
            The number of cost J, which foluma is J=dfd_norm + Lagrange multiplier * Total_Bitrate encoded by motion vector and residual of an image block.
        """
        search_range = block_size*2+1

        # Calculate the number of bits used to encode the motion vector.
        bits_mv = math.log2(math.pow(search_range, 2))

        # Calculate the number of bits used to encode the residual.
        bits_res = math.pow(block_size,2)*8

        # Calculate the total number of bits used to encode the motion vector and residual.
        bits_total = bits_mv + bits_res

        # Calculate the Rate and supposed the video frame rate is 30
        Rate = bits_total / 30 

        return dfd_norm + self.LM*Rate

    def plot_Blocks(self):
        """ Visualized All of blocks included bigsize mediansize smallsize """

    def plot_motionField(self):
        """ Construct the motion field from motion-vectors """

    def Blocks2Color(self, colorspace):
        """ Turn all of motion-vectors from motion field into color by color_space2"""

    def getPSNR(self):
        """ Print the PSNR from predicted frame """


        