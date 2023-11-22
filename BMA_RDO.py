# Block_Matching_Algorithm with Rate-Distortion-Optimization
# ================================================================
import utils
import numpy as np
import itertools 
import cv2
import math
import copy

class Block():
    # min = None
    # max = None 
    max_mp_amp = 0

    def __init__(self,x,y,w,h, max):
        self.coord   = (x,y,w,h)
        self.center  = (x+w//2,y+h//2)
        self.mv      = (0,0)
        self.mv_amp  = 0

        # specifying the scope of a block
        self.min     = 0
        self.max     = max
    
    def check_inside_frame(self,x,y):
        """check if the searched box inside the target frame"""
        check = True
        if x<self.min or x>self.max[0] \
            or y<self.min or y>self.max[1]:                                                   
            check = False
        return check
    def calculate_mv_amp(self):
        """calculate L2 norm of motion-vector"""
        amp = (self.mv[0]**2 + self.mv[1]**2)**0.5
        if amp > Block.max_mp_amp:
            Block.max_mp_amp = amp
        self.mv_amp = amp
    

class EBMA_RDO():
    def __init__(self, dfd, blockSize_o, LM, motionIntensity=True):
        """
            dfd: {0: MAD, 1: MSE, 2: SAD, 3: SSD}
            BlockSize: 64*64, 32*32, 16*16
            searchMethod: Exhaustive
            searchRange: (int) +- BlockSize*2
        """
        # given parameters
        self.dfd = dfd
        self.blockSize = blockSize_o
        self.motionIntensity = motionIntensity

        # given frames 
        self.current = None
        self.reference = None
        self.shape = None

        # blocks in current
        self.blocks = []        # The total of 64*64 blocks 
        self.result_blocks = [] # The splited blocks calculated by Rate-Ditortion-Optimization
        self.LM = LM            # Lagrange Multipler

        # return frames 
        self.prediction = None
        self.prediction_color = None
        self.motionField = None
    
    def step(self, frame_inp):
        self.current = frame_inp[1]     # Current_frame
        self.reference = frame_inp[3]   # Reference_frame
        self.current_color = frame_inp[4]
        self.reference_color = frame_inp[5]
        self.shape = self.current.shape

        print(f"self.shape={self.shape}")

        # Split the entire frame into 64*64 blocks
        self.frame2blocks(None, self.blockSize, None)

        # Exhaustive Block Matching Algo by recursive way
        for index, block in enumerate(self.blocks):
            _, _ = self.EBMA_RDO(block=block, block_size=self.blockSize)  # 64*64
            # Debug
            print("(Bebug)")
            for i, debug in enumerate(self.result_blocks):
                print(f"result_block[{i}]_coordinate={debug.coord}", end="->")
            print()
            print(f"---------------------------Go on to the block[{index+1}]--------------------------")

        self.plot_Blocks()
        self.plot_motionField()
        self.blocks2frame()
        self.blocks2frame_color()
    
    def frame2blocks(self, blocks, size, subblocks):
        """ Split the entire frame matrix into blocks by parapeter_size """
        # Initial Configuration
        if blocks is None:
            (H,W) = self.shape
            # Debug
            # print(f"shape = {self.shape}")
            max_block_range = (W-size, H-size)
            for h in range(H//size):
                for w in range(W//size):
                    x=w*size ; y=h*size
                    self.blocks.append(Block(x, y, size, size, max=max_block_range))
            # for future check if the searched block inside the frame
            # Block.min = self.blocks[0].coord
            # Block.max = self.blocks[-1].coord
            #Debug
            # print(f"[min, max] = {Block.min, Block.max}")
            # print(f"len(blocks)={len(self.blocks)}, len(counter)={self.cnt}")
        
        # Split the designated block into 4 sub_blocks
        else:
            max_block_range = (self.shape[1]-size, self.shape[0]-size)
            (X,Y,H,W) = blocks.coord
            for h in range(H//size):
                for w in range(W//size):
                    x=X+w*size ; y=Y+h*size
                    subblocks.append(Block(x, y, size, size, max=max_block_range))
            # Block.min = subblocks[0].coord
            # Block.max = subblocks[-1].coord

    def EBMA_RDO(self, block, block_size):
        """ Exhaustive Search Algorithm with Rate Distortion Optimized by recursive """

        searchRange = block_size//8
        dx = dy = [i for i in range(-searchRange,searchRange+1)]
        searchArea = [r for r in itertools.product(dx,dy)]
        # get block coordinates for current frame
        (x, y, w, h) = block.coord
        # extract the block from current frame
        block_c = self.current[y:y+h, x:x+w]

        # displaced frame difference := initially infinity
        dfd_norm_min = np.Inf

        # Defined a variable for Displaced Frame Difference & cost for comparison
        cost = 0

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

            if dfd_norm < dfd_norm_min:  # replace dfd_norm make it minimize
                block.mv = (dx, dy)
                block.calculate_mv_amp()
                dfd_norm_min = dfd_norm

        # Implement Rate Detortion Optimization to get cost by inputing block_size and min_dfd_norm
        cost = self.calculate_cost(block.coord, dfd_norm_min, block.mv)
            

        # Debug
        print(f"The block coordinates: {block.coord}, block_size={block_size}, block.mv={block.mv}, dfd_norm_min={dfd_norm_min}, cost={cost}")
        print("----")
        Flag_Split = False       # A condition of RDO recursive

        if block_size//2 >= 16:
            # self.result_blocks.append(block)
            print(f"This block has been splitted-------------------------------------------")
            self.result_blocks.append(block)
            subblocks = []     # sub_blocks for comparison
            self.frame2blocks(block, block_size//2, subblocks)
            s_block_cost = 0                # variable for Mixing all min_dfd_norm from 4 sub-block
            for s_block in subblocks:
                sub_Flag, sub_cost = self.EBMA_RDO(s_block, block_size//2)
                s_block_cost += sub_cost
                Flag_Split = Flag_Split or sub_Flag
                print(f"The s_block coordinates: {s_block.coord}, s_block block_size={block_size//2}, Flag_Split={Flag_Split}", end=' ')
                print()
                # print(f"cost = {cost}, s_block_cost = {s_block_cost}")     
                #Debug
                # print(f"The subblock: {s_block.coord}, block_size={block_size}, cost={s_block_cost}")
            s_block_cost /= 3
            print(f"The s_block_cost: {s_block_cost}")
            print("-------------------")
            if s_block_cost < cost:
                print(f"s_block_cost={s_block_cost},cost={cost}")
                try:
                    b_position = self.result_blocks.index(block)
                except ValueError:
                    for sb in subblocks:
                        self.result_blocks.append(sb)              
                else:  
                    self.result_blocks[b_position:b_position] = subblocks   
                    self.result_blocks.pop(b_position)
                Flag_Split = True
            else:
                if Flag_Split is True:
                    b_position = self.result_blocks.index(block)
                    self.result_blocks.pop(b_position)
                else:
                    for sb in subblocks:
                        try:
                            sb_position = self.result_blocks.index(sb)
                            self.result_blocks.pop(sb_position)
                        except ValueError:
                            pass
        return Flag_Split, cost
    
    def calculate_cost_original(self, block_size, dfd_norm):
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
        Rate = bits_total

        return dfd_norm + self.LM*Rate

    def calculate_cost(self, block_coord, dfd_norm, motion_vector):
        """Calculates the number of bits used to encode the motion vector and residual of an image block.

        Args:
            block_coord:    The coordinate of current block 
            dfd_norm:       The minimum of dfd norm of the block.
            motion_vector:  The minimum of distortion of motion_vector

        Returns:
            The cost J, which is the sum of the dfd norm and the Lagrange multiplier times the total bitrate encoded by the motion vector and residual of an image block.
        """

        (x, y, w, h) = block_coord
        # print(f"block_coord={block_coord}, motion_vector={motion_vector}, dfd_norm={dfd_norm}")
        # MV_search_range = math.sqrt(motion_vector[0]**2 + motion_vector[1]**2)
        # MV_search_range = w
        # Calculate the number of bits used to encode the motion vector which is the minimum distortion.
        # bits_mv = math.ceil(math.log2(MV_search_range))
        bits_mv = 0
        
        block_c = self.current[y:y+h, x:x+w]
        (x_ref, y_ref) = x+motion_vector[0], y+motion_vector[1]
        block_ref = self.reference[y_ref:y_ref+w, x_ref:x_ref+h]
        
        # Calculate the luminance of difference within current frame and reference
        residual = np.sum(np.abs(block_c - block_ref))

        try:
            bits_res = math.ceil(math.log2(residual))
        except ValueError as e:
            bits_res = 0

        # Calculate the total number of bits used to encode the motion vector and residual.
        bits_total = bits_mv + bits_res

        # Calculate the cost using the dfd norm and the Lagrange multiplier times the rate
        cost = dfd_norm + self.LM * bits_total

        # print(f"dfd_norm={dfd_norm}, bits_mv: {bits_mv}, bits_res={bits_res}, cost={cost}")

        return cost

    def plot_Blocks(self):
        """ Visualized All of blocks included bigsize mediansize smallsize """

    def plot_motionField(self):
        """Construct the motion field from motion-vectors"""
        frame = np.zeros(self.shape,dtype="uint8")           

        for block in self.result_blocks:
            intensity = round(255 * block.mv_amp/Block.max_mv_amp) if self.motionIntensity else 255
            intensity = 100 if intensity<100 else intensity
            (x2,y2) = block.mv[0]+block.center[0], block.mv[1]+block.center[1]
            cv2.arrowedLine(frame, block.center, (x2,y2), intensity, 1, tipLength=0.3)
        
        self.motionField = frame

    def blocks2frame(self):
        """Construct the predicted"""
        # frame = np.zeros(self.shape,dtype="uint8")
        target_frame = copy.deepcopy(self.reference)
        for block in self.result_blocks:
            # get block coordinates for current frame
            (x,y,w,h) = block.coord
            # extract the block from current frame
            block_c = self.current[y:y+h, x:x+w]
            # append motion-vector to prediction coordinates 
            (x,y) = x+block.mv[0], y+block.mv[1]
            # shift the block to new coordinates
            target_frame[y:y+h, x:x+w] = block_c

        self.prediction = target_frame

    def blocks2frame_color(self):
        """Construct the predicted"""
        # frame = np.zeros(self.shape,dtype="uint8")
        target_frame = copy.deepcopy(self.reference_color)
        for block in self.result_blocks:
            # get block coordinates for current frame
            (x,y,w,h) = block.coord
            # extract the block from current frame
            block_c = self.current_color[y:y+h, x:x+w]
            # append motion-vector to prediction coordinates 
            (x,y) = x+block.mv[0], y+block.mv[1]
            # shift the block to new coordinates
            target_frame[y:y+h, x:x+w] = block_c

        self.prediction_color = target_frame

    def getPSNR(self, img1, img2):
        """ Print the PSNR from predicted frame """
        mse = np.mean((img1 - img2) ** 2)
        max_intensity = 255
        return 20 * np.log10(max_intensity) - 10 * np.log10(mse)
        
    def motionField2color(self, ColorSpace, blocksize):
        frame = np.zeros((256, 448 , 3),dtype="uint8")  
        ColorSpace = cv2.resize(ColorSpace, (blocksize,blocksize), interpolation=cv2.INTER_AREA)
        cs_center = (blocksize//2, blocksize//2)
        for block in self.result_blocks:
            (x,y,w,h) = block.coord
            (x_mv,y_mv) = (block.mv[0], block.mv[1])
            color_x = cs_center[0] + x_mv
            color_y = cs_center[1] + y_mv
            frame[y:y+h, x:x+w,:] = ColorSpace[color_y,color_x,:]
        return frame
        