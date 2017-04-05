# Frame Handler
#
#
# George
#
# Just does all the preprocessing on an incoming frame
# 1. rectify using a homography
# 2. threhsold gradient and color spaces
# 3. transform images and polynomials back to normal view


import cv2
import numpy as np

class FrameHandler():
    
    def __init__(self,
                 H, 
                 rows, 
                 cols,
                 s_thresh,     # s-channel threshold tuple
                 h_thresh,     # l-channel threshold tuple
                 dir_thresh,   # gradient angle threshold tuple
                 mag_thresh,    # gradient norm threshold tuple
                 xgrad_thresh,  # absolute gradient threshold in the x-direction
                 sobel_kernel = 15, # derivative operator kernel size
                 win_margin = 90    # margin of strips from the "center", used to crop left and righgt lanes
                 ):
        
        # assign parameters
        self.H = H
        
        self.Hinv = np.linalg.inv(H)
        self.s_thresh = s_thresh
        self.h_thresh = h_thresh
        self.dir_thresh = dir_thresh
        self.mag_thresh = mag_thresh
        self.sobel_kernel = sobel_kernel        
        self.xgrad_thresh = xgrad_thresh
        self.win_margin = win_margin        
        # image rows, cols
        self.rows = rows
        self.cols = cols
        
        # The default lines defining the cropping strips for the lanes
        self.crop_ll = ( (int)(self.cols/4) - 30 , self.rows)
        self.crop_ul = ( (int)(self.cols/4) + 247+20, self.rows-(int)(self.rows/3) - 40 )
        
        self.crop_lr = ( (int)(self.cols/4) - 30 + 840 -20, self.rows)
       
        self.crop_ur = ( (int)(self.cols/4) + 250 + 160 -50 , self.rows-(int)(self.rows/3) - 40 ) 
        self.crop_horiz = self.rows-(int)(self.rows/3) - 40 
        
        # current image (None for none)        
        self.img = None
        # rectified image
        self.img_warped = None
        # HLS image
        self.img_hls = None
        self.h_channel = None
        self.s_channel = None
        # the final binary image(s)
        self.h_binary = None
        self.s_binary = None
        self.gray = None
        self.mag_binary = None
        self.dir_binary = None
        self.color_binary = None
        self.img_binary = None
        self.sobelx = None
        self.sobely = None
        self.gradx_binary = None
        self.out_img = None # display the binary
    # Get the gradient magnitude- based binary image    
    def mag_threshold(self,                    
                    sobelx,         # ready-made x-gradient
                    sobely,          # ready-made y-gradient
                    thresh=(0, 255)
                   ):
    
    
        norm_sobel = np.sqrt(sobelx*sobelx + sobely*sobely)
    
        scaled_sobel = np.uint8(255*norm_sobel / np.max(norm_sobel))
    
        binimg = np.zeros( (self.rows, self.cols))
    
        # implicit thresholding using the scaled image for masking
        binimg[ (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[0]) ] = 1
    
        return binimg
    
    # directional gradient threshold    
    def dir_threshold(self,
                      sobelx,
                      sobely,
                      thresh = (0, 90)
                      ):
    
        angle_sobel = np.arctan2(np.absolute(sobely) , np.absolute(sobelx))
    
        #scaled_sobel = np.uint8(255*norm_sobel / np.max(norm_sobel))
    
        binimg = np.zeros_like( sobelx)
    
        # implicit thresholding using the scaled image for masking
        binimg[ (angle_sobel >= thresh[0] * np.pi / 180.0) & (angle_sobel <= thresh[1] * np.pi / 180.0) ] = 1
    
        return binimg
    # gradient magontude thresholding in either direction ('x' or otherwise)
    def abs_grad_threshold(self, abs_sobel, orient='x', thresh=(0, 255)):
    
        if orient == 'x':    
            axisx = 1
        else:
            axisx = 0
            
        scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
    
        binimg = np.zeros_like( self.gray)
    
        # implicit thresholding using the scaled image for masking
        binimg[(scaled_sobel <= thresh[1]) & (scaled_sobel >= thresh[0])] = 1
    
        return binimg
    
    
    
    # crop the region of interest in an image
    def crop_roi(self):
        
#        ll = ( (int)(self.cols/4) - 30 - 50 , self.rows-40 )
#        ul = ( (int)(self.cols/4) + 247 - 40, self.rows-(int)(self.rows/3) - 40 )
#        lr = ( (int)(self.cols/4) - 30 + 840 + 50, self.rows-20)
#        ur = ( (int)(self.cols/4) + 250 + 160 + 40, self.rows-(int)(self.rows/3) - 40 ) 
        # the narrowing of the margin (just came up with the figure, althoug I could compute it)
        narrowing = 50
        # 1. Ok, fitting the two left lines as Y = A1x+B1 and Y = A2x+B2
        #print("Tuple : ", ll)
        
        # now fitting the left line        
        l1_x1 = self.crop_ll[0] - self.win_margin
        l1_y1 = self.crop_ll[1] 
        l1_x2 = self.crop_ul[0] - (self.win_margin - narrowing) # make it like a trapeziium (roughly follow perspective)
        l1_y2 = self.crop_ul[1]        
        Al1 = (l1_y2 - l1_y1) / (l1_x2 - l1_x1)
        Bl1 = l1_y1 - l1_x1 * Al1
        print("First line : ", Al1, ", ", Bl1)
        # now fitting the right line in the same way...
        l2_x1 = self.crop_ll[0] + self.win_margin
        l2_y1 = self.crop_ll[1] 
        l2_x2 = self.crop_ul[0] + (self.win_margin - narrowing) # make it like a trapezium (roughly follow perspective)
        l2_y2 = self.crop_ul[1]        
        Al2 = (l2_y2 - l2_y1) / (l2_x2 - l2_x1)
        Bl2 = l2_y1 - l2_x1 * Al2
        
        # 2. Now fitting right couple of lines
        # left line
        r1_x1 = self.crop_lr[0] - self.win_margin
        r1_y1 = self.crop_lr[1]
        r1_x2 = self.crop_ur[0] - (self.win_margin - narrowing)
        r1_y2 = self.crop_ur[1]        
        Ar1 = (r1_y2 - r1_y1) / (r1_x2 - r1_x1)
        Br1 = r1_y1 - r1_x1 * Ar1
        
        # right line
        r2_x1 = self.crop_lr[0] + self.win_margin
        r2_y1 = self.crop_lr[1]
        r2_x2 = self.crop_ur[0] + (self.win_margin - narrowing)
        r2_y2 = self.crop_ur[1]        
        Ar2 = (r2_y2 - r2_y1) / (r2_x2 - r2_x1)
        Br2 = r2_y1 - r2_x1 * Ar2
           
        #
        # Find the region inside the lines
        XX, YY = np.meshgrid(np.arange(0, self.cols), np.arange(0, self.rows))
#        region_thresholds = ( ( YY < (Al1 * XX + Bl1 ) ) & ( YY > (Al2 * XX + Bl2 ) ) ) | \
#                            ( ( YY < (Ar2 * XX + Br2 ) ) & ( YY > (Ar1 * XX + Br1 ) ) ) | \
#                            ( YY < self.crop_ur[1]) 
        region_thresholds = ( ( ( YY < (Al1 * XX + Bl1) ) | ( YY > (Al2 * XX + Bl2 ) ) ) & \
                            ( ( YY < (Ar2 * XX + Br2) ) | ( YY > (Ar1 * XX + Br1 ) ) ) ) | \
                            ( YY < self.crop_horiz) 
                            
        # Color pixels red which are inside the region of interest
        self.img_binary[region_thresholds] = 0
        
        #create the display image       
        self.out_img = np.dstack((self.img_binary, self.img_binary, self.img_binary))*255
        # Plot the lines in the display image
        self.out_img = cv2.polylines(self.out_img,[np.array([[l1_x1, l1_y1], [l1_x2, l1_y2]])],False,(0,0,255))
        self.out_img = cv2.polylines(self.out_img,[np.array([[l2_x1, l2_y1], [l2_x2, l2_y2]])],False,(0,0,255))
        self.out_img = cv2.polylines(self.out_img,[np.array([[r1_x1, r1_y1], [r1_x2, r1_y2]])],False,(0,0,255))
        self.out_img = cv2.polylines(self.out_img,[np.array([[r2_x1, r2_y1], [r2_x2, r2_y2]])],False,(0,0,255))
        
        
        
        
    
    
    # process the frame
    def process_frame(self, image):
        
        # make sure frame has the correct dimensions
        assert( (self.rows == image.shape[0]) and (self.cols == image.shape[1]) )
        self.img = image        
        # 1. Convert to HLS space
        self.img_hls =  cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        #self.h_channel = self.img_hls[:,:,0]
        self.s_channel = self.img_hls[:,:,2]
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)        
        
        # 2. Thresholding the s and l channels        
        self.s_binary = np.zeros_like(self.s_channel)
        self.s_binary[(self.s_channel >= self.s_thresh[0]) & (self.s_channel <= self.s_thresh[1]) ] = 1
        
        #self.h_binary = np.zeros_like(self.h_channel)
        #self.h_binary[(self.h_channel >= self.h_thresh[0]) & (self.h_channel <= self.h_thresh[1]) ] = 1
        
        # 3. Get the directional gradients from the s_channel / l_channel
        self.sobelx = np.abs(cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize = self.sobel_kernel))
        self.gradx_binary = self.abs_grad_threshold(self.sobelx, orient = 'x', thresh = self.xgrad_thresh)
        
        # final binary
        self.img_binary = np.zeros_like(self.s_channel)
        self.img_binary[(self.gradx_binary == 1) |  (self.s_binary == 1)] = 1
        
        # crop the left and right strips!!!!        
        self.crop_roi()
        
        # warp the binary
        self.img_warped = cv2.warpPerspective(self.img_binary,self.H,(self.cols, self.rows))