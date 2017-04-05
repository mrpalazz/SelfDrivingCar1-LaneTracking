# Calibrator class
#
# George

import os
import cv2
import numpy as np

class calibrator():
    # Create a calibrator object    
    # by passing the directory with calibration images
    #def __init(self, path = "/home/george/SelfDrivingCar/CarND-Advanced-Lane-Lines/camera_cal/")__:
    def __init__(self, 
                 calibration_path = "./camera_cal/", # path to calibration files
                 filename_prefix = "calibration",
                 num_corners_x = 9, # number of corners horizontally in the chess pattern
                 num_corners_y = 6 # number of corners vertically in the chess pattern
                 ): 
        
        self.calibration_path = calibration_path
        self.filename_prefix = filename_prefix
        self.num_corners_x = num_corners_x
        self.num_corners_y = num_corners_y
        # corners in the entire sequence        
        self.sequence_corners = []
        # retrieve a list of the files in the given path
        # assuming every file inside path is an image        
        self.image_filenames = []        
        # camera intrinsics defaults (identity)
        self.K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float)
        self.dist_coefs = np.array([[0, 0, 0, 0, 0]], np.float)        
        # calibration done flag
        self.flag_calibrated = False;
        # setup a num_corners_x * num_cornmers_y grid coordinates 
        # as triplets: [0, 0, 0], [1, 0, 0],...
        self.grid_corners = np.zeros((self.num_corners_y * self.num_corners_x,3), np.float32)
        self.grid_corners[:,:2] = np.mgrid[0:self.num_corners_x,0:self.num_corners_y].T.reshape(-1,2)
        # The list of local grid coordinates per calibration image
        self.sequence_grid_corners = []
        
        for fname in os.listdir(self.calibration_path):
            if (fname[0:11]==self.filename_prefix):
                self.image_filenames.append(fname)
        # just making sure we opened-up the correct path
        assert( len(self.image_filenames)>0 )
    
    # Do the job!
    def calibrate(self, 
                  show_corners = True # use this to enable/disable corner display on image
                  ):
        assert(len(self.image_filenames) > 0)
        #print(self.image_filenames)
        # termination criteria for use in subpixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # reset sequence corners list
        self.sequence_corners = []
        self.sequence_grid_corners = []
        # looping through calibration images
        for img_filename in self.image_filenames:
            # open the image
            img = cv2.imread(self.calibration_path+img_filename)
            
            # obtain the grayscale version
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Now find the corners
            corners_found, corners = cv2.findChessboardCorners(gray, 
                                                               (self.num_corners_x, self.num_corners_y),
                                                                None)
            # if corners were detected, let's do something with them            
            if corners_found:  
                if show_corners:
                    # create a named window
                    cv2.namedWindow("calibration image")
                    # and start this blessed window thread 
                    # otherwise we woint be able to get rid of the window!!!!!
                    cv2.startWindowThread()
                # refine with sub-pixel accuracy
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                self.sequence_corners.append(corners2)
                # now simply inserting a copy of local grid coordinates
                # to the global list of local grid coordinates per calibration image
                self.sequence_grid_corners.append(self.grid_corners)
                # draw the corners 
                if show_corners:
                    # npot necessary, but get a copy of the original just in case                
                    drawimg = cv2.drawChessboardCorners(img, 
                                                        (self.num_corners_x, self.num_corners_y), 
                                                        corners2,
                                                        corners_found)
                    # now show the image
                    cv2.namedWindow("calibration image")
                    cv2.imshow("calibration image", drawimg)
                    # wait for the user to press a key                    
                    cv2.waitKey(-1) 
        # All done! destroy the image window (if necessary)
        cv2.destroyWindow("calibration image")
        
        #################### Actual Calibration below ###############
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.sequence_grid_corners, 
                                                           self.sequence_corners, 
                                                           gray.shape[::-1],None,None)

        if ret:        
            # assign intrinsics        
            self.K = mtx;
            # assign distortion coefficients
            self.dist_coefs = dist;
            self.flag_calibrated = True
    
    # radially undistort an image 
    def undistort(self, img):
        
        
        if self.flag_calibrated:
            undist_img = cv2.undistort(img, self.K, self.dist_coefs, None, None)
        else:
            undist_img = img
        
        return undist_img
        
    # reset the calibrator (i.e., reset flags and corner lists)
    def reset(self):
        
        self.flag_calibrated = False
        self.sequence_corners = [] 
        self.K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.dist_coefs = np.array([[0, 0, 0, 0, 0]])        
        
    def save(self, fname = "camera_params.txt"):
        if not self.flag_calibrated:
            return
        text_file = open(fname, "w")
        text_file.write(str(self.K[0][0])+" , "+str(self.K[0][2])+" , "+str(self.K[1][1])+" , "+str(self.K[1][2])+" , ")
        dist_str = ""
        for i in range(self.dist_coefs.shape[1]):     
            dist_str += str(self.dist_coefs[0][i])
            if i < self.dist_coefs.shape[1]-1:
                dist_str += " , "
        text_file.write(dist_str)        
        text_file.close()
    
    # load parameters 
    def load(self, fname = "./camera_params.txt"):
        
        try:        
            text_file = open(fname, 'r')
            paramstr = text_file.readline()
            text_file.close()
            param_strings = paramstr.split(',')
            print("Split param strings : ", param_strings)
            self.K[0][0] = np.float(param_strings[0])
            self.K[0][2] = np.float(param_strings[1])
            self.K[1][1] = np.float(param_strings[2])
            self.K[1][2] = np.float(param_strings[3])
            self.dist_coefs = np.zeros( (1, len(param_strings)-4), np.float)            
            for i in range(4, len(param_strings)):            
                self.dist_coefs[0][i-4] = float(param_strings[i])
            self.flag_calibrated = True
        except (OSError, IOError) as e:
            self.flag_calibrated = False
            
            
############# test the Calibrator here... #####################
#cal = calibrator()
#cal.load()
#if not cal.flag_calibrated:
#    cal.calibrate(show_corners = True)
#    print("Intrinsics : ", cal.K)
#    print("Distortion coeffcients : ", cal.dist_coefs)
#    img_name = cal.calibration_path+"calibration3.jpg"
#    img = cv2.imread(img_name)
#
#    cv2.namedWindow("skata")
#    cv2.startWindowThread()
#    cv2.imshow("skata", cal.undistort(img))
#    cv2.waitKey(-1)
#    cv2.destroyWindow("skata")
#    cal.save()
#else:
#    print("intrinsics: ", cal.K)
#    print("Distportion: ", cal.dist_coefs)
