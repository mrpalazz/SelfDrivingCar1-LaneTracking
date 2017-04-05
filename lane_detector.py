# ()Advanced) Lane Detector
# 
# George

import cv2
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

# OpenCV VideoCapture will not open files (although works fine with a camera)
# So, its faster (and safer) to open the videos with a standard python method....
import pylab
import imageio

from calibrator import calibrator
from frame_handler import FrameHandler
from LaneTracking import LaneTracker



################ CALIBRATION - CAMERA PARAMETERS ESTIMATION ######################

cal = calibrator()
# try to load camera parameters,
cal.load()
# otherwise do calibration
if not cal.flag_calibrated:
    print("Camera parameters file not found. Calibrating....")    
    if cal.fflag_calibrated == False:
        print("Something went wrong during calibration. Check your calibration image paths!")
        exit(0)
        
    cal.calibrate(show_corners = False)
    cal.save()
else:
    print("Camera parameters file found. Parameters successfully loaded!")
    print("-------------------------------------------------------------")
    print()
print("---------Camera parameters-------------")
print("Intrinsics:")
print(cal.K)
print("(Un)distortion coefficients:")
print(cal.dist_coefs)

##################### CAMERA CALIBRATED - PARAMETERS in "cal" object ##################

video_name = "project_video.mp4"
#video_name = "challenge_video.mp4"
#video_name = "harder_challenge_video.mp4"

#writer = imageio.get_writer('project_video.result.mp4', fps = 24)

# start the capture
vid = imageio.get_reader(video_name,  'ffmpeg')
# get the video metadata (interested in frame size and number of frames)
metadata = vid.get_meta_data()
rows = metadata['size'][1]
cols = metadata['size'][0]
print("Frames (width x height) : ", "(", cols, " x ", rows, ")")
num_frames = metadata['nframes']
print("Total numnber of frames : ", num_frames)
# Done! Now processing file
thickness = 2
color = (255, 0, 0)


############################## Rectification Homography ###############################
#left trapezium side
x1 = (int)(cols/4) - 30
y1 = rows-10
x2 = (int)(cols/4) + 247
y2 = rows-(int)(rows/3)
# Right trapezium side
x3 = (int)(cols/4) - 30 + 840
y3 = rows-10
x4 = (int)(cols/4) + 250 + 160
y4 = rows-(int)(rows/3)

# Destination points (a rectangle)
dx1 = x1
dy1 = y1
dx2 = x1
dy2 = 150

dx3 = x3-200
dy3 = y3
dx4 = x3-200
dy4 = 150
# the src and destination points in arrays for getPerspectiveTransform()
src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)
dst = np.array([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]], np.float32)

# Get the homography that maps src to dst (hence removes perspective distortion from the image)
H = cv2.getPerspectiveTransform(src, dst)
print("Homography : ", H)
############################################################################
# MAYBE plt.figure() wont hang with this... But it will ... Pleeeze fix this!
matplotlib.interactive(True)
#f = plt.figure()
# create the frame handler
frame_handler = FrameHandler(H, 
                             rows, 
                             cols, 
                             h_thresh = (230,  255),
                             s_thresh = (100, 255), # (100, 255) initially
                             dir_thresh = (0 , 45),
                             mag_thresh = (100, 140),
                             xgrad_thresh = (40, 150),
                             sobel_kernel = 17
                             )
lane_tracker = LaneTracker(H, rows, cols, nwindows = 9)
#cv2.namedWindow("h-binary")
#cv2.namedWindow("original frame")
#cv2.namedWindow("undistorted frame")
cv2.namedWindow("s-binary")
cv2.namedWindow("xgrad-binary")
cv2.namedWindow("final binary")
cv2.namedWindow("warped")
cv2.namedWindow("Tracked Lines")

# necessary for namedwindow not to get stuck:
# Seriously, what's the matter with threading and image windows in python???
cv2.startWindowThread() 

#for i in range(326, num_frames):
for i in range(num_frames):
    print("Retrieving frame#: ", i)    
    img_p = vid.get_data(i)
    
    img = cal.undistort(img_p)    
    #cv2.imshow("original frame", cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB))
    #cv2.imshow("undistorted frame", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #cv2.line(img1, (x1, y1), (x2, y2), color, thickness)
    #cv2.line(img1, (x3, y3), (x4, y4), color, thickness)
    #cv2.line(img1, (x2, y2), (x4, y4), color, thickness)
        
    #gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    #warped = cv2.warpPerspective(img1,H,(img.shape[1], img.shape[0]))
    #cv2.imshow("frame", warped)
    
    
    frame_handler.process_frame(img)
    #cv2.imshow("h-binary",np.uint8(255*frame_handler.h_binary))
    cv2.imshow("s-binary",np.uint8(255*frame_handler.s_binary))
    #cv2.imshow("x-gradient", np.uint8(255*(frame_handler.sobelx - np.min(frame_handler.sobelx))/(np.max(frame_handler.sobelx) - np.min(frame_handler.sobelx))))    
    cv2.imshow("xgrad-binary", np.uint8(255*frame_handler.gradx_binary))    
    cv2.imshow("final binary", frame_handler.out_img)
    #cv2.imshow("warped", np.uint8(255*frame_handler.img_warped) )
    lane_tracker.track_lanes(frame_handler.img_warped, img)
    # updating the cropping line end-points in the frame-handler
    #frame_handler.crop_ll = lane_tracker.ll
    #frame_handler.crop_ul = lane_tracker.ul
    #frame_handler.crop_lr = lane_tracker.lr
    #frame_handler.crop_ur = lane_tracker.ur
    cv2.imshow("warped", np.uint8(lane_tracker.out_img))
    # yet another conversion for display purposes 
    # (BGR2RGB - recall image was NOT opened with OpenCV becausde I had problems with ffmped 
    # which took a long time to resolve, so I decided to leave the VideoCapture aside)     
    cv2.imshow("Tracked Lines", cv2.cvtColor(lane_tracker.final_image, cv2.COLOR_BGR2RGB))
    #writer.append_data(lane_tracker.final_image)
    #plt.imshow(frame_handler.s_binary)
    key = cv2.waitKey(-1)
    #key = input()    
    if (key == 27):
        break
vid.close()
stop = False
while not stop:
    stop = cv2.waitKey(5) == 27
    #stop = input() == 27
#writer.close()
cv2.destroyAllWindows()
    

