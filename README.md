# Advanced Lane Tracking

This lane tracking algorithm is a robust line detector in video sequences taken from a camera mounted at the center-front of a vehicle.

## Code Organization
The file `lane_detector.py` runs the pipeline. It invokes the camera calibrator in file `calibrator.py`either for initial calibration (if the intrinsics and distortion coefficients are not saved), or simply for loading the existing parameters from a file. The is responsible for removing distortion from the input frames. File `lane_tracking.py`fits the quadratics in the right and left lines in the bird's-eye-view image, while the pre-processing is done in `frame_handler.py`.

Note that the calibrator will look for calibration images in the directory `camera_cal`, which exists already in the repository. The calibration images are zipped in it.

## Brief Description
The first processing step is to isolate edges and pixels that belong to the road lanes. With a bit of cropping, we obtain a binary image which can ebe used to fit the right and left lane.

The idea here is to remove perspective distortion from the ground plane in the binary image and fit two parallel lines with quadratics in the resulting _bird's eye view_. Then, a dense set of samples from the two quadratics is transformed back to the original image. 

## More Details
Detailed description of the overall method can be found in the [writeup_report.pdf](https://github.com/terzakig/SelfDrivingCar1-LaneTracking/blob/master/writeup_report.pdf) included in this repository.

## Demo Video
A video with the lane-tracking in action can be found [here](https://youtu.be/zzDMb7RPT0o). 
