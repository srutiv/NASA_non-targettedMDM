READ ME
Last updated 8/9/19 by Sruti Vutukury
-refer to documentation for detailed discussion on algorithms

-spec-file.txt: create an environment with these packages to make sure all the algorithms and classes are available

-cv_fun: folder of scripts from openCV sample folder

-buggy folder: contains scripts that I was playing around with; very buggy, but kept just incase

-sv_checker_calib.py: checkerboard calibration script; for 1 camera; returns camera matrix with instrinsic parameters; includes distortion and key; need at least 7 checkerboard calibration images

-sv_blob_finder.py: simple blob finder

-sv_blob_param_iter.py: iterates over thresholds of blob finder until a certain number of blobs are found; good when you have a known number of targets or a calibration wand; can be added as error handling code when you are not able to find enough blobs

-sv_harris_corner.py: detects and matches harris corners

-sv_hough_detect.py: hough circle finder; does blurring and increasing threshold to detect and track circles

-sv_dense_optical_flow.py: as its name implies; not great for our application as feature detection and tracking is done over ALL pixels --> too much noise

-sv_sparse_optical_flow.py: as its name implies; better for our application as feature detection and tracking is done only when certain thresholds are met; powerful enough for our application

-sv_detect_more.py: playing around with blurring, binarizing, and downsampling to find more corners using Shi-Tomasi

-sv_sift.py: SIFT/SURF feature detector and  FLANN based matcher; between two images

-sv_motemplate.py: motion template technique; Motion templates is alternative technique for detecting motion and computing its direction; finding a small part of an image that matches a template image; template matcher used to make comparisons

-sv_get_disps.py: has 2D and 3D displacement calculator; 3D displacement calc has camera matrices hard-coded in

-sv_getROIs: for user-selected ROIs; can be used to define masks in feature detection algorithms


Framework 1:

-sv_charmander.py: sparse optical flow; input either goodFeaturestoTrack (Shi-Tomasi corners) or blobs (from blob_finder--not that great) to do tracking with; loads images -->  corner/blob detector --> optical flow tracking -->pixel displacement; 1 cam; no user-selected ROI

-sv_charmeleon.py: Shi-Tomasi corners + sparse optical flow; loads images --> corner detector --> optical flow tracking --> pixel displacement; no user-selected ROIs; homography matrix, 1 camera intrinsic matrix, calculates 2D physical displacements; if same # of corners not found, iterates over parameters for optical flow

-sv_charizard.py: ran in the recorded demo video; Shi-Tomasi corners + sparse optical flow; loads images --> corner detector --> optical flow --> pixel displacement; user-selected ROIs (from get_ROIs); uses 2+ camera parameters matrix (hardcoded in sv_get_disps), calculates 3D physical displacements (sv_get_disps); if same # of corners not found, iterates over parameters for optical flow


Framework 2:
-sv_bulbasoar.py: loads images, SIFT/SURF/FLANN feature detector and tracker, homography matrix, camera matrix, displacements not right --> main problem: the same features are not found from image pair to image pair

-sv_ivysaur.py: loads images, SIFT/SURF feature detector + FLANN matcher, homography matrix, uses 1 camera intrinsic parameter matrix, does 2D displacement calculation; does not have user-selected ROIs

-sv_venusaur.py: unfortunately did not have time to debug; SIFT/SURF/FLANN feature detector and matcher; outputs are keypoint pixel coordinates are inputed in sv_get_disps.py to get 3D displacement calculations; uses 2+ camera intrinsic and extrinsic parameter matrices; has user-selected ROIs with sv_getROIs


