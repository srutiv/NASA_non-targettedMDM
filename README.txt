READ ME
Last updated 7/18/19 by Sruti Vutukury
-refer to report/poster presentation for detailed discussion on algorithms
-hopefully my code is commented enough

1. cv_fun: folder of scripts from openCV sample folder

2. spec-file.txt: create an environment with these packages to make sure all the algorithms and classes are available

3. “buggy” folder: contains scripts that I was playing around with; very buggy, but kept just incase

4. sv_checker_calib.py: checkerboard calibration script; returns camera matrix with extrinsic and instrinsic parameters; need at least 7 checkerboard calibration images

5. sv_harris_corner.py: detects and matches harris corners

6. sv_hough_detect.py: hough circle finder; does blurring and increasing threshold to detect and track circles

7. sv_blob_finder.py: simple blob finder

8. sv_blob_param_iter.py: iterates over thresholds of blob finder until a certain number of blobs are found; good when you have a known number of targets or a calibration wand

9. sv_dense_optical_flow.py: as its name implies

10. sv_sparse_optical_flow.py: as its name implies

11. sv_sift.py: SIFT/SURF feature detector and  FLANN based matcher;  key point = coord of detected feature, descriptor = array of numbers that describe the feature; descriptors are similar —> feature is similar 

12. sv_motemplate.py: motion template technique; Motion templates is alternative technique for detecting motion and computing its direction; finding a small part of an image that matches a template image; template matcher used to make comparisons

13. sv_bulbasoar.py: loads images, SIFT/SURF/FLANN feature detector and tracker, homography matrix, camera matrix, displacements not right --> main problem: the same features are not found from image pair to image pair

14. sv_ivysaur.py: loads images, SIFT/SURF/FLANN-or BF feature detector and tracker, homography matrix, camera matrix, hopefully the right displacements

15. sv_charmander.py: sparse optical flow with FlowPyrLK (corner matcher); input either goodFeaturestoTrack (corners) or blobs (not that great) from blob_finder to do tracking with; loads images, corner/blob detector, optical flow, pixel displacement

16. sv_charmeleon.py: sparse optical flow with FlowPyrLK (corner matcher); loads images, corner detector, optical flow, homography matrix, camera matrix, physical displacements; if same # of corners not found —> iterates over parameters for optical flow


