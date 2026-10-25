## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./docs/undistort_output.png "Undistorted"
[image2]: ./docs/test1_undistorted.png "Road Transformed"
[image3]: ./docs/binary_combo_example.png "Binary Combo"
[image4]: ./docs/warped_straight_lines.png "Warp"
[image5]: ./docs/color_fit_lines.png "Fit Visual"
[image6]: ./docs/example_output.png "Output"
[video1]: ./docs/hough_video_out.mp4 "Video"
[video2]: ./docs/project_video01_out.mp4 "Video"
[video3]: ./docs/project_video02_out.mp4 "Video"
[video4]: ./docs/project_video03_out.mp4 "Video"
[video5]: ./docs/challenge01_video_out.mp4 "Video"
---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

A camera lens bends straight lines, especially near the edges of the image. To correct this, I need to know how the lens distorts the image. I know how a chessboard should look (a perfect flat grid), and I can detect how the camera actually sees it, so comparing the two gives the camera parameters.

* The board has 9 x 6 inner corners. I prepared the "object points", the ideal 3D coordinates of the corners '(0,0,0), (1,0,0), ..., (8,5,0)'. The board is flat, so 'z = 0', and the same array is used for every image because the real board does not change.
* For every calibration image I converted it to grayscale and called 'cv2.findChessboardCorners'. When the corners were found, I refined them with 'cv2.cornerSubPix' (sub-pixel accuracy) and stored them as "image points".
* Of the 20 calibration images, 17 were used. In 'calibration1.jpg', 'calibration4.jpg' and 'calibration5.jpg' the whole board is not visible, so the corners cannot be found and these images are skipped.
* 'cv2.calibrateCamera' then returns the camera matrix 'mtx' (focal lengths and optical center) and the distortion coefficients 'dist'. Several images with the board in different positions are needed so that the problem is well constrained.
* The result is saved to 'camera_cal/calib.npz', so the other scripts only load it and never repeat the calibration.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![Undistorted][image1]

![Road Transformed][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code is in 'thresholded_binary.py'. I combine two sources of information, because each one fails where the other works.

**Gradient ('sobel_x_mask').** Lane lines are almost vertical in the camera image, so I use the Sobel operator along the x axis ('dx=1, dy=0'), which responds to vertical edges. Horizontal edges (horizon, car hood, guard rails) are ignored. Before the filter I apply a 5x5 Gaussian blur, which suppresses the fine texture of the asphalt. The result is normalized to the range 0-255 (divided by its maximum) instead of simply clipped with 'convertScaleAbs'; with a 3x3 kernel the gradients can be large, and normalization gives the threshold (40) the same meaning on every image.

**Color ('color_mask').** The image is converted to the HSV color space, where the color (H) is separated from brightness (V). With 'cv2.inRange' I select yellow lines ('H 15-35, S 80-255, V 60-255') and white lines ('S 0-30, V 200-255').

**Combination ('threshold').** The two masks are combined with OR. The reason for OR instead of AND is that either source can be enough: a yellow line on bright concrete has a weak gradient but the right color, and a worn white line has a clear edge but a faint color. Finally I apply a morphological opening with a 3x3 kernel (erosion followed by dilation), which removes isolated white specks from the asphalt and keeps the (larger) lines.

![Binary Combo][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is in 'perspective.py'. The function 'warp' uses 'cv2.getPerspectiveTransform' and 'cv2.warpPerspective'.
I chose four source points forming a trapezoid around the lane and mapped them to a rectangle. To pick the coordinates I used the helper 'pick_points.py', which prints the pixel position of every mouse click on the undistorted image. The points were chosen on 'straight_lines1.jpg','straight_lines2.jpg', where the road is straight: the lane lines are parallel in reality, so after the transform they must be vertical and parallel. This is the check that the points are correct. On a curved road this check would not be possible.

| Source | Destination |
|:---:|:---:|
| 598, 460 | 320, 0 |
| 706, 460 | 960, 0 |
| 1071, 718 | 960, 720 |
| 247, 719 | 320, 720 |

I verified the transform by drawing the trapezoid on the original and the rectangle on the warped image (below): the lane lines are vertical and parallel. The inverse matrix 'M_INV' is computed too, and it is used later to project the detected lane back onto the original image.

![Wrap][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is in 'lane_detection.py', functions 'find_lane_pixels' and 'fit_polynomial'.

1. **Starting points.** On the warped binary image I sum the white pixels in each column, using only the *bottom half* of the image. The bottom is the part closest to the car, where the lines are the clearest and the most vertical. The histogram has two peaks, the left one (left half of the image) and the right one (right half), and they are the starting x positions of the two lines.
2. **Sliding windows.** The image is split into 9 horizontal strips. Starting at the bottom, in each strip I take the white pixels inside a window of +-100 pixels around the current center, for both lines. If the window contains more than 50 pixels, the center of the next window (the one above) is moved to the mean x position of those pixels. In this way the windows follow the line, also when it curves.
3. **Polynomial fit.** All the collected pixels are fitted with 'np.polyfit' using a second-order polynomial 'x = A*y^2 + B*y + C'. I fit x as a function of y (and not the other way around) because the lines are close to vertical, so for each y there is exactly one x, while y = f(x) would be undefined or extremely steep for a vertical line.

The picture shows the pixels assigned to the left line (red), to the right line (blue) and the fitted polynomials with their coefficients (yellow).

![Fit Visual][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code is in 'lane_detection.py', functions 'curvature_radius' and 'vehicle_offset'.

**Radius of curvature.** For a curve 'x = A*y^2 + B*y + C' the radius is evaluated at the bottom of the image (the point closest to the car). The polynomial fitted in pixels cannot be used directly, because a radius in pixels has no physical meaning. So I fit the polynomials again, this time with the pixel positions converted to meters:

* 'XM_PER_PIX = 3.7 / (960 - 320)': a lane is about 3.7 m wide and, in the warped image, it spans from x = 320 to x = 960.
* 'YM_PER_PIX = 30 / 720': I assume that the warped image shows about 30 m of road. This is an assumption (it depends on how far up the source trapezoid goes), so the radius is an estimate.

The final radius is the average of the left and right line. On the straight-road images it is very large (kilometers), and on the curved ones it is in the order of a few hundred meters.

**Vehicle position.** I assume that the camera is mounted in the center of the car, so the center of the car is the center of the image ('width / 2'). The center of the lane is the middle between the x positions of the left and right polynomial at the bottom of the image. The difference in pixels, multiplied by 'XM_PER_PIX', is the offset in meters; a negative value means that the car is to the left of the center. This only works if the source trapezoid is symmetric around the image center; otherwise a systematic shift appears.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![Output Image][image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I implemented two approaches.

**Approach 1: Hough transform ('hough_lanes.py').** Grayscale, Gaussian blur, Canny edges, a region-of-interest mask and 'cv2.HoughLinesP'. The segments are split into left and right by the sign of their slope, nearly horizontal ones are discarded, and one straight line 'x = a*y + b' is fitted per side. The parameters are smoothed over frames ('SMOOTHING = 0.8'). It works on straight roads, but a straight line cannot follow a curve, and in shadows it picks up barrier edges instead of lane lines.

**Approach 2: polynomial pipeline ('lane_video.py').** The same steps as for single images (threshold, perspective transform, polynomial fit), plus a tracker ('LaneTracker') for robustness:

* **Search around the previous polynomial** ('search_around_poly'): only pixels within 8% of the image width are used, so shadows outside the band are ignored. If too few pixels are found, it falls back to sliding windows.
* **Plausibility check** ('is_plausible'): a fit is accepted only if the lane width is realistic and the lines are roughly parallel; otherwise the frame is skipped.
* **Smoothing and restart:** the result is the average of the last 8 accepted fits, and the history is cleared after 15 rejected frames in a row.
* **Resolution independence:** the trapezoid is given as fractions of the frame size, chosen per resolution ('SRC_FRAC_BY_SIZE'), so the same script handles 960x540 and 1280x720 videos.
* **Drawing:** a band around each polynomial is warped back to the camera view, and the real yellow/white paint pixels inside it ('color_mask') are painted red. The red line therefore follows the actual marking, solid or dashed. Curvature radius and vehicle offset are written on each frame.

Output videos:

* Hough approach: ![Video][video1]
* Polynomial approach: 
 ![Video][video2]
 ![Video][video3]
 ![Video][video4]
 ![Video][video5]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


**Problems I faced**

* *Asphalt noise.* The first binary images had many white dots from the asphalt texture. A Gaussian blur before Sobel, a stricter white mask and a morphological opening removed most of them. A vertical closing to join dashed lines made things worse, because it ran before the perspective transform, where lines are still tilted.
* *Shadows and changing light.* A line in shade has lower brightness (V) and fell out of the color range, which interrupted the red line. Lowering the V limit for yellow helped, since saturation stays high in shade and is more reliable than brightness.
* *Different cameras.* Calibration and the perspective source points are valid only for the camera they were made for. For the test videos I skipped calibration and defined the trapezoid as fractions of the frame size, per resolution.

**Where the pipeline will likely fail**

* Strong shadows (bridges), glare, rain, night driving, and worn or missing markings. The thresholds are tuned for the test material.
* Sharp curves where a line leaves the view or a second-order polynomial is not enough, and other vehicles in the lane that produce edges or white/yellow pixels.
* Radius and vehicle position are estimates: they assume a 3.7 m lane, 30 m of visible road and a camera at the center of the car, and are less accurate for uncalibrated videos.
* The Hough approach cannot follow curves at all.

**How to make it more robust**

* Adaptive thresholds, CLAHE and the HLS saturation channel for varying light.
* Choosing the source points automatically (for example from the vanishing point).
* Fitting both lines together with a parallel-lines and constant-width constraint, so a weak line can be inferred from the strong one.
* Computing the new camera matrix once instead of per frame if calibration is used on video.
