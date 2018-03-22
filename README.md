
## Advanced Lane Finding Project

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/road_transform.png "Road Transformed"
[image3]: ./examples/thresholded_binary.png "Thresholded Binary"
[image13]: ./examples/color_binary.png "Color Binary"
[image14]: ./examples/grad_binary_y.png "Gradient Binary - Y"
[image15]: ./examples/grad_binary_x.png "Gradient Binary - X"
[image16]: ./examples/grad_binary.png "Gradient Binary"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.png "Fit Visual - Blind Search"
[image8]: ./examples/color_fit_lines2.png "Fit Visual"
[image6]: ./examples/output.png "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"
[image7]: ./examples/histogram.png "Histogram"
[image9]: ./examples/curvature_formula.jpg "Radius of Curvature Formula"
[image10]: ./examples/curvature_formula2.png "Radius of Curvature Formula"
[image11]: ./examples/curvature_formula3.png "Radius of Curvature Formula - Derivatives"
[image12]: ./examples/curvature_formula4.png "Radius of Curvature Formula"


---

### Camera Calibration

The code for this step is contained in the code cell 3 to 8 of the IPython notebook located in `find_lane.ipynb`.

I start by preparing `object points`, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Undistort image

I used the same distortion coefficient obtained from the camera calibration step to undistort the image:
![alt text][image2]

#### 2. Thresholded binary

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cells 12 through 25 in `lane_finding.py`).  

Most lane lines are yellow or white. I used OpenCV's function `cv2.cvtColor(img, cv2.COLOR_RGB2HSV)` to convert the image into HSV color space. In HSV color space, the color is separated from brightness, and it can identify colors more accurately. The ranges used to filter out yellow and white colors are listed below:

| Color         | Min.          | Max.          | 
|:-------------:|:-------------:|:-------------:| 
| Yellow        | 0, 60, 100    | 50, 255, 255  | 
| White         | 0,  0, 210    | 179, 70, 255  |
| Yellow 2      | 18,31, 100    | 50, 130, 255  |
| Green         | 30,60, 100    | 120,255, 255  |

Yellow 2 is the threshold range to detect yellow with windshield reflections, and Green is the range used to wipe out road side trees.

With this color thresholded binary, combining with the other steps as described below, I had successfully identified lane lines from `project_vidoe.mp4`.

Here's an example of the color thresholded binary:

![alt text][image13]

It was naive to assume all lane lines are yellow and white. In the real world, they could be in any color. For example, in the beginning of `challenge_video.mp4`, the right lane line is in dark gray. The color thresholded binary for yellow and white failed to identify the gray color. 

To identify lane lines in other colors, I then used Sobel operator to find the gradient along both x and y direction on V channel of HSV color space. Taking gradient in x direction emphasizes edges closer to vertical, taking gradient in y directin emphasizes edges closer to horizontal. Considering x-gradient does a cleaner job picking up the lane lines, but y-gradient picks up the lane lines as well - I took 2 graidents over both x and y directions, then applied different kernel sizes for the x and y Sobel opertions. For the first gradient, a greater kernel size was assigned to x Sobel, and for the second gradient, a greater kernel size was assigned to y Sobel. Then I kept their intersection as the final result.

Here's an example of a gradient binary with greater kernel size for y Sobel:

![alt text][image14]

Here's an example of a gradient binary with greater kernel size for x Sobel:

![alt text][image15]

Here's an example of the combined gradient binary:

![alt text][image16]

Here's an example of my output for this step. 

![alt text][image3]

#### 3. Perspective transform

The code for my perspective transform includes a function called `warper()`, which appears in cell 27 in the IPython notebook `find_lane.ipynb`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
         [[(img_size[0] / 2) - 89, min_y],
         [(img_size[0] * 297/1280), max_y]
         [(img_size[0] * 1001/1280), max_y],
         [(img_size[0] / 2 + 90), min_y]])
dst = np.float32(
         [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), max_y],
         [(img_size[0] * 3 / 4), max_y],
         [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 551, 480      | 320, 0        | 
| 297, 648      | 320, 648      |
| 1127, 720     | 960, 648      |
| 1001, 648     | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identify lane-line pixels and fit with a polynomial

To identify pixels belong to the left and right line. I first take a histogram along all the columns in the lower half of the thresholded binary image:

![alt text][image7]

The pixels are either 0 or 1 in the thresholded binary image, so the two most prominent peaks in this histogram can be good indicators of teh x-position of the base of the lane lines. I can use it as a starting point for where to search for the lines. Then, from this point, I placed sliding windows around the line centers, to find and follow the lines up to the top of the frame.

Then I fit my lane lines with a 2nd order polynomial like this:

![alt text][image5]

Once I know where the line is, I don't need to do the blind search again for the next frame. Instead, I can just search in a margin around the previous line position like this:

![alt text][image8]

The green shaded area are where I searched for the lines this time, without sliding windows. This is equivalent to use customized region of interst for each frame of video, and would help to track the lane through sharp curves and tricky conditions. 

#### 5. Radius of curvature and position of the vehicle

I then used the obtained second order polynomial curve f(y) = Ay^2 + By + C to calculate the radius of curvature. 

![alt text][image9]

First draw a circle that closely fits nearby points on a local section of a curve. In a given curve, the radius of curvature in some point is the radius of the circle that "kisses" it. 

The formula for the radius at any point x for the curve f(y) is:

![alt text][image10]

The first and second derivatives are:

![alt text][image11]

The equation for the radius of curvature becomes:

![alt text][image12]

I assumed the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two detected lines. The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane.

The lane in image is about 30 meters long and 3.7 meters wide, I converted the units from pixels to meters as following:

```
ym_per_pix = 30/720
xm_per_pix = 3.7/700
```

I did this in cell 34 and 35 in my code in `find_lane.ipynb`

#### 6. Result output

I implemented this step in cell 38 in my code in `find_lane.ipynb` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Tracking frames
In order to apply the pipeline to video streams, I have to keep track of things like where the previous detections of lane lines were and what the curvature was, so I could properly treat the new detections. The class `Line()` in cell 42 of `find_lane.ipynb` was defined to track all these related parameters.

#### Final video output

Here's a [link to my project_video.mp4 result](./test_videos_output/project_video.mp4)

---

### Discussion

#### Problems / issues 
My first attempt was applying the gradient threshold technique. My initial pipeline consisted of undistorting the image, using Sobel operator to find gradient along x and y directions, performing perspective transform to convert the image to a bird eye view, using histogram and sliding window to fit the polynomial lines, then unwarping the image back to the original perspective. It worked well on the test images, but failed at few points in [project_video.mp4](./test_videos_output/project_video.mp4). The challenges encountered was at the point where the color of road surface changes from light gray to dark gray, or the other way around, and also the shadow of trees.

I then decided to take advantage of the fact that most of the lane lines are in either yellow or white color. HSV color space separates color from brightness, and is able to detect color rather accurate than RGB. Considering the environment's brightness can change during driving, I converted the image to HSV color space and set threshold ranges for yellow and white. With only this color threshold technique, I was able to successfully detect the lane lines in [project_video.mp4](./test_videos_output/project_video.mp4). 

However, the color threshold techinique has its own limits. In [challenge_video.mp4](./test_videos_output/challenge_video.mp4), it failed to detect the lane lines when one of the lines is dark gray, instead of the assumed yellow or white. It also failed at the shadow of an overhead bridge, and at where the lane was splitted in the middle into two colors, the left side is in dark gray, and the right is in light gray.

To detect the the lane lines other than yellow and white, I added a gradient thresholded binary. The missing dark gray lane line was found, but also produced the new problems of unwanted splitting line in the center of lane, and the shadow of the overhead bridge. The splitting line caused the lane detected as only half of the correct width. To overcome this, I added a second gradient thresholded binary, and assigned different kernel sizes in x and y direction. One with larger kernel size in y, and another with larger kernel size in x. I then only kept the intersection of these two gradient thresholded images. In this way, I was able to cancel out the unwanted noises and succesfully detect lane lines in [challenge_video.mp4](./test_videos_output/challenge_video.mp4).

My attempt to detect lane lines in [harder_challenge_video.mp4](./test_videos_output/harder_challenge_video.mp4) failed at several points. First was the trees at the roadside - the color thresholded binary did not wipe out the trees. This caused problem in the histogram step, I couldn't identify the line's position correctly. To solve this, I added an additional threshold range to exclude the trees. Second was that whenever the reflections occur on the car's windshield, the yellow lane lines were not detected. I adjusted the color threshold range to solve this. 

To enhance the pipeline, I further applied sanity checks and smoothing technique. The left and right lane lines occasionally came across each other, and the lane sometimes detected with a wrong width. With sanity checks, I made sure to include only the lane lines that are parallel, with similar curvature, and reasonable distance apart from each other. Another problem was that the line detections jump from frame to frame. Whenever a high-confident measurement passed the sanity checkes, I append it to a list of recent measurements and take average over n past measurements to obtain a smoother frame to frame transition.

The detection for [harder_challenge_video.mp4](./test_videos_output/harder_challenge_video.mp4) should be further improved. The major fail point was at a sharp turn, and the yellow grass at the roadside was mistakenly detected as the right line. For future works, I might want to enhance the threshold step to better identify the white line at the side, and experiment with convolutions for the sliding window step.  


