
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


<br>

| Project                  |  Challenge                | 
:-------------------------:|:-------------------------:|
![alt text][image17]       | ![alt text][image18]      | 
| [Youtube](https://youtu.be/OfX10Osa058) / [File](./test_videos_output/project_video.mp4) | [Youtube](https://youtu.be/_vk5zFe4qaw) / [File](./test_videos_output/challenge_video.mp4)

<br>


[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/road_transform.png "Road Transformed"
[image13]: ./examples/color_binary.png "Color Binary"
[image16]: ./examples/grad_binary.png "Gradient Binary"
[image3]: ./examples/thresholded_binary.png "Thresholded Binary"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image7]: ./examples/histogram.png "Histogram"
[image5]: ./examples/color_fit_lines.png "Fit Visual - Blind Search"
[image8]: ./examples/color_fit_lines2.png "Fit Visual"
[image6]: ./examples/output.png "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"
[image9]: ./examples/curvature_formula.jpg "Radius of Curvature Formula"
[image10]: ./examples/curvature_formula2.png "Radius of Curvature Formula"
[image11]: ./examples/curvature_formula3.png "Radius of Curvature Formula - Derivatives"
[image12]: ./examples/curvature_formula4.png "Radius of Curvature Formula"
[image17]: ./examples/project_video.gif "Project"
[image18]: ./examples/challenge_video.gif "Challenge"
[image19]: ./examples/harder_challenge_video.gif "Harder Challenge"

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

I used color thresholds to generate a binary image (thresholding steps at cell 57 in `find_lane.ipynb`). 

Assumed most lane lines are yellow or white. I used OpenCV's function `cv2.cvtColor(img, cv2.COLOR_RGB2HSV)` to convert the image into HSV color space. HSV color space can identify colors accurately, because it separates color from brightness. I used following ranges used to filter out yellow and white lane lines:

| Color         | Min.           | Max.          | 
|:-------------:|:--------------:|:-------------:| 
| Yellow        |  0, 90, 200    |  50 255, 255  | 
| White         |  0,  0, 210    | 179, 70, 255  |
| Yellow 2      |  -,  -, 150    | -             |
| White 2       |  -,  -, 210    | -             |
| Green         |  30,60, 100    | 120,255, 255  |

I have added three additional color thresholds to the binary, Yellow 2, White 2, and Green to enhance the detection in `harder_challenge_video.mp4`. Yellow 2 uses the B channel of LAB color space, White 2 uses the V channel in HSV channel. Green was used to wipe out road side trees. Yellow and white binaries were combined using OR operator, then AND with Green binary.

Here's an example of the color thresholded binary:

![alt text][image13]

I had successfully identified lane lines from `project_vidoe.mp4` and `challenge_vidoe.mp4` with this color binary.

In `harder_challenge_video.mp4`, the color binary failed to identify lane lines because of the shadows and sunlights. To better identify the yellow lane with shadows, I added a graident binary over both x and y directions using the V channel from LUV color space.

Here's an example of the gradient binary:

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

The pixels are either 0 or 1 in the thresholded binary image, so the two most prominent peaks in this histogram can be good indicators of teh x-position of the base of the lane lines. I used it as a starting point for where to search for the lines. Then, from this point, I placed sliding windows around the line centers, to find and follow the lines up to the top of the frame.

Then I fit my lane lines with a 2nd order polynomial like this:

![alt text][image5]

Once I know where the line is, I don't need to do the blind search again for the next frame. Instead, I can just search in a margin around the previous line position like this:

![alt text][image8]

The green shaded area are where I searched for the lines this time, without sliding windows. This is equivalent to use customized region of interst for each frame of video, and would help to track the lane through sharp curves and tricky conditions. 

I did this in cell 30 to 33 in `find_lane.ipynb`.

#### 5. Radius of curvature and position of the vehicle

I then used the second order polynomial curve f(y) = Ay^2 + By + C to calculate the radius of curvature. 

![alt text][image9]

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

I did this in cell 34 to 36 in my code in `find_lane.ipynb`

#### 6. Result output

I implemented this step in cell 39 in my code in `find_lane.ipynb` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Tracking frames
In order to apply the pipeline to video streams, I have to keep track of things like where the previous detections of lane lines were and what the curvature was, so I could properly treat the new detections. The class `Line()` in cell 43 of `find_lane.ipynb` was defined to track all these related parameters.

#### Final video outputs

The final pipeline for video is in cell 45 in `find_lane.ipynb`. Here are the links to my video outputs:

* [project_video.mp4](./test_videos_output/project_video.mp4)
* [challenge_video.mp4](./test_videos_output/challenge_video.mp4)
* [harder_challenge_video.mp4](./test_videos_output/harder_challenge_video.mp4)

---

### Discussion

#### Problems / issues 

##### 1. Color of road surface and shadows in `project_video.mp4`

My first attempt was using a gradient thresholded binary without color binary. It worked well on the test images, but failed in videos when the colors of road surface change, and shadows of trees and overhead bridge.I used color binary instead of gradient to solve it.

##### 2. Roadside trees

In `harder_challenge_video.mp4`, the trees on roadside were not excluded in the thresholded binary, and it caused the histogram step failed to identify the line's position correctly. I adjusted the color threshold range to solve this. 

##### 3. Reflections on windshield

In `harder_challenge_video.mp4`, there were reflections on the car's windshield causing the color binary failed to identify the yellow lane line. I adjusted the color threshold range to solve this.

##### 4. Left and right lane lines across each other

To solve this, I added sanity checks to only include the lane lines that are parallel, with similar curvature, and reasonable distance apart from each other.

##### 5. Line detections jump from frame to frame

Whenever a high-confident measurement passed the sanity checkes, I append it to a list of recent measurements and take average over n past measurements to obtain a smoother frame to frame transition.

#### Future Works

Using the traditional computer vision techniques to find lane lines was not an easy task. It was time consuming and required a lot of work on tuning different things manually. I might want to use deep learning instead, train a CNN model to find the lane lines. 


