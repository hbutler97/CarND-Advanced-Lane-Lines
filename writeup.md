[//]: # (Image References)

[image1]: ./output_images/header.png "Header"
[image2]: ./output_images/calibration.png "Calibration"
[image3]: ./output_images/pipeline.png "Pipeline"
[image4]: ./output_images/sobel.png "Sobel"
[image5]: ./output_images/mag.png "Mag"
[image6]: ./output_images/dir.png "Dir"
[image7]: ./output_images/color.png "Color"
[image8]: ./output_images/comp.png "Comp"
[image9]: ./output_images/warp.png "Warp"
[image10]: ./output_images/first_stage.png "First Stage"
[image11]: ./output_images/original.png "Original"
[image12]: ./output_images/fit.png "Fit"
[image13]: ./output_images/bin_hist.png "binary hist"
[image14]: ./output_images/hist.png "Hist"
[image15]: ./output_images/final.png "Final"

![alt text][image1]
## **Overview**

The purpose of this project is to use advanced computer vision techniques to detect lane lines.  Techniques include camera calibration, multiple types of filtering, and prospective transforms. A software pipeline is constructed to identify lane boundaries from a front-facing camera.  Those boundaries are then projected back on to the original image.  

Link to [project code](https://github.com/hbutler97/CarND-Advanced-Lane-Lines/blob/master/find_lanes.ipynb)

Link to [Result YouTube Video](https://youtu.be/GWFBVorvk5I)



## **Camera Calibration**

Prior to use for proper lane detection, the cameras must be calibrated.  Cameras produce images that can be distorted radially and/or tangentially, which if not corrected can impact proper lane detection.  The Camera calibration matrix and distortion coefficients are required to correct for distortion.

In function [calibrate_camera()](https://github.com/hbutler97/CarND-Advanced-Lane-Lines/blob/master/find_lanes.ipynb), multiple chess board images are used to calibrate the camera. The OpenCV findChessboardCorners() function is use to find the corners of the chessboard along with object points.  Those values are then passed to the OpenCV function calibrateCamera() which produces the camera calibration matrix and distortion coefficients.  Attaining the camera matrix and distortion coefficients is a one time process and these values will be used by the pipeline to undistort camera images.



## **Software Pipeline**

The software pipeline used to detect the lane lines.  High Level functions preformed by the pipeline are as follows:
* Undistorts image
* Filters the image
* Combines filter outputs
* Locates the lane lines and makes measurements
* Projects the lane line on the image

Figure below is a block diagram of the software pipeline.

![alt text][image3]

### **Imagine Distortion Correction**

Once the camera matrix and distortion coefficients are attained from the camera calibration, in the undistort_image() function, a call is made to OpenCV undistort() function was called to produce the undistored image as shown below

![alt text][image2]

### **Imagine Filtering**

As shown in the software pipeline diagram, once we have an undistorted image, it is passed to different image filters.  The image filters purpose is to extract features of the image to isolate lane lines.  Different filters are used to assist with managing detecting the lanes under different conditions.

#### **X/Y Sobel Filter**

The Sobel Filter stage is going to take the derivative in the specified direction of the image and if the derivative is above a specified threshold, then that pixel remains in the output. The abs_sobel_thresh() function, calls the OpenCV Sobel() to produce the output shown below.

    	  original image, 	   	    X Sobel    	      			     Y Sobel

![alt text][image4]

#### **Gradient Magnitude Filter**

Once the Sobel filter output is generated, then the absolute magnitude(intensity) of the X and Y outputs are calculated and filtered again in the mag_thresh() function.  Below is example output from this function.

![alt text][image5]

#### **Direction Filter**

Lane lines have the attribute that they are in most cases vertical in the images... There for we can also filter for angle to further isolate the lane lines.  This is done in the dir_threshold() function.

![alt text][image6]

#### **Color Filter**

Lastly, the color is used to help locate the lane lines.  Lane lines can be different colors and also the detection of the colors can be affected by overall lighting and contrast with the road itself. In functions color_rgb_threshold() and color_hls_threshold(), the r channel of the rgb image and the s channel of the hls image are used to detect the lane lines as they seem to give good results

    	  original image, 	   	    R Channel    	      			S Channel
	  
![alt text][image7]

### **Composite Image**

After filtering the images, a combined composite of the filtered image is produced (by anding/oring and the filter output) in the function a called first_stage_pipeline() and the final output is shown below.

![alt text][image8]

### **Prospective Transform**

In order to do lane fitting and measurements, the image prospected was changed to view the image from top down prospective.  To do the prospective transform, the OpenCV function getPerspectiveTransform() will give us the transformation matrix if provided the source matrix and destination matrix.  Four points were identified on the lane lines in the original image.  The shape formed by the points was a trapezoid.  The destination matrix was computed by maintaining the width of the base of the trapeziod, and drawing a rectangle with the same area of the trapeziod.  The points were extracted from the newly formed rectangle to provide the destination matrix to the getPerspectiveTransform() function.  In the prospective_transform() function, the image is warped using the OpenCV warpPerspective() function.  Example of the prospective transformation is shown below

![alt text][image9]


### **Lane Line Identification**

The input to the Find Lane Lines Fit block is a composite, prospective transformed binary image. As shown below.

![alt text][image10]

To locate the lines a histogram is created.  The peaks of the histogram represent the X location of the lane lines and was fitted to a polynomial. A sliding window is place around the x location of the peaks and the window is relocated to the center of the peaks as you move upwards in the image.  This is done in the find_line_fits() function. 

![alt text][image11]  ![alt text][image12]
![alt text][image13]  ![alt text][image14]

### **Distance from Center and Radius of Curvature**

Given the X location of the lane lines and known the dimensions of the image, we can calculate the offset of the car in pixels by subtracting the midpoint of the lane line locations with the midpoint of the image.  That value was translated into meters by multiplying the result by 3.7/700.

To calculate the radius of curvature, a second order polynomial fit is done starting at the base of the image and that is used to calculate the radius of curvature using the following formula.  

R​curve​​=​​((1+(2Ay+B)​^2​​)^(​3/2)​​​​)/|2A|

These calculations are preformed in find_distance_from_center() and radius_of_curve().

### **Project the Lane**

The detected lane lines are projected back on to the original image as shown below

![alt text][image15]

### **Video Results**

Link to [Result YouTube Video](https://youtu.be/GWFBVorvk5I) 

### **Discussion**

While doing this project there were several challenges that came up.
1. Camera Calibration
* Several of the camera calibration images wouldn't correctly pass through the findChessboardCorners() function.  The images were 9x6 and images that didn't have all corners clearly visible would fail this function.  Thus calibration of the camera probably wasn't as good as it should have been.  Images could have been replaced. 
2. Prospective Transform
* Performing the prospective transform had a big impact on the detect of the lanes.  The initial selection of the src matrix yielded poor results if not chosen carefully.  Given this, angle of attach(hills) changes to the car would probably cause this pipeline to have problems.  One way to help with this would be to dynamically calculate the src and dest matrix based on the orientation of the car.
3. Remembering the location of the line
* Rather than finding the lane all over again, the pipeline could skip some of the calcuations by looking back at previous lane line locations and reducing the number of operations. 


