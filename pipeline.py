import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
'''
1. Camera Calibration
2. Distortion Correction
3. Color and Gradient Thresholds
4. Prospective Transform
'''





'''
Camera Calibration
Comput the Calibration Matrix
'''
def bgr_to_rgb(image):
    b,g,r = cv2.split(image)
    return cv2.merge([r,g,b])

def compare_dot_images(image1, image2, src, dst, cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap=cmap)
    ax1.plot(src[0][0], src[0][1], '.')
    ax1.plot(src[1][0], src[1][1], '.')
    ax1.plot(src[2][0], src[2][1], '.')
    ax1.plot(src[3][0], src[3][1], '.')

    ax1.plot(dst[0][0], dst[0][1], 'x')
    ax1.plot(dst[1][0], dst[1][1], 'x')
    ax1.plot(dst[2][0], dst[2][1], 'x')
    ax1.plot(dst[3][0], dst[3][1], 'x')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(image2, cmap=cmap)
    ax2.plot(src[0][0], src[0][1], '.')
    ax2.plot(src[1][0], src[1][1], '.')
    ax2.plot(src[2][0], src[2][1], '.')
    ax2.plot(src[3][0], src[3][1], '.')

    ax2.plot(dst[0][0], dst[0][1], 'x')
    ax2.plot(dst[1][0], dst[1][1], 'x')
    ax2.plot(dst[2][0], dst[2][1], 'x')
    ax2.plot(dst[3][0], dst[3][1], 'x')

    ax2.set_title('Processed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def compare_lines_images(image1, image2, src, dst, cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap=cmap)
    ax1.axvline(x=dst[0][0])
    ax1.axvline(x=dst[2][0])
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(image2, cmap=cmap)
    ax2.axvline(x=dst[0][0])
    ax2.axvline(x=dst[2][0])
    ax2.set_title('Processed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def compare_images(image1, image2, cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap=cmap)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(image2, cmap=cmap)
    ax2.set_title('Processed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
def calibrate_camera(nx, ny):
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane

    # Prepare object points like (0,0,0, (1,0,0), (2,0,0)....,(7,4,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    images = []
    image_files = glob.glob('./camera_cal/calibration*.jpg')

    for image_file in image_files:
        images.append(bgr_to_rgb(cv2.imread(image_file)))
        
    gray_image = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    gray_shape = gray_image.shape[::-1]


    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print("Error detecting corners in image %s. Exiting" % image_file)
            ret = False

    return cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

def abs_sobel_thresh(image, orient='x', thresh=(0,255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        print("bad orient passed")
        exit()
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def color_rgb_threshold(image, channel='r', thresh=(0,255)):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    if channel == 'r':
        convert_channel = R
    elif channel == 'G':
        convert_channel = G
    elif channel == 'b':
        convert_channel = b
    else:
        print("bad channel")
        exit()
    binary = np.zeros_like(convert_channel)
    binary[(convert_channel > thresh[0]) & (convert_channel <= thresh[1])] = 1
    return binary

def color_hls_threshold(image, channel='h', thresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    if channel == 'h':
        convert_channel = H
    elif channel == 'l':
        convert_channel = L
    elif channel == 's':
        convert_channel = S
    else:
        print("bad channel")
        exit()
    binary = np.zeros_like(convert_channel)
    binary[(convert_channel > thresh[0]) & (convert_channel <= thresh[1])] = 1
    return binary

def trap_area(deminsions):
    bt1 = np.float32(deminsions[0][0])
    bt2 = np.float32(deminsions[3][0])
    bb1 = np.float32(deminsions[1][0])
    bb2 = np.float32(deminsions[2][0])
    h1 = np.float32(deminsions[0][1])
    h2 = np.float32(deminsions[1][1])
    baset = abs(bt1-bt2)
    baseb = abs(bb1-bb2)
    height = abs(h1-h2)
    return ((baset + baseb)/2.0)*height +29

def rec_deminsions(rec_deminsions, trap_area):
    new_demin = np.copy(rec_deminsions)
    new_demin[0][0] = rec_deminsions[1][0]
    new_demin[3][0] = rec_deminsions[2][0]
    sq_width = abs(rec_deminsions[1][0] - rec_deminsions[2][0])
    sq_height = trap_area/sq_width
    new_demin[0][1] = int(new_demin[0][1] - sq_height)
    new_demin[3][1] = int(new_demin[3][1] - sq_height)
    return new_demin

def prospective_transform(image, src, dst, area):
   
    img_size = (image.shape[1], image.shape[0])
    plt.imshow(image)
    plt.plot(src[0][0], src[0][1], '.')
    plt.plot(src[1][0], src[1][1], '.')
    plt.plot(src[2][0], src[2][1], '.')
    plt.plot(src[3][0], src[3][1], '.')

    plt.plot(dst[0][0], dst[0][1], 'x')
    plt.plot(dst[1][0], dst[1][1], 'x')
    plt.plot(dst[2][0], dst[2][1], 'x')
    plt.plot(dst[3][0], dst[3][1], 'x')
    plt.axvline(x=1020)
    plt.show()
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def drawline_histogram(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

                # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

def pipeline(image, mtx, dist, src, dst, area):
    new_image = cv2.undistort(image, mtx, dist, None, mtx)

   
    xsobel = abs_sobel_thresh(new_image,'x', (20, 100))

    ysobel = abs_sobel_thresh(new_image,'y', (20, 100))

    magthresh =  mag_thresh(new_image,3, (30, 100))
   
    dirthresh = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    
    combined = np.zeros_like(dirthresh)
    combined[((xsobel == 1) & (ysobel == 1)) | ((magthresh == 1) & (dirthresh == 1))] = 1

    #combined[((xsobel == 1)) | ((magthresh == 1) & (dirthresh == 1))] = 1

   
    r =  color_rgb_threshold(new_image, channel='r', thresh=(200,255))
    s =  color_hls_threshold(new_image, channel='s', thresh=(90,200))
    
    color_combined =np.zeros_like(r)
    color_combined[(r == 1) & (s == 1)] = 1
    #color_combined[(s == 1)] = 1
    combined_binary = np.zeros_like(color_combined)
    combined_binary[(color_combined == 1) | (combined == 1)] = 1
    
    color_binary = np.dstack(( np.zeros_like(combined), combined, color_combined))
    combined_binary = prospective_transform(combined_binary, src, dst, area)
    drawline_histogram(combined_binary)
    #compare_lines_images(color_binary, combined, cmap='gray')
    #compare_lines_images
    #compare_lines_images(combined_binary, image, src, dst, cmap='gray')

#    compare_dot_images(image, new_image, src, dst)
#Main

src = np.float32([[716, 470],
                  [1009, 660],
                  [288, 660],
                  [567, 470]])

area = trap_area(src)
dst = rec_deminsions(src, area)

#1. Calibrate Camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera(9, 6)

images = []
image_files = glob.glob('./test_images/*.jpg')

for image_file in image_files:
    images.append(bgr_to_rgb(cv2.imread(image_file)))


#img_size = images[0].shape
#src = np.array([[585. /1280.*img_size[1], 455./720.*img_size[0]],
#                [705. /1280.*img_size[1], 455./720.*img_size[0]],
#                [1130./1280.*img_size[1], 720./720.*img_size[0]],
#                [190. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)

#dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
#                [1000./1280.*img_size[1], 100./720.*img_size[0]],
#                [1000./1280.*img_size[1], 720./720.*img_size[0]],
#                [300. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)

src = np.asarray([[240, 686], [1060, 686], [738, 480], [545 ,480]], np.float32)
dst = np.asarray([[300, 700], [1000, 700], [1000 , 300], [300, 300]], np.float32)

for image in images:
    pipeline(image, mtx, dist, src,dst, area)

exit()
#test code



    

for image in images:

    xsobel = abs_sobel_thresh(new_image,'x', (20, 100))
    ysobel = abs_sobel_thresh(new_image,'y', (20, 100))
    magthresh =  mag_thresh(new_image,3, (30, 100))
    dirthresh = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    combined = np.zeros_like(dirthresh)
    combined[((xsobel == 1) & (ysobel == 1)) | ((magthresh == 1) & (dirthresh == 1))] = 1

    r =  color_rgb_threshold(image, channel='r', thresh=(200,255))
    s =  color_hls_threshold(image, channel='s', thresh=(90,255))

    color_combined =np.zeros_like(r)
    color_combined[(r == 1) & (s == 1)] = 1
    combined_binary = np.zeros_like(color_combined)
    combined_binary[(color_combined == 1) | (combined == 1)] = 1

    color_binary = np.dstack(( np.zeros_like(combined), combined, color_combined))
    compare_images(combined_binary, color_binary, cmap='gray')


'''
1. there were three images that would not work with the findChessboardCorners function in calibrate camera.  These images looked valuable as they had some fish eye
'''
