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
    
    
    binary_output =  np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    
    # 5) Create a binary mask where direction thresholds are met
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
    return ((baset + baseb)/2.0)*height

def rec_deminsions(rec_deminsions, trap_area):
    new_demin = np.copy(rec_deminsions)
    print(new_demin)
    new_demin[0][0] = rec_deminsions[1][0]
    new_demin[3][0] = rec_deminsions[2][0]


    sq_width = abs(rec_deminsions[1][0] - rec_deminsions[2][0])

    sq_height = trap_area/sq_width
    new_demin[0][1] = int(new_demin[0][1] - sq_height)
    new_demin[3][1] = int(new_demin[3][1] - sq_height)
    print(sq_height)
    print(new_demin)
    return new_demin


def prospective_transform(image, src, dst, area):
   
   # plt.plot(260, 680, '.')
   # plt.plot(525, 500, '.')
   # plt.plot(1050, 680, '.')
   # plt.plot(765, 500, '.')
   # plt.plot(565, 470, '.')
   # plt.plot(720, 470, '.')
   # plt.show()

    img_size = (image.shape[1], image.shape[0])
 
    #src = np.float32([[763, 500],
    #                  [893, 586],
    #                  [400, 586],
    #                  [525, 500]])

   # src = np.float32([[763, 500],
   #                   [845, 550],
   #                   [450, 550],
   #                   [525, 500]])

   # src = np.float32([[763, 500],
   #                   [791, 520],
   #                   [495, 520],
   #                   [525, 500]])
   
   # area = trap_area(src)
    #dst = rec_deminsions(src, area)
    plt.imshow(image)
    plt.plot(src[0][0], src[0][1], '.')
    plt.plot(src[1][0], src[1][1], '.')
    plt.plot(src[2][0], src[2][1], '.')
    plt.plot(src[3][0], src[3][1], '.')

    plt.plot(dst[0][0], dst[0][1], 'x')
    plt.plot(dst[1][0], dst[1][1], 'x')
    plt.plot(dst[2][0], dst[2][1], 'x')
    plt.plot(dst[3][0], dst[3][1], 'x')
    plt.show()
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
#Main
src = np.float32([[716, 470],
                   [1009, 660],
                   [288, 660],
                   [567, 470]])

area = trap_area(src)
dst = rec_deminsions(src, area)
#1. Calibrate Camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera(9, 6)


#test code

images = []
image_files = glob.glob('./test_images/*.jpg')

for image_file in image_files:
    images.append(bgr_to_rgb(cv2.imread(image_file)))

for image in images:
    new_image = cv2.undistort(image, mtx, dist, None, mtx)
    p = prospective_transform(new_image, src, dst, area)
    compare_dot_images(new_image, p, src, dst)
exit()
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
