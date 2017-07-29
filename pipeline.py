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

def compare_images(image1, image2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(image2)
    ax2.set_title('Processed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
def calibrate_camera(nx, ny, images):
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane
    gray_image = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    gray_shape = gray_image.shape[::-1]
    # Prepare object points like (0,0,0, (1,0,0), (2,0,0)....,(7,4,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

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
    


calibration_images = []
image_files = glob.glob('./camera_cal/calibration*.jpg')

for image_file in image_files:
    calibration_images.append(cv2.imread(image_file))



ret, mtx, dist, rvecs, tvecs = calibrate_camera(9, 6, calibration_images)

for image in calibration_images:
    compare_images(bgr_to_rgb(image), cv2.undistort(bgr_to_rgb(image), mtx, dist, None, mtx))


'''
1. there were three images that would not work with the findChessboardCorners function in calibrate camera.  These images looked valuable as they had some fish eye
'''
