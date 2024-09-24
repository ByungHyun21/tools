import cv2
import numpy as np
import os

# Load the image
image_paths = os.listdir('capture')

ph, pw = 7, 10

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((ph*pw,3), np.float32)
objp[:,:2] = np.mgrid[0:ph,0:pw].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

for img_path in image_paths:
    img = cv2.imread('capture/' + img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (ph,pw), None)

    if ret:
        objpoints.append(objp)  # Add object points
        imgpoints.append(corners)  # Add image points

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (ph,pw), corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('ret: ', ret)
print('mtx: ', mtx)
print('dist: ', dist)
print('rvecs: ', rvecs)
print('tvecs: ', tvecs)

# undistort the image
for img_path in image_paths:
    img = cv2.imread('capture/' + img_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imshow('Undistorted Image', dst)
    cv2.waitKey(100)