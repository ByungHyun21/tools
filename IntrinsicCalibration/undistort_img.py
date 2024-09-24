import os
import cv2
import numpy as np
import copy

k = [-0.39687487, 0.21475015, 0.00060158, -0.00052996, -0.07387129]
fx = 430.99048049
fy = 431.55761589
cx = 331.37292199
cy = 248.73673233

cap = cv2.VideoCapture(0)

print_once = True
while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # undistort
    # if print_once:
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array(k), (w, h), 1, (w, h))
    dst = cv2.undistort(frame, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array(k), None, newcameramtx)

    roi_x, roi_y, roi_w, roi_h = roi
    dst = dst[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    dst = cv2.resize(dst, (w, h))

    if print_once:
        print_once = False
        
        ratio_w = w / roi_w
        ratio_h = h / roi_h
        
        intrinsics = copy.deepcopy(newcameramtx)
        
        intrinsics[0, 0] *= ratio_w
        intrinsics[0, 2] *= ratio_w
        intrinsics[1, 1] *= ratio_h
        intrinsics[1, 2] *= ratio_h
        print('intrinsics: ', intrinsics)

    cv2.imshow('Undistorted Image', dst)
    cv2.waitKey(1)