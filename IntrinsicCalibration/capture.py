import cv2
import numpy as np
import os

# directory
if not os.path.exists('capture'):
    os.makedirs('capture')

cap = cv2.VideoCapture(0)

cnt = 0
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"capture/frame_{cnt}.png", frame)
        cnt += 1