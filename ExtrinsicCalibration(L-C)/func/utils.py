import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os

def readDirectory(directory):
    files = os.listdir(directory)
    files = [file for file in files if file.endswith('.png')]
    files.sort()
    return files