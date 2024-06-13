import os
import numpy as np

from ImageVisualizer import ImageVisualizer
from LiDARVisualizer import LiDARVisualizer 

class DataLoader:
    def __init__(self):
        self.folder_path = None
        self.cameraWidgets = []
        self.lidarWidgets = []
        
    def setDataset(self, path):
        self.folder_path = path
        self.cameraWidgets.clear()
        self.lidarWidgets.clear()
        
        sensor_list = os.listdir(self.folder_path)
        for sensor in sensor_list:
            sensor_path = os.path.join(self.folder_path, sensor)
            if os.path.exists(os.path.join(sensor_path, 'Image')):
                cameraWidget = ImageVisualizer()
                cameraWidget.setTitle(sensor)
                cameraWidget.setRootPath(sensor_path)
                self.cameraWidgets.append(cameraWidget)
            elif os.path.exists(os.path.join(sensor_path, 'LiDAR')):
                lidarWidget = LiDARVisualizer()
                lidarWidget.setTitle(sensor)
                lidarWidget.setRootPath(sensor_path)
                self.lidarWidgets.append(lidarWidget)
        
    def itemSelected(self, sub_dir, item):
        for cameraWidget in self.cameraWidgets:
            cameraWidget.setItem(sub_dir, item)
        for lidarWidget in self.lidarWidgets:
            lidarWidget.setItem(sub_dir, item)
        
        