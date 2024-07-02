import sys
import os
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import random
import cv2
import json

class LiDARVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.root_path = None
        self.sensor_name = None
        self.config = {}
        self.config['show_box3d'] = True
        
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        
        self.GLspace = gl.GLViewWidget()

        self.initUI()
        self.show()
        
    def initUI(self):
        x = self.screen_width * 0.1 + self.screen_width * (0.4 - 0.1) * random.random()
        y = self.screen_height * 0.1 + self.screen_height * (0.4 - 0.1) * random.random()
        self.setGeometry(x, y, 800, 600)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        
        self.hlayout = QHBoxLayout()
        self.vlayout = QVBoxLayout()
        self.hlayout.addLayout(self.vlayout)
        
        self.vlayout.addWidget(self.GLspace)
        
        self.setLayout(self.hlayout)
        
        self.setGeometry(100, 100, 800, 600)

    def setTitle(self, title):
        self.sensor_name = title
        self.setWindowTitle(title)
    
    def setRootPath(self, root_path):
        self.root_path = root_path
        
    def setItem(self, sub_dir, item_name):
        if self.root_path is None:
            return
        if sub_dir is None:
            return
        if item_name is None:
            return
        
        self.GLspace.clear()
        
        lidar_dir_path = os.path.join(self.root_path, 'LiDAR', sub_dir)
        label_dir_path = os.path.join(self.root_path, 'Label', sub_dir)
        
        for lidar_name in os.listdir(lidar_dir_path):
            if item_name in lidar_name:
                lidar_path = os.path.join(lidar_dir_path, lidar_name)
                break
            
        label_path = os.path.join(label_dir_path, item_name + '.json')
        
        if not os.path.exists(lidar_path):
            return
        if not os.path.exists(label_path):
            return
        
        with open(label_path, 'r') as f:
            label = json.load(f)
            
        columns = label['columns']
        
        # lidar read
        pointcloud = np.fromfile(lidar_path, dtype=np.float32)
        n_columns = len(columns)
        pointcloud = pointcloud.reshape(-1, n_columns)
        pc_xidx = columns.index('x')
        pc_yidx = columns.index('y')
        pc_zidx = columns.index('z')
        
        
        if 'intensity' in columns:
            intensity_idx = columns.index('intensity')
            intensity = pointcloud[:, intensity_idx]
            
            # intensity 값을 8비트 정수형으로 변환
            intensity = intensity.astype(np.uint8)
            
            mean = np.mean(intensity)
            std = np.std(intensity)
            
            # 2 시그마를 벗어나는 값을 2 시그마에 위치한 값으로 대체
            lower_bound = mean - 2 * std
            upper_bound = mean + 2 * std

            adjusted_data = np.clip(intensity, lower_bound, upper_bound)
            
            # Find the histogram and the bins
            hist, bins = np.histogram(adjusted_data, bins=256, density=True)
            cdf = hist.cumsum()  # Cumulative distribution function
            cdf_normalized = cdf * (bins[:-1] - bins[0])
            
            # Normalize the cdf
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            
            equalized_data = cdf[adjusted_data.astype('uint8')]
            
            # colormap 적용을 위한 준비
            equalized_intensity = equalized_data.astype('uint8')

            # 예시 colormap
            cmap = plt.get_cmap('turbo')

            # 결과값을 cmap.map(equalized_intensity, mode='byte')에 넣음
            colors = cmap(equalized_intensity / 255, bytes=True)
            colors = colors / 255.0
            
            pointcloud = pointcloud[:, [pc_xidx, pc_yidx, pc_zidx]]
            self.GLspace.addItem(gl.GLScatterPlotItem(pos=pointcloud, size=1, color=colors))
        else:
            # pointcloud visualization
            pointcloud = pointcloud[:, [pc_xidx, pc_yidx, pc_zidx]]
            self.GLspace.addItem(gl.GLScatterPlotItem(pos=pointcloud, size=1, color=(1, 1, 1, 1)))
        
        # label visualization
        for obj in label['objects']:
            obj_class = obj['class']
            
            if 'box3d' in obj and self.config['show_box3d']:
                obj_width = obj['box3d']['size']['width']
                obj_height = obj['box3d']['size']['height']
                obj_length = obj['box3d']['size']['length']
                obj_rotation = np.array(obj['box3d']['rotation'], dtype=np.float32).reshape(3, 3)
                obj_tx = obj['box3d']['translation']['x']
                obj_ty = obj['box3d']['translation']['y']
                obj_tz = obj['box3d']['translation']['z']
                obj_translation = np.array([obj_tx, obj_ty, obj_tz]).reshape(3, 1)
                
                points = np.array([
                    [obj_length, obj_length, obj_length, obj_length, -obj_length, -obj_length, -obj_length, -obj_length],
                    [obj_width, obj_width, -obj_width, -obj_width, obj_width, obj_width, -obj_width, -obj_width],
                    [obj_height, -obj_height, -obj_height, obj_height, obj_height, -obj_height, -obj_height, obj_height]
                ])
                
                points[0, :] = points[0, :] / 2
                points[1, :] = points[1, :] / 2
                points[2, :] = points[2, :] / 2
                
                points = np.dot(obj_rotation, points)
                points = points + obj_translation
                
                edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                            [4, 5], [5, 6], [6, 7], [7, 4],
                            [0, 4], [1, 5], [2, 6], [3, 7]]
                
                for edge in edges:
                    p1 = points[:, edge[0]]
                    p2 = points[:, edge[1]]
                    self.GLspace.addItem(gl.GLLinePlotItem(pos=np.array([p1, p2]), color=(0, 1, 0, 1)))
                    
        # draw lidar axis
        # x-axis red, y-axis green, z-axis blue
        axis = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
        self.GLspace.addItem(gl.GLLinePlotItem(pos=axis[[0, 1]], color=(1, 0, 0, 1)))
        self.GLspace.addItem(gl.GLLinePlotItem(pos=axis[[0, 2]], color=(0, 1, 0, 1)))
        self.GLspace.addItem(gl.GLLinePlotItem(pos=axis[[0, 3]], color=(0, 0, 1, 1)))
        
    
    
    def setConfig(self, config):
        self.config = config
        