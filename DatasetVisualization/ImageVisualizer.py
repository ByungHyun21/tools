import sys
import os
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import numpy as np
from scipy.spatial.transform import Rotation as R

import random
import cv2
import json

class ImageVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.root_path = None
        self.sensor_name = None
        self.config = {}
        self.config['show_box2d'] = True
        self.config['show_pose2d'] = True
        self.config['show_box3d'] = True
        self.config['show_pose3d'] = True
        
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        
        self.Limage = QLabel()
        # self.GLspace = gl.GLViewWidget()
        self.GLbox3d = None
        self.GLpose3d = None

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
        
        self.vlayout.addWidget(self.Limage)
        
        self.setLayout(self.hlayout)

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
        
        image_dir_path = os.path.join(self.root_path, 'Image', sub_dir)
        label_dir_path = os.path.join(self.root_path, 'Label', sub_dir)
        
        for image_name in os.listdir(image_dir_path):
            if item_name in image_name:
                image_path = os.path.join(image_dir_path, image_name)
                break
            
        label_path = os.path.join(label_dir_path, item_name + '.json')
        
        # image read
        image = cv2.imread(image_path)
        if image is None:
            return
        
        imh, imw, imc = image.shape
            
        # label read
        with open(label_path, 'r') as f:
            label = json.load(f)
        
        if 'intrinsic' in label:
            fx = label['intrinsic']['fx']
            fy = label['intrinsic']['fy']
            cx = label['intrinsic']['cx']
            cy = label['intrinsic']['cy']
            
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        if 'extrinsic' in label:
            ext_rotation = np.array(label['extrinsic']['rotation'], dtype=np.float32).reshape(3, 3)
            
            tx = label['extrinsic']['translation']['x']
            ty = label['extrinsic']['translation']['y']
            tz = label['extrinsic']['translation']['z']
            
            ext_translation = np.array([[tx], [ty], [tz]])
            
            extrinsic = np.hstack((ext_rotation, ext_translation))
            extrinsic = np.vstack((extrinsic, np.array([0, 0, 0, 1])))
            
            
        # draw label
        for obj in label['objects']:
            if 'box2d' in obj and self.config['show_box2d']:
                obj_cx = obj['box2d']['cx'] * imw
                obj_cy = obj['box2d']['cy'] * imh
                obj_w = obj['box2d']['w'] * imw
                obj_h = obj['box2d']['h'] * imh
                
                cv2.rectangle(image, (int(obj_cx - obj_w/2), int(obj_cy - obj_h/2)), (int(obj_cx + obj_w/2), int(obj_cy + obj_h/2)), (0, 255, 0), 2)
                
            if 'pose2d' in obj and self.config['show_pose2d']:
                pass
                
            if 'box3d' in obj and self.config['show_box3d']:
                obj_width = obj['box3d']['size']['width']
                obj_height = obj['box3d']['size']['height']
                obj_length = obj['box3d']['size']['length']
                obj_tx = obj['box3d']['translation']['x']
                obj_ty = obj['box3d']['translation']['y']
                obj_tz = obj['box3d']['translation']['z']
                
                obj_rotation = np.array(obj['box3d']['rotation'], dtype=np.float32).reshape(3, 3)
                obj_translation = np.array([[obj_tx], [obj_ty], [obj_tz]])
                obj_extrinsic = np.hstack((obj_rotation, obj_translation))
                obj_extrinsic = np.vstack((obj_extrinsic, np.array([0, 0, 0, 1])))
                
                obj_points = np.array([
                    [obj_length, obj_length, obj_length, obj_length, 0, 0, 0, 0],
                    [0, 0, obj_height, obj_height, 0, 0, obj_height, obj_height],
                    [0, obj_width, obj_width, 0, 0, obj_width, obj_width, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                ])
                obj_points[0, :] -= obj_length / 2
                obj_points[1, :] -= obj_height / 2
                obj_points[2, :] -= obj_width / 2
                
                obj_points = np.dot(obj_extrinsic, obj_points)
                obj_points = np.dot(intrinsic, obj_points[:3, :])
                obj_points = obj_points / obj_points[2, :]
                obj_points = obj_points[:2, :]
                
                edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                        [4, 5], [5, 6], [6, 7], [7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]]
                
                for edge in edges:
                    x1, y1 = obj_points[:, edge[0]]
                    x2, y2 = obj_points[:, edge[1]]
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
            if 'pose3d' in obj and self.config['show_pose3d']:
                pass
        
        # image display
        # image max width 800
        if imw > 800:
            new_w = 800
            new_h = int(imh * 800 / imw)
            image = cv2.resize(image, (new_w, new_h))
        if imh > 600:
            new_h = 600
            new_w = int(imw * 600 / imh)
            image = cv2.resize(image, (new_w, new_h))
        image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image)
        self.Limage.setPixmap(pixmap)

    
    def setConfig(self, config):
        self.config = config