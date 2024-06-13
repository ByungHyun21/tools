import sys
import os
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import numpy as np

import random
import cv2
import json

class LiDARVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.root_path = None
        self.sensor_name = None
        self.config = {}
        
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
    
        self.setGeometry(100, 100, 800, 600)

    def setTitle(self, title):
        self.sensor_name = title
        self.setWindowTitle(title)
    
    def setRootPath(self, root_path):
        self.root_path = root_path
        
    def setItem(self, sub_dir, item_name):
        print('Item Name:', item_name)
        pass
    
    def setConfig(self, config):
        self.config = config
        