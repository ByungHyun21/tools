import sys
import os
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

import pyqtgraph as pg
import numpy as np

from func.func import *
from widget.Widgets import *

class PageManual(QWidget):
    def __init__(self):
        super().__init__()
        self.root_path = QDir.rootPath()
        
        self.initUI()
        
    def initUI(self):
        self.layout_0 = QVBoxLayout()
        self.setLayout(self.layout_0)
        
        self.layout_top = QHBoxLayout()
        self.layout_0.addLayout(self.layout_top)
        
        self.layout_top_spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout_top.addSpacerItem(self.layout_top_spacer)
        
        self.layout_top_label_directory = QLabel("Directory: ")
        self.layout_top.addWidget(self.layout_top_label_directory)
        
        self.layout_top_lineedit_directory = QLabel(" ")
        self.layout_top.addWidget(self.layout_top_lineedit_directory)
        
        self.layout_top_button_directory = QPushButton("Select Directory")
        self.layout_top_button_directory.clicked.connect(self.select_folder)
        self.layout_top.addWidget(self.layout_top_button_directory)
        
        self.layout_top_help = QPushButton("Help")
        self.layout_top_help.clicked.connect(self.layout_top_help_clicked)
        self.layout_top.addWidget(self.layout_top_help)
        
        
        # Layout1: 디렉토리 선택, 도움말
        self.layout_1 = QHBoxLayout()
        self.layout_0.addLayout(self.layout_1)
        
        # layout2: 좌우 분할
        self.layout_2 = QHBoxLayout()
        self.layout_0.addLayout(self.layout_2)
        
        # layout3: 상하 분할
        self.layout_viewer = QVBoxLayout()
        self.layout_2.addLayout(self.layout_viewer)
        
        # LiDAR Viewer
        self.tab_lidar = QTabWidget()
        self.layout_viewer.addWidget(self.tab_lidar)
        
        # Camera Viewer
        self.tab_camera = QTabWidget()
        self.layout_viewer.addWidget(self.tab_camera)
        
        # layout4: 상하 분할
        self.layout_4 = QVBoxLayout()
        self.layout_2.addLayout(self.layout_4)
        
        # Folder Structure
        self.label_folder_structure = QLabel("Folder Structure")
        self.layout_4.addWidget(self.label_folder_structure)
        
        self.folder_tree = QTreeView()
        self.folder_tree.header().hide()
        self.folder_tree.clicked.connect(self.treeItemClicked)
        self.layout_4.addWidget(self.folder_tree)
        
        self.folder_tree_model = QStandardItemModel()
        
        self.tab_param = QTabWidget()
        self.layout_4.addWidget(self.tab_param)
        
        self.cameraWidgets = {}
        self.lidarWidgets = {}
        
        self.layout_bottom = QHBoxLayout()
        self.layout_0.addLayout(self.layout_bottom)
        
        self.progress_bar = QProgressBar()
        self.layout_bottom.addWidget(self.progress_bar)
        self.progress_bar.setValue(0)
        
        self.progress_label = QLabel("Progress: ")
        self.layout_bottom.addWidget(self.progress_label)
        
        self.progress_label_value = QLabel(" ")
        self.layout_bottom.addWidget(self.progress_label_value)
        self.progress_label_value.setFixedWidth(200)

    def select_folder(self):
        self.folder_tree_model.clear()
        
        camera_list = []
        lidar_list = []
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", self.root_path)
        self.root_path = folder_path
        if folder_path:
            self.layout_top_lineedit_directory.setText(folder_path)
            
            folder_list = os.listdir(folder_path)
            for folder in folder_list:
                if os.path.isdir(os.path.join(folder_path, folder, 'Image')):
                    camera_list.append(folder)
                
                if os.path.isdir(os.path.join(folder_path, folder, 'LiDAR')):
                    lidar_list.append(folder)
        
        self.tab_param.clear()    
        self.cameraWidgets = {}
        self.tab_camera.clear()
        self.lidarWidgets = {}
        self.tab_lidar.clear()
        
        for camera_name in camera_list:
            new_cameraWidget = CameraWidget()
            new_cameraWidget.setName(camera_name)
            new_cameraWidget.setParentPage(self)
            new_cameraWidget.setDataTab(self.tab_camera)
            new_cameraWidget.setParamTab(self.tab_param)
            self.cameraWidgets[camera_name] = new_cameraWidget
            self.tab_camera.addTab(new_cameraWidget.image, camera_name)
            self.tab_param.addTab(new_cameraWidget.param, camera_name)
                
        for lidar_name in lidar_list:
            new_lidarWidget = LiDARWidget()
            new_lidarWidget.setName(lidar_name)
            new_lidarWidget.setParentPage(self)
            new_lidarWidget.setDataTab(self.tab_lidar)
            new_lidarWidget.setParamTab(self.tab_param)
            self.lidarWidgets[lidar_name] = new_lidarWidget
            self.tab_lidar.addTab(new_lidarWidget.lidar, lidar_name)
            self.tab_param.addTab(new_lidarWidget.param, lidar_name)
            
        # add lidar name to camera widget
        for name, widget in self.cameraWidgets.items():
            widget.setLiDARList(lidar_list)
            
        # add camera name to lidar widget
        for name, widget in self.lidarWidgets.items():
            widget.setCameraList(camera_list)
            
        item_list = {}
        for camera_name in camera_list:
            self.updateProgressBar(0, f"{camera_name} 데이터 탐색중...")
            subDirList = os.listdir(os.path.join(folder_path, camera_name, 'Image'))
            for cnt, subDir in enumerate(subDirList):
                if subDir not in item_list:
                    item_list[subDir] = []

                files = os.listdir(os.path.join(folder_path, camera_name, 'Image', subDir))
                
                for file in files:
                    file = file.split('.')[0]
                    if file not in item_list[subDir]:
                        item_list[subDir].append(file)
                    
                self.updateProgressBar(cnt/len(subDirList)*100, f"{camera_name} 데이터 탐색중...")
                    
        for lidar_name in lidar_list:
            self.updateProgressBar(cnt/len(subDirList)*100, f"{lidar_name} 데이터 탐색중...")
            subDirList = os.listdir(os.path.join(folder_path, lidar_name, 'LiDAR'))
            for cnt, subDir in enumerate(subDirList):
                if subDir not in item_list:
                    item_list[subDir] = []

                files = os.listdir(os.path.join(folder_path, lidar_name, 'LiDAR', subDir))
                
                for file in files:
                    file = file.split('.')[0]
                    if file not in item_list[subDir]:
                        item_list[subDir].append(file)
                    
                self.updateProgressBar(cnt/len(subDirList)*100, f"{lidar_name} 데이터 탐색중...")
            
        cnt = 0    
        for key, value in item_list.items():
            parent = QStandardItem(key)
            parent.setFlags(parent.flags() & ~Qt.ItemIsEditable)
            for val in value:
                child = QStandardItem(val)
                child.setFlags(child.flags() & ~Qt.ItemIsEditable)
                parent.appendRow(child)
            self.folder_tree_model.appendRow(parent)
            cnt += 1
            
        self.folder_tree.setModel(self.folder_tree_model)
        self.updateProgressBar(100, "폴더 열기 완료")
            
    def layout_top_help_clicked(self):
        pass
    
    def updateProgressBar(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_label_value.setText(text)
        
    def treeItemClicked(self, index):
        item = self.folder_tree_model.itemFromIndex(index)
        parent_item = item.parent()

        self.updateProgressBar(0, "데이터 탐색중...")
        if parent_item is not None:
            for name, widget in self.cameraWidgets.items():
                dirPath = os.path.join(self.layout_top_lineedit_directory.text(), name)
                widget.openItem(dirPath, parent_item.text(), item.text())
                
            self.updateProgressBar(50, "데이터 탐색중...")
            
            for name, widget in self.lidarWidgets.items():
                dirPath = os.path.join(self.layout_top_lineedit_directory.text(), name)
                widget.openItem(dirPath, parent_item.text(), item.text())
                
            self.updateProgressBar(100, "데이터 탐색 완료")
            
            for name, widget in self.cameraWidgets.items():
                widget.updateRequest()
                widget.updateView()
                
            for name, widget in self.lidarWidgets.items():
                widget.updateRequest()
                widget.updateView()
            
    def updateFrame(self):
        for name, widget in self.cameraWidgets.items():
            widget.updateView()
            
        for name, widget in self.lidarWidgets.items():
            widget.updateView()
    
    def updateRequestFromCamera(self, camera_name, lidar_name):
        camera_widget = self.cameraWidgets[camera_name]
        data_C, param_C = camera_widget.getCameraData()
        
        lidar_widget = self.lidarWidgets[lidar_name]
        lidar_widget.setCameraData(data_C, param_C)
        data_L, param_L, objects_L = lidar_widget.getLiDARData()
        
        camera_widget.setLiDARData(data_L, param_L, objects_L)
        
        self.updateFrame()
    
    def updateRequestFromLiDAR(self, lidar_name, camera_name):
        lidar_widget = self.lidarWidgets[lidar_name]
        data_L, param_L, objects_L = lidar_widget.getLiDARData()
        
        camera_widget = self.cameraWidgets[camera_name]
        camera_widget.setLiDARData(data_L, param_L, objects_L)
        data_C, param_C = camera_widget.getCameraData()
        
        lidar_widget.setCameraData(data_C, param_C)
        
        self.updateFrame()
        
            