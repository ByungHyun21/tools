import sys
import os
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import ColorMap

import pyqtgraph as pg
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import math
import copy

from func.func import *

class LiDARDataWidget(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.parentWidget = None
        self.paramWidget = None
        
        self.pc = None
        self.colors = None
        
        self.opts['distance'] = 30
        self.opts['elevation'] = 30
        self.opts['azimuth'] = 45
        self.opts['fov'] = 60
        self.opts['up'] = pg.QtGui.QVector3D(0, 0, 1)
        self.opts['center'] = pg.QtGui.QVector3D(0, 0, 0)
        
        self.cameraData = None
        self.cameraParam = None
        
    def setParentWidget(self, parentWidget):
        self.parentWidget = parentWidget

    def setData(self, itemPath, columns):
        # bin read
        with open(itemPath, 'rb') as f:
            data = f.read()
            
        # bin to numpy array
        data = np.frombuffer(data, dtype=np.float32)
        data = data.reshape(-1, len(columns))
        
        self.pc = data
        if self.pc.shape[1] == 4:
            intensity = self.pc[:, 3]
            
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
            self.colors = cmap(equalized_intensity / 255, bytes=True)
            

            # Map intensity values to colors
            # self.colors = cmap.map(equalized_intensity, mode='byte')
        else:
            self.colors = np.ones((self.pc.shape[0], 4)) * np.array([255, 255, 255, 255])
        
    def updateView(self):
        if self.pc is None:
            return
        # Clear all items
        self.items = []
        
        self.axis = gl.GLAxisItem()
        self.addItem(self.axis)
        
        if self.pc.shape[1] == 4:
            intensity = self.pc[:, 3]
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
            self.colors = cmap(equalized_intensity / 255, bytes=True)

            # Map intensity values to colors
            # self.colors = cmap.map(equalized_intensity, mode='byte')
        else:
            self.colors = np.ones((self.pc.shape[0], 4)) * np.array([255, 255, 255, 255])
        
        pc_color = copy.deepcopy(self.colors)
        if self.paramWidget.checkColorFromCamera.isChecked():
            if self.cameraData is not None and self.cameraParam is not None:
                image = cv2.imread(self.cameraData['image'])
                fx = float(self.cameraParam['fx'])
                fy = float(self.cameraParam['fy'])
                cx = float(self.cameraParam['cx'])
                cy = float(self.cameraParam['cy'])
                Rx_C = math.radians(float(self.cameraParam['Rx']))
                Ry_C = math.radians(float(self.cameraParam['Ry']))
                Rz_C = math.radians(float(self.cameraParam['Rz']))
                tx_C = float(self.cameraParam['tx'])
                ty_C = float(self.cameraParam['ty'])
                tz_C = float(self.cameraParam['tz'])
                
                intrinsic_C = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                rot_C = eulerAnglesToRotationMatrix(np.array([Rx_C, Ry_C, Rz_C]), 'ZYX')
                trans_C = np.array([tx_C, ty_C, tz_C]).reshape(3, 1)
                ext_C = np.vstack([np.hstack([rot_C, trans_C]), np.array([0, 0, 0, 1])])
                
                Rx_L = math.radians(self.paramWidget.Rx)
                Ry_L = math.radians(self.paramWidget.Ry)
                Rz_L = math.radians(self.paramWidget.Rz)
                tx_L = self.paramWidget.tx
                ty_L = self.paramWidget.ty
                tz_L = self.paramWidget.tz
                
                rot_L = eulerAnglesToRotationMatrix(np.array([Rx_L, Ry_L, Rz_L]), 'ZYX')
                trans_L = np.array([tx_L, ty_L, tz_L]).reshape(3, 1)
                ext_L = np.vstack([np.hstack([rot_L, trans_L]), np.array([0, 0, 0, 1])])
                
                ext_L2C = np.dot(ext_C, np.linalg.inv(ext_L))
                
                pc = self.pc[:, :3].T
                pc = np.vstack([pc, np.ones(pc.shape[1])])
                pc = np.dot(ext_L2C, pc)
                pc = pc[:3, :]
                
                # z>0 idx
                idx = pc[2, :] > 0
                
                pc = np.dot(intrinsic_C, pc)
                pc = pc[:2, :] / pc[2, :]
                
                # get color from image
                for i in range(pc.shape[1]):
                    if idx[i]:
                        x = int(pc[0, i])
                        y = int(pc[1, i])
                        if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
                            pc_color[i] = np.hstack([image[y, x], 255])
                        
        
        
        pointCloud = gl.GLScatterPlotItem(pos=self.pc[:, :3], size=2, color=pc_color/255.0)
        self.addItem(pointCloud)
        
        if self.paramWidget is not None:
            if self.paramWidget.objects is not None:
                self.drawObjects(self.paramWidget.objects)
        
    def setParamWidget(self, paramWidget):
        self.paramWidget = paramWidget
        
    def drawObjects(self, objects):
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])
        
        if self.paramWidget.checkShowObject.isChecked() == False:
            return
        
        for obj in objects:
            objectClass = obj['class']
            if 'box3d' in obj:
                box3d = obj['box3d']
                width = float(box3d['size']['width'])
                length = float(box3d['size']['length'])
                height = float(box3d['size']['height'])
                
                tx = float(box3d['translation']['x'])
                ty = float(box3d['translation']['y'])
                tz = float(box3d['translation']['z'])
                translation = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
                rotation = np.array(box3d['rotation'], dtype=np.float32).reshape(3, 3)
                
                extObject = np.vstack([np.hstack([rotation, translation]), np.array([0, 0, 0, 1])])
                
                boxPoints = np.array([
                    [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2],
                    [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2],
                    [-height/2, -height/2, -height/2, -height/2, height/2, height/2, height/2, height/2],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                ])
                
                boxPoints = np.dot(extObject, boxPoints)
                boxPoints = boxPoints[:3, :]
                for edge in edges:
                    line = gl.GLLinePlotItem(pos=np.array([boxPoints[:, edge[0]], boxPoints[:, edge[1]]]), color=(1, 0, 0, 1), width=2, antialias=True)
                    self.addItem(line)
        

class LiDARParamWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        self.columns = None
        self.Rx = None
        self.Ry = None
        self.Rz = None
        self.tx = None
        self.ty = None
        self.tz = None
        
        self.currentRotationType = 'ZYX'
        self.prevRotationType = 'ZYX'
        
        self.objects = None
        
        self.parentWidget = None
        self.dataWidget = None
        
    def setParentWidget(self, parentWidget):
        self.parentWidget = parentWidget
        
    def initUI(self):
        self.layout = QHBoxLayout()
        
        # Left Layout
        self.layout_left = QVBoxLayout()
        
        # Flag
        self.layout_flag = QVBoxLayout()
        self.label_fromCamera = QLabel("From Camera")
        self.layout_flag.addWidget(self.label_fromCamera)
        self.layout_selectCamera = QHBoxLayout()
        self.label_selectCamera = QLabel("Select Camera: ")
        self.layout_selectCamera.addWidget(self.label_selectCamera)
        self.combo_selectCamera = QComboBox()
        self.combo_selectCamera.currentIndexChanged.connect(self.selectCameraChanged)
        self.layout_selectCamera.addWidget(self.combo_selectCamera)
        self.layout_flag.addLayout(self.layout_selectCamera)
        
        self.checkColorFromCamera = QCheckBox("Color From Camera")
        self.checkColorFromCamera.setChecked(False)
        self.checkColorFromCamera.stateChanged.connect(self.colorFromCameraChanged)
        self.layout_flag.addWidget(self.checkColorFromCamera)
        
        self.label_fromLiDAR = QLabel("From LiDAR")
        self.layout_flag.addWidget(self.label_fromLiDAR)
        self.checkShowObject = QCheckBox("Show Object")
        self.checkShowObject.setChecked(True)
        self.checkShowObject.stateChanged.connect(self.showObjectChanged)
        self.layout_flag.addWidget(self.checkShowObject)
        self.layout_left.addLayout(self.layout_flag)
        
        self.layout_spacer_left = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout_left.addSpacerItem(self.layout_spacer_left)
        
        # Right Layout
        self.layout_right = QVBoxLayout()
        
        # LiDAR Extrinsic Parameters(Rotation)
        self.rotationWidget = QWidget()
        self.layoutRotation = QVBoxLayout()
        self.layoutRotation_label = QLabel("Rotation Parameters")
        self.layoutRotation.addWidget(self.layoutRotation_label)
        self.layoutRotation_table = QGridLayout()
        # 4x2 table
        self.layoutRotation_table.addWidget(QLabel("Type"), 0, 0)
        self.layoutRotation_table.addWidget(QLabel("Rx"), 1, 0)
        self.layoutRotation_table.addWidget(QLabel("Ry"), 2, 0)
        self.layoutRotation_table.addWidget(QLabel("Rz"), 3, 0)
        self.rotationTypeList = QComboBox()
        rotations = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
        self.rotationTypeList.addItems(rotations)
        self.rotationTypeList.setCurrentIndex(5)
        self.rotationTypeList.currentIndexChanged.connect(self.rotationTypeChanged)
        
        self.layoutRotation_table.addWidget(self.rotationTypeList, 0, 1)
        
        self.layout_Rx = QHBoxLayout()
        self.RxWidget = QLineEdit()
        self.layout_Rx.addWidget(self.RxWidget)
        self.Rx_dec = QPushButton("-")
        self.Rx_dec.setFixedWidth(40)
        self.Rx_dec.clicked.connect(self.buttonRxDecClicked)
        self.layout_Rx.addWidget(self.Rx_dec)
        self.Rx_inc = QPushButton("+")
        self.Rx_inc.setFixedWidth(40)
        self.Rx_inc.clicked.connect(self.buttonRxIncClicked)
        self.layout_Rx.addWidget(self.Rx_inc)
        self.layoutRotation_table.addLayout(self.layout_Rx, 1, 1)
        
        self.layout_Ry = QHBoxLayout()
        self.RyWidget = QLineEdit()
        self.layout_Ry.addWidget(self.RyWidget)
        self.Ry_dec = QPushButton("-")
        self.Ry_dec.setFixedWidth(40)
        self.Ry_dec.clicked.connect(self.buttonRyDecClicked)
        self.layout_Ry.addWidget(self.Ry_dec)
        self.Ry_inc = QPushButton("+")
        self.Ry_inc.setFixedWidth(40)
        self.Ry_inc.clicked.connect(self.buttonRyIncClicked)
        self.layout_Ry.addWidget(self.Ry_inc)
        self.layoutRotation_table.addLayout(self.layout_Ry, 2, 1)
        
        self.layout_Rz = QHBoxLayout()
        self.RzWidget = QLineEdit()
        self.layout_Rz.addWidget(self.RzWidget)
        self.Rz_dec = QPushButton("-")
        self.Rz_dec.setFixedWidth(40)
        self.Rz_dec.clicked.connect(self.buttonRzDecClicked)
        self.layout_Rz.addWidget(self.Rz_dec)
        self.Rz_inc = QPushButton("+")
        self.Rz_inc.setFixedWidth(40)
        self.Rz_inc.clicked.connect(self.buttonRzIncClicked)
        self.layout_Rz.addWidget(self.Rz_inc)
        self.layoutRotation_table.addLayout(self.layout_Rz, 3, 1)
        self.RxWidget.editingFinished.connect(self.paramChanged)
        self.RyWidget.editingFinished.connect(self.paramChanged)
        self.RzWidget.editingFinished.connect(self.paramChanged)
        self.layoutRotation.addLayout(self.layoutRotation_table)
        self.rotationWidget.setLayout(self.layoutRotation)
        self.layout_right.addWidget(self.rotationWidget)
        
        # LiDAR Extrinsic Parameters(Translation)
        self.translationWidget = QWidget()
        self.layoutTranslation = QVBoxLayout()
        self.layoutTranslation_label = QLabel("Translation Parameters")
        self.layoutTranslation.addWidget(self.layoutTranslation_label)
        self.layoutTranslation_table = QGridLayout()
        # 3x2 table
        self.layoutTranslation_table.addWidget(QLabel("tx"), 0, 0)
        self.layoutTranslation_table.addWidget(QLabel("ty"), 1, 0)
        self.layoutTranslation_table.addWidget(QLabel("tz"), 2, 0)
        
        self.layout_tx = QHBoxLayout()
        self.txWidget = QLineEdit()
        self.layout_tx.addWidget(self.txWidget)
        self.tx_dec = QPushButton("-")
        self.tx_dec.setFixedWidth(40)
        self.tx_dec.clicked.connect(self.buttonTxDecClicked)
        self.layout_tx.addWidget(self.tx_dec)
        self.tx_inc = QPushButton("+")
        self.tx_inc.setFixedWidth(40)
        self.tx_inc.clicked.connect(self.buttonTxIncClicked)
        self.layout_tx.addWidget(self.tx_inc)
        self.layoutTranslation_table.addLayout(self.layout_tx, 0, 1)
        
        self.layout_ty = QHBoxLayout()
        self.tyWidget = QLineEdit()
        self.layout_ty.addWidget(self.tyWidget)
        self.ty_dec = QPushButton("-")
        self.ty_dec.setFixedWidth(40)
        self.ty_dec.clicked.connect(self.buttonTyDecClicked)
        self.layout_ty.addWidget(self.ty_dec)
        self.ty_inc = QPushButton("+")
        self.ty_inc.setFixedWidth(40)
        self.ty_inc.clicked.connect(self.buttonTyIncClicked)
        self.layout_ty.addWidget(self.ty_inc)
        self.layoutTranslation_table.addLayout(self.layout_ty, 1, 1)
        
        self.layout_tz = QHBoxLayout()
        self.tzWidget = QLineEdit()
        self.layout_tz.addWidget(self.tzWidget)
        self.tz_dec = QPushButton("-")
        self.tz_dec.setFixedWidth(40)
        self.tz_dec.clicked.connect(self.buttonTzDecClicked)
        self.layout_tz.addWidget(self.tz_dec)
        self.tz_inc = QPushButton("+")
        self.tz_inc.setFixedWidth(40)
        self.tz_inc.clicked.connect(self.buttonTzIncClicked)
        self.layout_tz.addWidget(self.tz_inc)
        self.layoutTranslation_table.addLayout(self.layout_tz, 2, 1)
        self.txWidget.editingFinished.connect(self.paramChanged)
        self.tyWidget.editingFinished.connect(self.paramChanged)
        self.tzWidget.editingFinished.connect(self.paramChanged)
        self.layoutTranslation.addLayout(self.layoutTranslation_table)
        self.translationWidget.setLayout(self.layoutTranslation)
        self.layout_right.addWidget(self.translationWidget)
        
        self.layout_spacer_right = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout_right.addSpacerItem(self.layout_spacer_right)
        
        self.layout.addLayout(self.layout_left)
        self.layout.addLayout(self.layout_right)

        # 메인 위젯에 레이아웃 설정
        self.setLayout(self.layout)
        
    def setData(self, paramPath):
        with open(paramPath, 'r') as f:
            param = json.load(f)
        
        self.columns = param['columns']
        
        ext_rot = np.array(param['extrinsic']['rotation'], dtype=np.float32).reshape(3, 3)
        rot = rotationMatrixToEulerAngles(ext_rot, self.currentRotationType)
        self.Rx = math.degrees(rot[0])
        self.Ry = math.degrees(rot[1])
        self.Rz = math.degrees(rot[2])
        
        self.RxWidget.setText(str(self.Rx))
        self.RyWidget.setText(str(self.Ry))
        self.RzWidget.setText(str(self.Rz))
        
        self.tx = param['extrinsic']['translation']['x']
        self.ty = param['extrinsic']['translation']['y']
        self.tz = param['extrinsic']['translation']['z']
        
        self.txWidget.setText(str(self.tx))
        self.tyWidget.setText(str(self.ty))
        self.tzWidget.setText(str(self.tz))
        
        self.objects = None
        if 'objects' in param:
            self.objects = param['objects']
        if 'object' in param:
            self.objects = param['object']
        
    def rotationTypeChanged(self):
        self.currentRotationType = self.rotationTypeList.currentText()
        rx = math.radians(float(self.RxWidget.text()))
        ry = math.radians(float(self.RyWidget.text()))
        rz = math.radians(float(self.RzWidget.text()))
        
        if rx is not None and ry is not None and rz is not None:
            if self.currentRotationType != self.prevRotationType:
                ex_rot = np.array([rx, ry, rz])
                ex_rot = eulerAnglesToRotationMatrix(ex_rot, self.prevRotationType)
                Rx_, Ry_, Rz_ = rotationMatrixToEulerAngles(ex_rot, self.currentRotationType)
                self.Rx = math.degrees(Rx_)
                self.Ry = math.degrees(Ry_)
                self.Rz = math.degrees(Rz_)
                
                self.RxWidget.setText(str(self.Rx))
                self.RyWidget.setText(str(self.Ry))
                self.RzWidget.setText(str(self.Rz))
                
        self.prevRotationType = self.currentRotationType
        
    def paramChanged(self):
        Rx = math.radians(float(self.RxWidget.text()))
        Ry = math.radians(float(self.RyWidget.text()))
        Rz = math.radians(float(self.RzWidget.text()))
        
        rot = np.array([Rx, Ry, Rz])
        rotMatrix = eulerAnglesToRotationMatrix(rot, self.currentRotationType)
        rot = rotationMatrixToEulerAngles(rotMatrix, 'ZYX')
        
        self.Rx = math.degrees(float(rot[0]))
        self.Ry = math.degrees(float(rot[1]))
        self.Rz = math.degrees(float(rot[2]))
        self.tx = float(self.txWidget.text())
        self.ty = float(self.tyWidget.text())
        self.tz = float(self.tzWidget.text())
        
        self.updateRequest()
        
    def setDataWidget(self, dataWidget):
        self.dataWidget = dataWidget
        
    def selectCameraChanged(self):
        self.updateRequest()
    
    def colorFromCameraChanged(self):
        self.updateRequest()
    
    def showObjectChanged(self):
        self.updateRequest()
        
    def updateRequest(self):
        self.parentWidget.updateRequest()
        
    def getSelectedCamera(self):
        return self.combo_selectCamera.currentText()
    
    def buttonRxDecClicked(self):
        self.Rx = float(self.RxWidget.text()) - 0.1
        self.RxWidget.setText(str(self.Rx))
        self.updateRequest()
        
    def buttonRxIncClicked(self):
        self.Rx = float(self.RxWidget.text()) + 0.1
        self.RxWidget.setText(str(self.Rx))
        self.updateRequest()
        
    def buttonRyDecClicked(self):
        self.Ry = float(self.RyWidget.text()) - 0.1
        self.RyWidget.setText(str(self.Ry))
        self.updateRequest()
        
    def buttonRyIncClicked(self):
        self.Ry = float(self.RyWidget.text()) + 0.1
        self.RyWidget.setText(str(self.Ry))
        self.updateRequest()
        
    def buttonRzDecClicked(self):
        self.Rz = float(self.RzWidget.text()) - 0.1
        self.RzWidget.setText(str(self.Rz))
        self.updateRequest()
        
    def buttonRzIncClicked(self):
        self.Rz = float(self.RzWidget.text()) + 0.1
        self.RzWidget.setText(str(self.Rz))
        self.updateRequest()
        
    def buttonTxDecClicked(self):
        self.tx = float(self.txWidget.text()) - 0.1
        self.txWidget.setText(str(self.tx))
        self.updateRequest()
        
    def buttonTxIncClicked(self):
        self.tx = float(self.txWidget.text()) + 0.1
        self.txWidget.setText(str(self.tx))
        self.updateRequest()
        
    def buttonTyDecClicked(self):
        self.ty = float(self.tyWidget.text()) - 0.1
        self.tyWidget.setText(str(self.ty))
        self.updateRequest()
        
    def buttonTyIncClicked(self):
        self.ty = float(self.tyWidget.text()) + 0.1
        self.tyWidget.setText(str(self.ty))
        self.updateRequest()
        
    def buttonTzDecClicked(self):
        self.tz = float(self.tzWidget.text()) - 0.1
        self.tzWidget.setText(str(self.tz))
        self.updateRequest()
        
    def buttonTzIncClicked(self):
        self.tz = float(self.tzWidget.text()) + 0.1
        self.tzWidget.setText(str(self.tz))
        self.updateRequest()
        

class LiDARWidget():
    def __init__(self):
        super().__init__()
        
        self.name = None
        self.dataTab = None
        self.paramTab = None
        
        self.page = None
        self.lidar = LiDARDataWidget()
        self.param = LiDARParamWidget()
        
        self.lidar.setParentWidget(self)
        self.param.setParentWidget(self)
        self.lidar.setParamWidget(self.param)
        self.param.setDataWidget(self.lidar)
        
    def setParentPage(self, page):
        self.page = page
        
    def setName(self, name):
        self.name = name
        
    def setDataTab(self, tab):
        self.dataTab = tab
        
    def setParamTab(self, tab):
        self.paramTab = tab
        
    def setCameraList(self, cameraList):
        self.param.combo_selectCamera.addItems(cameraList)
        
    def openItem(self, rootDir, subDir, itemName):
        paramFolder = os.path.join(rootDir, 'Label', subDir)
        for file in os.listdir(paramFolder):
            if itemName in file:
                paramPath = os.path.join(paramFolder, file)
                self.param.setData(paramPath)
                
        lidarFolder = os.path.join(rootDir, 'LiDAR', subDir)
        for file in os.listdir(lidarFolder):
            if itemName in file:
                itemPath = os.path.join(lidarFolder, file)
                self.lidar.setData(itemPath, self.param.columns)
    
    def updateView(self):
        self.lidar.updateView()
    
    def getLiDARData(self):
        data = {'pc': self.lidar.pc, 'colors': self.lidar.colors}
        param = {'Rx': self.param.Rx, 'Ry': self.param.Ry, 'Rz': self.param.Rz, 
                 'tx': self.param.tx, 'ty': self.param.ty, 'tz': self.param.tz}
        objects = {'objects': self.param.objects}
        return data, param, objects
    
    def setCameraData(self, data, param):
        self.lidar.cameraData = data
        self.lidar.cameraParam = param
    
    def updateRequest(self):
        camera_name = self.param.getSelectedCamera()
        self.page.updateRequestFromLiDAR(self.name, camera_name)
        
    