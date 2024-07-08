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
import cv2
import json
import copy
import math

from func.func import *

class CameraDataWidget(pg.ImageView):
    def __init__(self):
        super().__init__()
        self.view = pg.PlotItem()
        
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.ui.histogram.hide()
        
        self.parentWidget = None
        self.paramWidget = None
        self.imagePath = None
        
        self.image = None
        
        self.LiDARData = None
        self.LiDARParam = None
    
    def setParentWidget(self, parentWidget):
        self.parentWidget = parentWidget
        
    def setData(self, imagePath):
        self.imagePath = imagePath
        self.image = cv2.imread(self.imagePath)
        
    def updateView(self):
        self.image = cv2.imread(self.imagePath)
        if self.image is None:
            return
        image = copy.deepcopy(self.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imh, imw, imc = image.shape

        Rx_L = math.radians(self.LiDARParam['Rx'])
        Ry_L = math.radians(self.LiDARParam['Ry'])
        Rz_L = math.radians(self.LiDARParam['Rz'])
        tx_L = self.LiDARParam['tx']
        ty_L = self.LiDARParam['ty']
        tz_L = self.LiDARParam['tz']
        
        rot_matrix_L = eulerAnglesToRotationMatrix(np.array([Rx_L, Ry_L, Rz_L]), 'ZYX')
        translation_L = np.array([tx_L, ty_L, tz_L]).reshape(3, 1)
        ext_lidar = np.vstack([np.hstack([rot_matrix_L, translation_L]), [0, 0, 0, 1]])
        
        fx = float(self.paramWidget.fx)
        fy = float(self.paramWidget.fy)
        cx = float(self.paramWidget.cx)
        cy = float(self.paramWidget.cy)
        
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        Rx_C = math.radians(float(self.paramWidget.Rx))
        Ry_C = math.radians(float(self.paramWidget.Ry))
        Rz_C = math.radians(float(self.paramWidget.Rz))
        tx_C = float(self.paramWidget.tx)
        ty_C = float(self.paramWidget.ty)
        tz_C = float(self.paramWidget.tz)
        
        rot_matrix_C = eulerAnglesToRotationMatrix(np.array([Rx_C, Ry_C, Rz_C]), 'ZYX')
        translation_C = np.array([tx_C, ty_C, tz_C]).reshape(3, 1)
        ext_camera = np.vstack([np.hstack([rot_matrix_C, translation_C]), [0, 0, 0, 1]])
                                
        ext_L2C = np.dot(ext_camera, np.linalg.inv(ext_lidar))
        
        # LiDAR projection
        if self.paramWidget.checkShowLiDAR.isChecked():           
            pointCloud = copy.deepcopy(self.LiDARData['pc'])
            colors = copy.deepcopy(self.LiDARData['colors'])
            
            pointCloud = np.vstack([pointCloud[:, :3].T, np.ones((1, pointCloud.shape[0]))])
            pointCloud = np.dot(ext_L2C, pointCloud)
            pointCloud = pointCloud[:3, :]
            
            # z > 0
            idx = pointCloud[2, :] > 0
            pointCloud = pointCloud[:, idx]
            if colors is not None:
                colors = colors[idx, :]
            
            projPoints = np.dot(intrinsic, pointCloud)
            projPoints = projPoints[:2, :] / projPoints[2, :]
            
            # Draw LiDAR points
            for i in range(projPoints.shape[1]):
                x = int(projPoints[0, i])
                y = int(projPoints[1, i])
                if x >= 0 and x < imw and y >= 0 and y < imh:
                    if colors is not None:
                        cv2.circle(image, (x, y), 2, colors[i, :3].tolist(), -1)
                    else:   
                        cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
            
        # Object projection
        if self.paramWidget.checkShowLiDARObject.isChecked():
            if self.paramWidget.objects_LiDAR is not None:
                for obj_LiDAR in copy.deepcopy(self.paramWidget.objects_LiDAR['objects']):
                    objClass = obj_LiDAR['class']
                    if 'box3d' in obj_LiDAR:
                        box3d = obj_LiDAR['box3d']
                        size = box3d['size']
                        
                        width = float(size['width'])
                        height = float(size['height'])
                        length = float(size['length'])
                        
                        rotation = np.array(box3d['rotation'], dtype=np.float32).reshape(3, 3)
                        tx = box3d['translation']['x']
                        ty = box3d['translation']['y']
                        tz = box3d['translation']['z']
                        translation = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
                        ext = np.vstack([np.hstack([rotation, translation]), [0, 0, 0, 1]])
                        
                        boxPoints = np.array([
                            [length/2, length/2, length/2, length/2, -length/2, -length/2, -length/2, -length/2],
                            [width/2, width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2],
                            [height/2, -height/2, -height/2, height/2, height/2, -height/2, -height/2, height/2],
                            [1, 1, 1, 1, 1, 1, 1, 1]
                        ])
                        
                        boxPoints = np.dot(ext, boxPoints)
                        boxPoints = np.dot(ext_L2C, boxPoints)
                        boxPoints = boxPoints[:3, :]
                        
                        fx = float(self.paramWidget.fx)
                        fy = float(self.paramWidget.fy)
                        cx = float(self.paramWidget.cx)
                        cy = float(self.paramWidget.cy)
                        
                        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        
                        boxPoints = np.dot(intrinsic, boxPoints)
                        boxPoints = boxPoints[:2, :] / boxPoints[2, :]
                        
                        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                        for edge in edges:
                            p1 = (int(boxPoints[0, edge[0]]), int(boxPoints[1, edge[0]]))
                            p2 = (int(boxPoints[0, edge[1]]), int(boxPoints[1, edge[1]]))
                            cv2.line(image, p1, p2, (0, 255, 0), 2)
                            
                        cv2.putText(image, objClass, (int(boxPoints[0, 0]), int(boxPoints[1, 0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
        
        # Camera Object projection
        if self.paramWidget.checkShowObject.isChecked() and self.paramWidget.objects is not None:
            for obj in copy.deepcopy(self.paramWidget.objects):
                objClass = obj['class']
                if 'box2d' in obj:
                    box2d = obj['box2d']
                    cx = float(box2d['cx']) * imw
                    cy = float(box2d['cy']) * imh
                    w = float(box2d['w']) * imw
                    h = float(box2d['h']) * imh
                    
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, objClass, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                if 'box3d' in obj:
                    box3d = obj['box3d']
                    size = box3d['size']
                    
                    width = float(size['width'])
                    height = float(size['height'])
                    length = float(size['length'])
                    
                    rotation = np.array(box3d['rotation'], dtype=np.float32).reshape(3, 3)
                    tx = box3d['translation']['x']
                    ty = box3d['translation']['y']
                    tz = box3d['translation']['z']
                    translation = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
                    ext = np.vstack([np.hstack([rotation, translation]), [0, 0, 0, 1]])
                    
                    boxPoints = np.array([
                        [length/2, length/2, length/2, length/2, -length/2, -length/2, -length/2, -length/2],
                        [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2],
                        [height/2, height/2, -height/2, -height/2, height/2, height/2, -height/2, -height/2],
                        [1, 1, 1, 1, 1, 1, 1, 1]
                    ])
                    
                    boxPoints = np.dot(ext, boxPoints)
                    boxPoints = boxPoints[:3, :]
                    
                    fx = float(self.paramWidget.fx)
                    fy = float(self.paramWidget.fy)
                    cx = float(self.paramWidget.cx)
                    cy = float(self.paramWidget.cy)
                    
                    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    
                    boxPoints = np.dot(intrinsic, boxPoints)
                    boxPoints = boxPoints[:2, :] / boxPoints[2, :]
                    
                    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                    for edge in edges:
                        p1 = (int(boxPoints[0, edge[0]]), int(boxPoints[1, edge[0]]))
                        p2 = (int(boxPoints[0, edge[1]]), int(boxPoints[1, edge[1]]))
                        cv2.line(image, p1, p2, (255, 0, 0), 2)
                        
                    cv2.putText(image, objClass, (int(boxPoints[0, 0]), int(boxPoints[1, 0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        image = np.transpose(image, (1, 0, 2))
        self.setImage(image)
        
    def setParamWidget(self, paramWidget):
        self.paramWidget = paramWidget

class CameraParamWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.Rx = None
        self.Ry = None
        self.Rz = None
        self.tx = None
        self.ty = None
        self.tz = None
        self.currentRotationType = 'ZYX'
        self.prevRotationType = 'ZYX'
        
        self.parentWidget = None
        self.dataWidget = None
        self.objects = None
        self.objects_LiDAR = None
        
    def setParentWidget(self, parentWidget):
        self.parentWidget = parentWidget
        
    def initUI(self):
        self.layout = QHBoxLayout()
        
        # Left Layout
        self.layout_left = QVBoxLayout()
        
        # Camera Intrinsic Parameters
        self.intrinsicWidget = QWidget()
        self.layout_intrinsic = QVBoxLayout()
        self.layout_intrinsic_label = QLabel("Intrinsic Parameters")
        self.layout_intrinsic.addWidget(self.layout_intrinsic_label)
        self.layout_intrinsic_table = QGridLayout()
        # 4x2 table
        self.layout_intrinsic_table.addWidget(QLabel("fx"), 0, 0)
        self.layout_intrinsic_table.addWidget(QLabel("fy"), 1, 0)
        self.layout_intrinsic_table.addWidget(QLabel("cx"), 2, 0)
        self.layout_intrinsic_table.addWidget(QLabel("cy"), 3, 0)
        
        self.layout_fx = QHBoxLayout()
        self.fxWidget = QLineEdit()
        self.layout_fx.addWidget(self.fxWidget)
        self.fx_dec = QPushButton("-")
        self.fx_dec.setFixedWidth(40)
        self.fx_dec.clicked.connect(self.buttonFxDecClicked)
        self.layout_fx.addWidget(self.fx_dec)
        self.fx_inc = QPushButton("+")
        self.fx_inc.setFixedWidth(40)
        self.fx_inc.clicked.connect(self.buttonFxIncClicked)
        self.layout_fx.addWidget(self.fx_inc)
        self.layout_intrinsic_table.addLayout(self.layout_fx, 0, 1)
        
        self.layout_fy = QHBoxLayout()
        self.fyWidget = QLineEdit()
        self.layout_fy.addWidget(self.fyWidget)
        self.fy_dec = QPushButton("-")
        self.fy_dec.setFixedWidth(40)
        self.fy_dec.clicked.connect(self.buttonFyDecClicked)
        self.layout_fy.addWidget(self.fy_dec)
        self.fy_inc = QPushButton("+")
        self.fy_inc.setFixedWidth(40)
        self.fy_inc.clicked.connect(self.buttonFyIncClicked)
        self.layout_fy.addWidget(self.fy_inc)
        self.layout_intrinsic_table.addLayout(self.layout_fy, 1, 1)
        
        self.layout_cx = QHBoxLayout()
        self.cxWidget = QLineEdit()
        self.layout_cx.addWidget(self.cxWidget)
        self.cx_dec = QPushButton("-")
        self.cx_dec.setFixedWidth(40)
        self.cx_dec.clicked.connect(self.buttonCxDecClicked)
        self.layout_cx.addWidget(self.cx_dec)
        self.cx_inc = QPushButton("+")
        self.cx_inc.setFixedWidth(40)
        self.cx_inc.clicked.connect(self.buttonCxIncClicked)
        self.layout_cx.addWidget(self.cx_inc)
        self.layout_intrinsic_table.addLayout(self.layout_cx, 2, 1)
        
        self.layout_cy = QHBoxLayout()
        self.cyWidget = QLineEdit()
        self.layout_cy.addWidget(self.cyWidget)
        self.cy_dec = QPushButton("-")
        self.cy_dec.setFixedWidth(40)
        self.cy_dec.clicked.connect(self.buttonCyDecClicked)
        self.layout_cy.addWidget(self.cy_dec)
        self.cy_inc = QPushButton("+")
        self.cy_inc.setFixedWidth(40)
        self.cy_inc.clicked.connect(self.buttonCyIncClicked)
        self.layout_cy.addWidget(self.cy_inc)
        self.layout_intrinsic_table.addLayout(self.layout_cy, 3, 1)
        self.fxWidget.textChanged.connect(self.paramChanged)
        self.fyWidget.textChanged.connect(self.paramChanged)
        self.cxWidget.textChanged.connect(self.paramChanged)
        self.cyWidget.textChanged.connect(self.paramChanged)
        self.layout_intrinsic.addLayout(self.layout_intrinsic_table)
        self.layout_spacer_intrinsic = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout_intrinsic.addSpacerItem(self.layout_spacer_intrinsic)
        self.intrinsicWidget.setLayout(self.layout_intrinsic)
        self.layout_left.addWidget(self.intrinsicWidget)
        
        # Flag(LiDAR)
        self.layout_flag = QVBoxLayout()
        self.fromLiDARLabel = QLabel("From LiDAR")
        self.layout_flag.addWidget(self.fromLiDARLabel)
        
        self.layout_combo_select_lidar = QHBoxLayout()
        self.label_combo_select_lidar = QLabel("Select LiDAR: ")
        self.layout_combo_select_lidar.addWidget(self.label_combo_select_lidar)
        self.combo_select_lidar = QComboBox()
        self.layout_combo_select_lidar.addWidget(self.combo_select_lidar)
        self.layout_flag.addLayout(self.layout_combo_select_lidar)
        self.layout_check_lidar_flag = QHBoxLayout()
        
        self.checkShowLiDAR = QCheckBox("Show LiDAR")
        self.checkShowLiDAR.setChecked(False)
        self.checkShowLiDAR.stateChanged.connect(self.checkShowLiDARChanged)
        self.layout_check_lidar_flag.addWidget(self.checkShowLiDAR)
        self.checkShowLiDARObject = QCheckBox("Show Object")
        self.checkShowLiDARObject.setChecked(False)
        self.checkShowLiDARObject.stateChanged.connect(self.checkShowLiDARObjectChanged)
        self.layout_check_lidar_flag.addWidget(self.checkShowLiDARObject)
        self.layout_flag.addLayout(self.layout_check_lidar_flag)
        
        self.layout_spacer_flag = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout_flag.addSpacerItem(self.layout_spacer_flag)
        
        # Flag(Camera)
        self.cameraLabel = QLabel("From Camera")
        self.layout_flag.addWidget(self.cameraLabel)
        
        self.checkShowObject = QCheckBox("Show Object")
        self.checkShowObject.setChecked(True)
        self.checkShowObject.stateChanged.connect(self.checkShowObjectChanged)
        self.layout_flag.addWidget(self.checkShowObject)
        self.layout_left.addLayout(self.layout_flag)
        
        self.layout_spacer_left = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout_left.addSpacerItem(self.layout_spacer_left)
        
        # Right Layout
        self.layout_right = QVBoxLayout()
        
        # Camera Extrinsic Parameters(Rotation)
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
        
        # Camera Extrinsic Parameters(Translation)
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
        #read json
        with open(paramPath, 'r') as f:
            param = json.load(f)
        
        
        
        self.fx = float(param['intrinsic']['fx'])
        self.fy = float(param['intrinsic']['fy'])
        self.cx = float(param['intrinsic']['cx'])
        self.cy = float(param['intrinsic']['cy'])
        
        #rotation decompose
        ex_rot = np.array(param['extrinsic']['rotation'], dtype=np.float32).reshape(3, 3)
        type = self.rotationTypeList.currentText()
        Rx, Ry, Rz = rotationMatrixToEulerAngles(ex_rot, type)
        Rx = float(Rx)
        Ry = float(Ry)
        Rz = float(Rz)
        
        Rx_, Ry_, Rz_ = rotationMatrixToEulerAngles(ex_rot, 'ZYX')
        self.Rx = math.degrees(float(Rx_))
        self.Ry = math.degrees(float(Ry_))
        self.Rz = math.degrees(float(Rz_))
        
        
        self.tx = float(param['extrinsic']['translation']['x'])
        self.ty = float(param['extrinsic']['translation']['y'])
        self.tz = float(param['extrinsic']['translation']['z'])
        
        self.fxWidget.setText(str(self.fx))
        self.fyWidget.setText(str(self.fy))
        self.cxWidget.setText(str(self.cx))
        self.cyWidget.setText(str(self.cy))
        self.RxWidget.setText(str(math.degrees(Rx)))
        self.RyWidget.setText(str(math.degrees(Ry)))
        self.RzWidget.setText(str(math.degrees(Rz)))
        self.txWidget.setText(str(self.tx))
        self.tyWidget.setText(str(self.ty))
        self.tzWidget.setText(str(self.tz))
        
        self.objects = None
        if 'objects' in param:
            self.objects = param['objects']
        
    def rotationTypeChanged(self):
        self.currentRotationType = self.rotationTypeList.currentText()
        rx = math.radians(float(self.RxWidget.text()))
        ry = math.radians(float(self.RyWidget.text()))
        rz = math.radians(float(self.RzWidget.text()))
        
        if rx is not None and ry is not None and rz is not None:
            if self.currentRotationType != self.prevRotationType:
                ex_rot = np.array([rx, ry, rz])
                ex_rot = eulerAnglesToRotationMatrix(ex_rot, self.prevRotationType)
                Rx, Ry, Rz = rotationMatrixToEulerAngles(ex_rot, self.currentRotationType)
                self.RxWidget.setText(str(math.degrees(Rx)))
                self.RyWidget.setText(str(math.degrees(Ry)))
                self.RzWidget.setText(str(math.degrees(Rz)))
                
                Rx_, Ry_, Rz_ = rotationMatrixToEulerAngles(ex_rot, 'ZYX')
                self.Rx = math.degrees(float(Rx_))
                self.Ry = math.degrees(float(Ry_))
                self.Rz = math.degrees(float(Rz_))

        self.prevRotationType = self.currentRotationType
            
    def paramChanged(self):
        self.fx = float(self.fxWidget.text())
        self.fy = float(self.fyWidget.text())
        self.cx = float(self.cxWidget.text())
        self.cy = float(self.cyWidget.text())
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
        
    def checkShowLiDARChanged(self):
        self.updateRequest()
        
    def checkShowObjectChanged(self):
        self.updateRequest()
        
    def checkShowLiDARObjectChanged(self):
        self.updateRequest()
        
    def updateRequest(self):
        self.parentWidget.updateRequest()
        
    def getSelectedLiDAR(self):
        return self.combo_select_lidar.currentText()
    
    def buttonFxDecClicked(self):
        self.fx = float(self.fxWidget.text()) - 1.0
        self.fxWidget.setText(str(self.fx))
        self.updateRequest()
        
    def buttonFxIncClicked(self):
        self.fx = float(self.fxWidget.text()) + 1.0
        self.fxWidget.setText(str(self.fx))
        self.updateRequest()
        
    def buttonFyDecClicked(self):
        self.fy = float(self.fyWidget.text()) - 1.0
        self.fyWidget.setText(str(self.fy))
        self.updateRequest()
        
    def buttonFyIncClicked(self):
        self.fy = float(self.fyWidget.text()) + 1.0
        self.fyWidget.setText(str(self.fy))
        self.updateRequest()
        
    def buttonCxDecClicked(self):
        self.cx = float(self.cxWidget.text()) - 1.0
        self.cxWidget.setText(str(self.cx))
        self.updateRequest()
        
    def buttonCxIncClicked(self):
        self.cx = float(self.cxWidget.text()) + 1.0
        self.cxWidget.setText(str(self.cx))
        self.updateRequest()
        
    def buttonCyDecClicked(self):
        self.cy = float(self.cyWidget.text()) - 1.0
        self.cyWidget.setText(str(self.cy))
        self.updateRequest()
        
    def buttonCyIncClicked(self):
        self.cy = float(self.cyWidget.text()) + 1.0
        self.cyWidget.setText(str(self.cy))
        self.updateRequest()
        
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
        
class CameraWidget():
    def __init__(self):
        super().__init__()
        
        self.name = None
        self.dataTab = None
        self.paramTab = None
        
        self.page = None
        self.image = CameraDataWidget()
        self.param = CameraParamWidget()
        
        self.image.setParentWidget(self)
        self.param.setParentWidget(self)
        self.image.setParamWidget(self.param)
        self.param.setDataWidget(self.image)
    
    def setParentPage(self, page):
        self.page = page
            
    def setName(self, name):
        self.name = name
        
    def setDataTab(self, tab):
        self.dataTab = tab
        
    def setParamTab(self, tab):
        self.paramTab = tab
        
    def setLiDARList(self, lidarList):
        self.param.combo_select_lidar.addItems(lidarList)
    
    def openItem(self, rootDir, subDir, itemName):
        paramFolder = os.path.join(rootDir, 'Label', subDir)
        for file in os.listdir(paramFolder):
            if itemName in file:
                paramPath = os.path.join(paramFolder, file)
                self.param.setData(paramPath)
                
        imageFolder = os.path.join(rootDir, 'Image', subDir)
        for file in os.listdir(imageFolder):
            if itemName in file:
                itemPath = os.path.join(imageFolder, file)
                self.image.setData(itemPath)
                
    def updateView(self):
        self.image.updateView()
                
    def getCameraData(self):
        data = {'image': self.image.imagePath}
        param = {'fx': self.param.fx, 'fy': self.param.fy, 'cx': self.param.cx, 'cy': self.param.cy, 
                 'Rx': self.param.Rx, 'Ry': self.param.Ry, 'Rz': self.param.Rz, 
                 'tx': self.param.tx, 'ty': self.param.ty, 'tz': self.param.tz}
        return data, param
    
    def setLiDARData(self, data, param, objects):
        self.image.LiDARData = data
        self.image.LiDARParam = param
        self.param.objects_LiDAR = objects
    
    def updateRequest(self):
        camera_name = self.name
        lidar_name = self.param.getSelectedLiDAR()
        self.page.updateRequestFromCamera(camera_name, lidar_name)