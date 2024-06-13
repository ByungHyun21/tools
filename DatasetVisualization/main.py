import sys
import os
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

from DataLoader import DataLoader

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Dataset Visualizer")
        
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        self.setGeometry(screen_width*0.8, screen_height * 0.1, screen_width*0.2, screen_height * 0.8)

        # 메뉴바 생성
        self.menuBar = self.menuBar()

        # 'File' 메뉴 추가
        fileMenu = self.menuBar.addMenu('File')

        # 'Open Folder' 액션 추가
        openFolderAction = QAction('Open Folder', self)
        openFolderAction.triggered.connect(self.open_folder)
        fileMenu.addAction(openFolderAction)
        
        self.initUI()
        
        self.dataloader = DataLoader()
        
        self.root_path = QDir.rootPath()
        self.index = 0
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.vlayout = QVBoxLayout(central_widget)
        
        self.folder_tree = QTreeView()
        self.folder_tree.header().hide()
        self.folder_tree.clicked.connect(self.treeItemClicked)
        self.vlayout.addWidget(self.folder_tree)
        
        self.folder_tree_model = QStandardItemModel()
        
        self.btnlayout = QHBoxLayout()
        self.vlayout.addLayout(self.btnlayout)
        
        self.btn_prev = QPushButton("<")
        self.btn_prev.clicked.connect(self.btnPrevClicked)
        self.btnlayout.addWidget(self.btn_prev)
        
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.btnPlayClicked)
        self.btnlayout.addWidget(self.btn_play)
        
        self.btn_next = QPushButton(">")
        self.btn_next.clicked.connect(self.btnNextClicked)
        self.btnlayout.addWidget(self.btn_next)

    def open_folder(self):
        self.folder_tree_model.clear()
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", self.root_path)
        if folder_path:
            self.root_path = folder_path
            self.dataloader.setDataset(folder_path)
            
            config = {}
            config['image_exist'] = False
            config['lidar_exist'] = False
            
            
            
            items = {}
            
            sensor_list = os.listdir(folder_path)
            for sensor in sensor_list:
                sensor_path = os.path.join(folder_path, sensor)
                if os.path.exists(os.path.join(sensor_path, 'Image')):
                    sub_dir_list = os.listdir(os.path.join(sensor_path, 'Image'))
                    sensor_type = 'Image'
                if os.path.exists(os.path.join(sensor_path, 'LiDAR')):
                    sub_dir_list = os.listdir(os.path.join(sensor_path, 'LiDAR'))
                    sensor_type = 'LiDAR'
                
                for sub_dir in sub_dir_list:
                    if sub_dir not in items:
                        items[sub_dir] = []
                    for sensor in sensor_list:
                        data_list = os.listdir(os.path.join(sensor_path, sensor_type, sub_dir))
                        for data in data_list:
                            if data.split('.')[0] not in items[sub_dir]:
                                items[sub_dir].append(data.split('.')[0])
                
            for key, value in items.items():
                parent = QStandardItem(key)
                parent.setFlags(parent.flags() & ~Qt.ItemIsEditable)
                for val in value:
                    child = QStandardItem(val)
                    child.setFlags(child.flags() & ~Qt.ItemIsEditable)
                    parent.appendRow(child)
                self.folder_tree_model.appendRow(parent)
                    
            self.folder_tree.setModel(self.folder_tree_model)

    def treeItemClicked(self, index):
        self.index = index
        item = self.folder_tree_model.itemFromIndex(index)
        
        if item.parent().text() is not None:
            self.dataloader.itemSelected(item.parent().text(), item.text())
        
    def btnPrevClicked(self):
        print("Previous button clicked")
        
    def btnNextClicked(self):
        print("Next button clicked")
    
    def btnPlayClicked(self):
        print("Play button clicked")
        

# QApplication 인스턴스 생성
app = QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec())