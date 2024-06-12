import sys
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

from DataLoader import DataLoader

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dataset Visualizer")
        self.setGeometry(100, 100, 300, 800)

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
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.vlayout = QVBoxLayout(central_widget)
        
        self.folder_tree = QTreeView()
        self.folder_tree.header().hide()
        self.folder_tree.clicked.connect(self.treeItemClicked)
        self.vlayout.addWidget(self.folder_tree)
        
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
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", QDir.rootPath())
        if folder_path:
            print(f"Selected folder: {folder_path}")
            
    def treeItemClicked(self, index):
        item = self.folder_tree.model().itemFromIndex(index)
        print(f"Selected item: {item.text()}")
        
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