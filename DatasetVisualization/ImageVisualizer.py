import sys
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

class ImageVisualizer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(" ")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()
        
    def initUI(self):
        pass