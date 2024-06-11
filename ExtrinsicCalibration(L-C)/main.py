import sys
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *

from page.PageManual import PageManual

from func.func import *
from widget.Widgets import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Extrinsic Calibration Tool Box (LiDAR-Camera)")
        self.setGeometry(100, 100, 1200, 800)

        # 탭 위젯 생성
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 첫 번째 탭: 보정 수작업
        self.manual = PageManual()
        self.tabs.addTab(self.manual, "Manual")

# QApplication 인스턴스 생성
app = QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec())
