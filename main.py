import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from openhsv import OpenHSV
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
import qdarkstyle
import sys
import time

_ROOT_DIR = Path(r'D:\Libraries\OpenHSV\openhsv')

if __name__ == '__main__':
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Show splash screen for 2 seconds
    pix = QPixmap(str(_ROOT_DIR / "openhsv_splashscreen.jpg"))
    splash = QSplashScreen(pix)
    splash.show()
    time.sleep(2)
    splash.close()
    base_folder = Path(__file__).parent / "openhsv"

    # Show OpenHSV main window
    w = OpenHSV(app, base_folder=base_folder)
    w.showMaximized()
    w.updateRangeIndicator()

    sys.exit(app.exec_())
