from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QDesktopWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie

class Waiting(QWidget):
    def __init__(self, msg="", show_gif=False, fn="waiting.gif"):
        """Waiting widget to show a waiting message and a waiting indicator.
        
        :param QWidget: QWidget base class
        :type QWidget: QtWidgets.QWidget
        :param msg: message to be shown, defaults to ""
        :type msg: str, optional
        :param show_gif: show waiting indicator, defaults to False
        :type show_gif: bool, optional
        :param fn: path to waiting indicator, defaults to "waiting.gif"
        :type fn: str, optional
        """
        super().__init__()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(0, 0, 128, 128)        
        
        self.l = QGridLayout(self)

        if show_gif:
            self.m = QMovie(fn)

            self.la = QLabel()
            self.la.setMovie(self.m)

            self.l.addWidget(self.la)
            self.m.start()

        if msg:
            t = QLabel(msg)
            t.setStyleSheet("color:white;")
            self.l.addWidget(t)

        d = QDesktopWidget().availableGeometry().center()
        self.move(d.x()-self.width()//2, d.y()-self.height()//2)
        self.show()

if __name__ == '__main__':
    app = QApplication([])

    w = Waiting("please wait, loading...", show_gif=True)

    app.exec_()