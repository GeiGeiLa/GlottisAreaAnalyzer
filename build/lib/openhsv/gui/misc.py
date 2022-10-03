from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QPushButton
import pyqtgraph as pg

class fullScreenPreview(QWidget):
    def __init__(self):
        """Full Screen Preview widget
        """
        super().__init__()

        # Top left
        self.l = QGridLayout(self)
        self.l.addWidget(QLabel("Camera Preview"))

        # Top right
        b = QPushButton("Close preview")
        b.clicked.connect(self.close)
        self.l.addWidget(b, 0, 9)

        # Main, span complete width
        self.im = pg.ImageView()
        self.l.addWidget(self.im, 1, 0, 1, 10)
        
    def setImage(self, im):
        """Sets image in central ImageView

        :param im: image to be shown
        :type im: numpy.ndarray
        """        
        self.im.setImage(im)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import numpy as np

    app = QApplication([])
    fs = fullScreenPreview()

    random_im = np.random.randint(0, 255, (128, 128)).astype(np.uint8)

    fs.setImage(random_im)
    fs.showFullScreen()

    app.exec_()