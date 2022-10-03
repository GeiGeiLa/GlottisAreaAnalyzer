from PyQt5.QtWidgets import QApplication, QGridLayout, QPushButton, QLabel, \
    QMessageBox, QDialog, QLineEdit, QCheckBox, QComboBox, QFileDialog
from PyQt5.QtGui import QIntValidator
import qdarkstyle
import json

class Settings(QDialog):
    def __init__(self, exposure, fps, audioSamplingRate, audioBlockSize, audioBufferSize, save_raw, base_folder):
        """Define settings for OpenHSV operation, especially camera, audio and save settings."""
        super().__init__()
        self.l = QGridLayout(self)
        # Ensure only numbers are entered in fields
        self.onlyInt = QIntValidator()

        # Exposure time
        self.l.addWidget(QLabel("Exposure time [us], e.g. 245 for 4000 fps"))
        self.exposureTime = QLineEdit(str(exposure))
        self.exposureTime.setValidator(self.onlyInt)
        self.l.addWidget(self.exposureTime)

        # Frames per second
        self.l.addWidget(QLabel("Frames per second, e.g. 4000 for full frame"))
        self.fps = QLineEdit(str(fps))
        self.fps.setValidator(self.onlyInt)
        self.l.addWidget(self.fps)

        # Audio sampling rate
        self.l.addWidget(QLabel("Audio Sampling Rate, e.g. 96000 for 4000 camera fps"))
        self.audioSamplingRate = QLineEdit(str(audioSamplingRate))
        self.audioSamplingRate.setValidator(self.onlyInt)
        self.l.addWidget(self.audioSamplingRate)

        # Audio block size
        self.l.addWidget(QLabel("Audio Block Size, e.g. 4800"))
        self.audioBlockSize = QLineEdit(str(audioBlockSize))
        self.audioBlockSize.setValidator(self.onlyInt)
        self.l.addWidget(self.audioBlockSize)

        # Audio buffer size
        self.l.addWidget(QLabel("Audio Buffer Size, e.g. 8"))
        self.audioBufferSize = QLineEdit(str(audioBufferSize))
        self.audioBufferSize.setValidator(self.onlyInt)
        self.l.addWidget(self.audioBufferSize)

        # Saving options
        self.l.addWidget(QLabel("Saving options"))

        # Raw files
        self.save_raw = QCheckBox("Save also raw data")
        self.save_raw.setChecked(save_raw)
        self.l.addWidget(self.save_raw)

        # Base folder for saving data
        self.l.addWidget(QLabel("Base folder:"))
        self.base_folder = QLabel(base_folder)
        self.select_base_folder = QPushButton("Select other base folder")
        self.select_base_folder.clicked.connect(self.selectBaseFolder)
        self.l.addWidget(self.base_folder)
        self.l.addWidget(self.select_base_folder)

        # Save and close
        b = QPushButton("Save settings")
        b.clicked.connect(self.saveAndClose)
        self.l.addWidget(b)

        self.setWindowTitle("Settings")

    def selectBaseFolder(self):
        folder = QFileDialog.getExistingDirectory(caption="Select base folder")

        if folder:
            self.base_folder.setText(folder)

    def get(self):
        return dict(exposureTime=int(self.exposureTime.text()),
                    videoSamplingRate=int(self.fps.text()),
                    audioSamplingRate=int(self.audioSamplingRate.text()),
                    audioBlockSize=int(self.audioBlockSize.text()),
                    audioBufferSize=int(self.audioBufferSize.text()),
                    baseFolder=self.base_folder.text(),
                    saveRaw=self.save_raw.isChecked())

    def saveAndClose(self):
        try:
            with open("settings.json", "w+") as fp:
                json.dump(self.get(), fp, indent=4)

            QMessageBox.information(self, 
                "Settings saved",
                "Settings were successfully saved to settings.json")

            self.close()
        
        except Exception as e:
            QMessageBox.critical(self, 
                "Settings could not be saved.",
                "An error occured during saving: \n\n{}".format(e))

if __name__ == '__main__':
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    s = Settings(245, # Camera exposure in us
                4000, # Camera framerate
                96000, # Audio sampling rate in Hz
                4800,  # Audio block size in datapoints
                8,     # Audio block length (x * block size)
                True,  # Save raw files as lossless h264 file
                "C:/openhsv") # base folder for saving data
    s.show()

    app.exec_()