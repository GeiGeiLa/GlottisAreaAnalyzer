# Libraries used
from email.mime import base
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QSlider, QPushButton, QProgressBar, QLabel, \
    QMessageBox, QSplashScreen, QSizePolicy, QDialog, QLineEdit, QDateEdit, QCheckBox, QComboBox, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QColor, QPen, QIntValidator, QFont
from PyQt5.QtCore import Qt, QTimer
import qdarkstyle
import pyqtgraph as pg
import sys
import numpy as np
import imageio as io
import flammkuchen as fl  # Saving HDF5 data
import time
import sounddevice as sd  # Recording audio
import queue  # Saving audio in memory
from threading import Timer
from glob import glob
from os.path import isdir
import os
from datetime import datetime
from scipy.io.wavfile import read
from pathlib import Path

# Own scripts
from openhsv.analysis.nn import Analysis
from openhsv.hardware.camera import DummyCamera as Camera

# Import additional windows
from openhsv.gui.settings import Settings
from openhsv.gui.patient import Patient
from openhsv.gui.waiting import Waiting
from openhsv.gui.misc import fullScreenPreview
from openhsv.gui.table import Table
from openhsv.gui.db import DB
import logging


class OpenHSV (QWidget):
    """OpenHSV is the main class for recording high-speed videoendoscopy footage
    and audio. It interacts with the audio interface and the camera, performs 
    deep neural network based analysis and saves the data.

    :param app: To init OpenHSV, you only need to pass the QApplication instance
    :type app: QtWidgets.QApplication
    :param base_folder: Location where data is stored
    :type base_folder: str, optional
    :param verbose: Prints additional information to the Python console, defaults to False.
    :type verbose: boolean, optional

    """
    def __init__(self, app, base_folder: Path, verbose=False):
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        self.camera = None
        self.recorder = None
        self.grabbing = False
        self.app = app
        self.play = False
        self.audioBlockSize = 5000
        self.audioSamplingRate = 50000
        self.buffer_id = 0
        self.audioBufferSize = 5
        self.audioBuffer = [0]*self.audioBufferSize
        self.hann = np.hanning(self.audioBlockSize*self.audioBufferSize)

        # Debug mode
        self.verbose = verbose

        # Save audio data in a queue
        self.audioQueue = queue.Queue()

        # Not currently used...
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrame)

        self.abortSaving = False
        self.save_raw = True

        self.base_folder = base_folder
        self.triggerFrameIndex = 0
        self.cur_frame = 0

        # No analysis has been performed here...
        self.analysis = None

        self.setWindowTitle("openHSV v.0.6")
        self.setGeometry(100, 100, 800, 800)

        # Setup layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)


        _, self.fake_reference = read(str(self.base_folder / "examples/audio.wav"))

        # F0
        self.f0_item = pg.TextItem("xxx Hz")
        self.f0_item.setFont(QFont("Arial", 15))
        self.F0_timer = QTimer()
        self.F0_timer.timeout.connect(self.F0)

        # Imaging data
        self.imagingData = []

        # Add Widgets to Layout
        icon = QLabel()
        icon_pix = QPixmap(str(self.base_folder / "openhsv_logo.png"))
        icon.setFixedHeight(40)
        icon.setPixmap(icon_pix.scaled(178, 40, Qt.KeepAspectRatio))
        self.layout.addWidget(icon, 0, 0, 1, 1)

        self.b5 = QPushButton("尋找患者")
        self.b5.clicked.connect(self.findpatient)
        self.b5.setFixedWidth(150)
        self.layout.addWidget(self.b5, 1, 1, 1, 1, Qt.AlignRight)

        self.bInitCamera = QPushButton("選擇影片")
        self.bInitCamera.clicked.connect(self.changeVideo)
        self.layout.addWidget(self.bInitCamera, 4, 1, 2, 1)

        self.start_slider = QSlider(orientation=Qt.Horizontal)
        self.start_slider.setMinimum(1)
        self.start_slider.setMaximum(4000)
        self.start_slider.setAutoFillBackground(False)
        self.start_slider.setStyleSheet(
            "QSlider::groove:horizontal, QSlider::groove:horizontal:hover, QSlider::sub-page:horizontal, QSlider::groove:horizontal:disabled { border:0;  background: #19232D; }")
        self.start_slider.valueChanged.connect(self._getFrameStart)
        self.start_slider.sliderReleased.connect(self._checkBordersStart)

        self.end_slider = QSlider(orientation=Qt.Horizontal)
        self.end_slider.setMinimum(1)
        self.end_slider.setMaximum(4000)
        self.end_slider.setValue(4000)
        self.end_slider.setAutoFillBackground(False)
        self.end_slider.setStyleSheet("QSlider::groove:horizontal, QSlider::groove:horizontal:hover, QSlider::sub-page:horizontal, QSlider::groove:horizontal:disabled { border:0;  background: #19232D; }")
        self.end_slider.valueChanged.connect(self._getFrameEnd)
        self.end_slider.sliderReleased.connect(self._checkBordersEnd)


        self.layout.addWidget(self.start_slider, 20, 0, 1, 2)


        self.rangeIndicator = QLabel()
        self.rangeIndicator.setFixedHeight(25)
        self.rangeIndicator.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)


        self.layout.addWidget(self.rangeIndicator, 21, 0, 1, 2)
        # self.l.addWidget(QLabel("End"))
        self.layout.addWidget(self.end_slider, 22, 0, 1, 2)

        self.startAnalysisButton = QPushButton("開始分析 - Glottis Segmentation")
        self.startAnalysisButton.setEnabled(False)
        self.startAnalysisButton.clicked.connect(self.analyze)

        self.progess = QProgressBar()
        self.progess.setMinimum(0)
        self.progess.setMaximum(100)
        self.progess.setEnabled(False)

        self.layout.addWidget(self.startAnalysisButton, 11, 1)
        self.layout.addWidget(self.progess, 12, 1)

        self.saveButton = QPushButton("儲存資料.")
        self.saveButton.clicked.connect(self.save)
        self.saveButton.setEnabled(False)
        self.layout.addWidget(self.saveButton, 13, 1)

        # Play/Stop preview of data, doesn't work well
        self.playStopButton = QPushButton("播放/停止")
        self.playStopButton.clicked.connect(self.playStop)
        self.playStopButton.setEnabled(False)
        self.layout.addWidget(self.playStopButton, 15, 1)
        self.updateRangeIndicator()
        self.initSettings(baseFolder=self.base_folder)
        self.doAutomaticTasksAfterInit()

    def doAutomaticTasksAfterInit(self):
        pass

    def findpatient(self):
        """Opens a window to select patient from database.
        """
        self.db = DB(str(self.base_folder))
        self.db.show()


    def showMaximized(self):
        """Shows the window maximized and updates the range indicator
        """
        super().showMaximized()

        # Updates the range indicator,
        # fix for changing window width
        self.delayedUpdate = Timer(.1, self.updateRangeIndicator)
        self.delayedUpdate.start()

    def initSettings(self,
                     exposureTime=245,
                     videoSamplingRate=4000,
                     audioSamplingRate=80000,
                     audioBlockSize=4000,
                     audioBufferSize=3,
                     baseFolder='',
                     saveRaw=True):
        """Initializes camera, audio and saving settings
        
        :param exposureTime: camera exposure time in us, defaults to 245
        :type exposureTime: int, optional
        :param videoSamplingRate: frames per second, defaults to 4000
        :type videoSamplingRate: int, optional
        :param audioSamplingRate: audio sampling rate in Hz, defaults to 80000
        :type audioSamplingRate: int, optional
        :param audioBlockSize: audio block size transmitted from interface, defaults to 4000
        :type audioBlockSize: int, optional
        :param audioBufferSize: audio buffer size (multiples of block size), defaults to 3
        :type audioBufferSize: int, optional
        :param baseFolder: base folder for data saving, defaults to ''
        :type baseFolder: str, optional
        :param saveRaw: if raw video data should be saved as lossless compressed mp4, defaults to True
        :type saveRaw: bool, optional
        """
        self.exposureTime = exposureTime
        self.videoSamplingRate = videoSamplingRate
        self.audioSamplingRate = audioSamplingRate
        self.audioBlockSize = audioBlockSize
        self.audioBufferSize = audioBufferSize
        self.audioBuffer = [0] * self.audioBufferSize

        # Create hanning window for better FFT ability of our signal
        self.hann = np.hanning(self.audioBlockSize * self.audioBufferSize)


        el = 6
        ticks = np.linspace(0, audioBlockSize, el)[::-1]

        # Other settings
        if baseFolder:
            self.base_folder = baseFolder
        
        self.save_raw = saveRaw

    def patient(self):
        """Opens interface for patient information
        """
        p = Patient(str(self.base_folder))
        p.exec_()

        self.patientInformation = p.get()

    def updateRangeIndicator(self):
        """updates the range indicator that shows the subselection of the video
        for download or analysis
        """
        # Get indicator width and height
        w, h = self.rangeIndicator.width(), self.rangeIndicator.height()

        # Create an empty image to draw on
        pix = QPixmap(w, h)
        p = QPainter(pix)

        # Draw background color
        p.setBrush(QBrush(QColor("#19232D")))
        p.drawRect(-1, -1, w, h)

        # Draw foreground color
        p.setBrush(QBrush(QColor("#1464A0")))

        # Define region start and end on x axis
        x0 = w*self.start_slider.value()/self.start_slider.maximum()
        x1 = w*self.end_slider.value()/self.end_slider.maximum()

        # Draw rect from start to end of selection
        p.drawRect(int(x0), 0, int(x1-x0), h)

        # Display the number of frames selected
        p.setPen(QPen(QColor("#FFFFFF")))
        p.drawText(0, # left
            17, # center
            "{} frames selected".format(self.end_slider.value()-self.start_slider.value()+1))
        p.end()

        # Show the image
        self.rangeIndicator.setPixmap(pix)

    def _crop(self, im):
        """crops image to ROI on the preview window
        
        :param im: camera/video frame
        :type im: numpy.ndarray
        :return: cropped video frame
        :rtype: numpy.ndarray
        """
        # Get ROI position and size
        pos = self.roi.pos()
        size = self.roi.size()

        # Crop image to ROI selection
        return im[int(pos[1]):int(pos[1]+size[1]), 
                  int(pos[0]):int(pos[0]+size[0])]

    def changeVideo(self, force_init=False):
        """Initializes camera connection. Open camera, do basic configuration
        and set advanced settings, such as exposure time and video sampling rate (fps). 
        If camera connection could be established, enable further buttons.
        
        :param force_init: forces (re-)initialization of camera, defaults to False
        :type force_init: bool, optional
        """
        inputVideoPath = Path(QFileDialog.getOpenFileName(
            parent=self, caption='Open file', filter="Video files (*.mp4)")[0])
        self.camera = Camera(video_path=inputVideoPath)
        success = self.camera.openCamera()
        if success:
            self.end_slider.setMaximum(
                len(self.camera.videoFrames))
            self.end_slider.setValue(
                len(self.camera.videoFrames))
            theshape = np.shape(self.camera.videoFrames[0])
            videoWidth = theshape[1]
            videoHeight = theshape[0]
            logging.info(theshape)
            self.roi = pg.RectROI([0, 0], [videoWidth, videoHeight])
            # Basic configuration, such as camera gain
            self.camera.configCam()
            # Specific settings, such as exposure time and sampling rate
            self.camera.setSettings(self.exposureTime, self.videoSamplingRate, roi=(videoWidth, videoHeight))
            self.saveButton.setEnabled(True)
            self.startAnalysisButton.setEnabled(True)
            self.playStopButton.setEnabled(True)
            self.analyze()
            return True
        QMessageBox.critical(self,
            "No camera found",
            "No high-speed camera was found. Please check connection.")   
        return False

    def playStop(self):
        if self.play:
            self.timer.stop()
            self.play = False
            self.start_slider.setEnabled(True)
            self.end_slider.setEnabled(True)

        else:
            self.start_slider.setEnabled(False)
            self.end_slider.setEnabled(False)
            # 30 fps --> 34 ms per frame
            self.timer.start(34)
            self.play = True

    def setImage(self, im, restore_view=True, restore_levels=False):
        """Shows image in the camera preview window. It further can restore the previous
        view, i.e. zoom and location, as well as the levels (contrast, brightness) of the
        previous image. Currently, the restoring level feature is by default deactivated,
        because in the examination procedure it is quite common that there is no signal
        (away from patient) or oversaturated (very close to the mouth/tongue).
        
        :param im: image to be shown in preview
        :type im: numpy.ndarray
        :param restore_view: if view should be restored from previous image, defaults to True
        :type restore_view: bool, optional
        :param restore_levels: if contrast/brightness should be restored from previous image, defaults to False
        :type restore_levels: bool, optional
        """
        # Save view from current image
        state, levels = self.im.getView().getState(), self.im.getImageItem().levels

        # Set new image
        self.im.setImage(im)

        # Restore view
        if restore_view:
            self.im.getView().setState(state)

        # Restore levels
        if restore_levels:
            self.im.getImageItem().setLevels(levels)

    def nextFrame(self):
        self.cur_frame += 1

        if self.cur_frame >= self.end_slider.value():
            self.cur_frame = self.start_slider.value()

        im = self.camera.getMemoryFrame(self.cur_frame, by_trigger=True)
        self.setImage(im.transpose((1, 0, 2)))

    def initAudio(self):
        """initialize audio recorder and empties the audio queue and data list. 
        It selects the first audio interface found (in OpenHSV the Focusrite Scarlet 2i2),
        selects both channels (by default, channel 1 is the camera reference signal and
        channel 2 the actual audio signal). Every audio data block is passed to the callback function.
        The callback works already on a separate thread, no need to move it to a different one.
        It also immmediately starts the recorder.
        """
        if self.recorder is None:
            self.audioQueue = queue.Queue()
            self.audioData = []

            # Create a new recorder input stream object.
            # Select first device, which is typically the desired audio interface.
            self.recorder = sd.InputStream(samplerate=self.audioSamplingRate, 
                device=1, 
                channels=2, 
                callback=self._audioCallback, 
                blocksize=self.audioBlockSize)

            self.recorder.start()

    def _audioCallback(self, data, *args):
        """Audio callback that retrieves the data from the audio signal.
        
        :param data: audio data from audio interface
        :type data: numpy.ndarray
        """
        # Fake reference signal when DummyCamera is used
        if self.camera.__class__.__name__ == 'DummyCamera':
            self.audioCurve1.setData(self.fake_reference[:len(data[:,0]),0]+1)
        # If real camera is used, use real audio signal
        else:
            self.audioCurve1.setData(data[:, 0]+1)

        self.audioCurve2.setData(data[:, 1]-1)
        self.audioQueue.put(data.copy())

    def stopAudio(self):
        """Stops audio recording and saves data from queue to internal memory
        """
        if self.recorder:
            self.recorder.stop()

        # While there is data in the queue,
        # save data to memory and calculate F0.
        while not self.audioQueue.empty():
            self.F0()



        self.recorder = None

    def _showF0(self, f0=None):
        """shows fundamental frequency in audio widget
        
        :param f0: fundamental frequency, defaults to None
        :type f0: float, optional
        """
        if f0 is None:
            f0_text = "xxx"

        else:
            f0_text = "{:.1f}".format(f0)

        self.f0_item.setText("F0: {} Hz".format(f0_text))

    def F0(self, channel_for_F0=1, intensity_threshold=5):
        """Calculates fundamental frequency from audio signal.
        It further saves the audio data to internal memory.
        
        :param channel_for_F0: 
            selected audio channel for F0 calculation. In our setting, 
            channel 0 is for the reference signal, 
            channel 1 for the audio signal, defaults to 1
        :type channel_for_F0: int, optional
        :param intensity_threshold: intensity threshold for calculating F0, defaults to 5
        :type intensity_threshold: int, optional
        """
        if self.audioQueue.empty():
            return

        # Get data from audio queue
        data = self.audioQueue.get()
        self.audioData.append(data)
        d = data[:, channel_for_F0]

        # Rolling audio buffer
        self.audioBuffer = self.audioBuffer[1:]
        self.audioBuffer.append(d)
        buffer = np.hstack(self.audioBuffer)

        # Compute F0
        # Get power spectrum of buffer with hanning window
        if self.hann.size == buffer.size:
            # FFT from hanning window modulated signal, only positive frequencies
            f = np.abs(np.fft.fft(buffer * self.hann))[:buffer.size//2]

            # Get associated frequencies
            freq = np.fft.fftfreq(buffer.size, 1/self.audioSamplingRate)[:buffer.size//2]

            # Only provide F0, if above a certain amplitude threshold, 
            # to avoid computing F0 from noise
            if f.max() > intensity_threshold:
                # Assume that fundamental frequency is also dominant frequency
                self._showF0(freq[np.argmax(f)])

            else:
                self._showF0()

    def _checkBordersStart(self):
        """Check if start slider is at valid position
        """
        v = self.start_slider.value()

        if v >= self.end_slider.value():
            self.start_slider.setValue(self.end_slider.value()-1)

    def _checkBordersEnd(self):
        """Check if end slider is at valid position
        """
        v = self.end_slider.value()

        if v <= self.start_slider.value():
            self.end_slider.setValue(self.start_slider.value()+1)

    def _getFrameStart(self):
        """Get frame from camera to preview, based on start slider
        """
        self._getFrame(0)

    def _getFrameEnd(self):
        """Get frame from camera to preview, based on end slider
        """
        self._getFrame(1)

    def _getFrame(self, slider=0):
        """Downloads frame from camera at given slider position and
        previews it.
        
        :param slider: slider selector (0: start, 1: end), defaults to 0
        :type slider: int, optional
        """
        if self.camera is not None:
            if slider == 0:
                sl = self.start_slider
            else:
                sl = self.end_slider

            try:
                new_frame = int(sl.value()-1)

                if self.cur_frame != new_frame:
                    self.updateRangeIndicator()

                    self.cur_frame = new_frame
                    im = self.camera.getMemoryFrame(self.cur_frame, by_trigger=True)
                    self.setImage(im.transpose((1, 0, 2)))
            except:
                pass

    def analyze(self):
        """Analyzes the selected range of video data. The selected frames will
        be downloaded from the camera and subsequently processed, i.e. segmented
        by the neural network.
        """
        start, end = self.start_slider.value(), self.end_slider.value()

        logging.info("Analyzing data from frame %s to %s", start, end)

        if start == end:
            return

        elif end-start < 5:
            return

        elif self.analysis is not None:
            k = QMessageBox.question(self,
                    "Delete analysis?",
                    "We found an analysis already, do you want to delete this analysis and start over again?")

            if k == QMessageBox.No:
                return

            else:
                self.analysis = None
                self.imagingData = []

        w = Waiting("please wait, loading...", show_gif=False)
        w.activateWindow()
        self.app.processEvents()

        self.progess.setEnabled(True)
        
        # Open Analysis window and raise to front
        self.analysis = Analysis()
        self.analysis.show()
        self.analysis.raise_()
        self.analysis.activateWindow()

        w.close()

        ims = []

        # Analyze data frame by frame
        for i, frame_index in enumerate(range(start-1, end)):
            # Get frame from camera
            im = self.camera.getMemoryFrame(frame_index, by_trigger=True)

            # Raw endoscopy image
            self.imagingData.append(im)

            # Crop to ROI
            im_crop = self._crop(im)
            # Normalize image
            im_crop = (im_crop - im_crop.min()) / (im_crop.max() - im_crop.min()) * 2 - 1

            # Segment cropped image
            self.analysis.segment(im_crop)

            # Show progress on progress bar
            self.progess.setValue(int(np.ceil(i / (end - start + 1e-5) * 100)))
            self.app.processEvents()

            # Analyze data, as long as window is still open
            if not self.analysis.isVisible():
                self.progess.setValue(0)
                self.progess.setEnabled(False)
                break


        # Show parameters
        params = self.analysis.computeParameters(dt_audio=1/self.audioSamplingRate,
            dt_video=1/self.videoSamplingRate)

        # Save metadata
        self.analysis = self.analysis.get()
        self.analysis['start_frame'] = start
        self.analysis['end_frame'] = end
        self.analysis['roi_pos'] = [int(i) for i in self.roi.pos()]
        self.analysis['roi_size'] = [int(i) for i in self.roi.size()]
        self.analysis['parameters'] = params        
        

    def save(self, save_last_seconds=4):
        """Saves the recorded and selected data. In particular, we save
        the metadata, including audio, video and patient metadata, audio
        data together with camera reference signal and video data.

        :param save_last_seconds: 
            the last seconds from recording end to be saved. 
            We record one second after the `stop`-trigger, and we 
            usually record one second of footage, thus, we need at least two
            seconds to ensure saving all relevant audio data. To adjust for some
            uncertainties, we recommend recording a few more seconds. Defaults to 4.
        :type save_last_seconds: int

        .. note::

            Saving the data in an appropriate format is not trivial. We both
            need to consider portability, cross-functionality and quality.
            Therefore, we save metadata as structured JSON file format, a common
            file format that can be opened and viewed with any text editor,
            but easily processed by a variety of data analysis software.
        
            Further, audio data is saved as common wav files, as well as packed as
            HDF5 file. HDF5 is a very common container format that allows storing
            of complex data in a very efficient and convenient way. 

            Video data, however, is saved as ``mp4`` file format, as this is 
            highly portable and can be viewed with a common video viewers. The h264 codec
            also allows saving the video data in a lossless file format, needed for accurate
            data analysis while keeping the file size at a reasonable level and still
            ensure the ability to preview the video.

            If there's any segmentation already available, the segmentation maps
            are stored as well in HDF5 file format as binary maps.
        """
        if self.saveButton.text() == "Abort saving":
            self.abortSaving = True
            self.saveButton.setText("Save data.")
            self.progess.setValue(0)
            self.progess.setEnabled(False)

            QMessageBox.information(self, 
                "Saving aborted.", 
                "User aborted saving process. Please start again.")

            return

        self.abortSaving = False
        self.progess.setEnabled(True)

        # Get start and end range from video
        start, end = self.start_slider.value(), self.end_slider.value()

        if self.verbose:
            print("Saving data from frame {} to {}".format(start, end))

        # No frames selected...
        if start == end:
            return

        # Select at least 5 frames!
        elif end - start < 5:
            return

        # No analysis found, maybe examiner wants to perform some analysis
        if self.analysis is None:
            ok = QMessageBox.question(self,
                    "No analysis found.",
                    "No analysis has been found. \nDo you want to save the data anyway?")

            if ok == QMessageBox.No:
                return

        # Analysis performed, 
        # but the video does not match the analysis length
        if self.analysis is not None:
            if len(self.imagingData) != (end-start+1):
                ok = QMessageBox.question(self, "Stored video",
                        "Stored video from analysis does not match the expected length.\n"+ \
                        "Should I keep the analysis (yes) or delete the analysis and download the video? (no)")

                if ok == QMessageBox.No:
                    self.analysis = None
                    self.imagingData = []

        from datetime import datetime
        import os
        import json
        from scipy.io.wavfile import write as wavwrite

        self.patient()

        # Saving settings
        folder_name = str(self.base_folder)+"\\"+self.patientInformation['folder']+"\\"+\
                      datetime.now().strftime("%Y%m%d")+"_"+self.patientInformation['last_name']

        # Create folder if it does not exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Create filename base
        now = datetime.now()
        fn_base = folder_name+"\\"+now.strftime("%Y%m%d_%H%M%S")+"_" + \
                  self.patientInformation['last_name']+"_"+self.patientInformation['first_name']

        saved = []

        if self.verbose:
            print(fn_base)
            sys.stdout.flush()

        if not len(self.imagingData):
            # Get all selected frames from camera
            ims = []

            self.saveButton.setText("Abort saving")

            for i, frame_index in enumerate(range(start-1, end)):
                im = self.camera.getMemoryFrame(frame_index, by_trigger=True)
                ims.append(im)

                # Show progress up to 90% for downloading the images
                self.progess.setValue(int(np.ceil(i / (end - start + 1e-5) * 90)))
                self.app.processEvents()

                if self.abortSaving:
                    return

        # We already segmented everything from the camera...
        else:
            ims = self.imagingData

        if self.verbose:
            print("Total images: ", len(ims))

        try:
            # Save movie as compressed mp4 (h264 codec)
            io.mimsave(fn_base+".mp4", ims)
            saved.append("Movie [mp4]")
        except Exception as e:
            sys.stderr.write("Movie could not be saved.\n{}".format(e))
        
        # Show progress at 92% for saving the video
        self.progess.setValue(92)
        self.app.processEvents()

        # Prepare metadata
        meta = {'Audio':
                    {
                        'Sample Rate [Hz]': self.audioSamplingRate
                    },
                'Video':
                    {
                        'Frames per second [Hz]': self.videoSamplingRate,
                        'Exposure time [us]': self.exposureTime,
                        'Start frame': self.start_slider.value(),
                        'End frame': self.end_slider.value(),
                        'Total frames recorded': self.camera.frames_to_record,
                        'Frames before trigger': self.camera.frames_before_trigger
                    },
                'Date': now.strftime("%Y-%m-%d %H:%M:%S"),
                'Patient': self.patientInformation
        }

        if self.analysis is not None:
            # Save analysis meta data (frame range, ROI position and size)
            meta['Analysis'] = {
                'Start frame': self.analysis['start_frame'],
                'End frame': self.analysis['end_frame'],
                'ROI position': self.analysis['roi_pos'],
                'ROI size': self.analysis['roi_size']
            }

            try:
                # Save binary segmentation maps as hdf5 file
                fl.save(fn_base+".segmentation", 
                    dict(segmentation=self.analysis['segmentation']), 
                    compression=('blosc', 4))

                saved.append("Segmentation [hdf5]")
            except Exception as e:
                sys.stderr.write("Segmentation could not be saved.\n{}".format(e))

        # Write metadata to pretty printed json file
        try:
            with open(fn_base+".meta", "w+") as fp:
                json.dump(meta, fp, indent=4)
            saved.append("Metadata [json]")

        except Exception as e:
            sys.stderr.write("Metadata could not be saved.\n{}".format(e))

        # Show progress at 95% for saving the metadata
        self.progess.setValue(95)
        self.app.processEvents()

        if len(self.audioData):
            # Save recorded audio 
            audio = np.vstack(self.audioData)
        
            # Crop to save_last_seconds if audio data is longer
            if len(audio) >= self.audioSamplingRate*save_last_seconds:
                audio = audio[-self.audioSamplingRate*save_last_seconds:]

            # Save WAV
            try:
                # Save audio also as wav file - for the time being
                wavwrite(fn_base +".wav", self.audioSamplingRate, audio)
                saved.append("Audio [wav]")

            except Exception as e:
                sys.stderr.write("Audio could not be saved as wav. \n{}".format(e))

            # Save HDF5
            try:
                # Save audio as compressed hdf5 file
                fl.save(fn_base+".audio", dict(audio=audio), compression=("blosc", 5))
                saved.append("Audio [hdf5]")

            except Exception as e:
                sys.stderr.write("Audio could not be saved as hdf5.\n{}".format(e))

        # Show progress at 97% for saving the audio data
        self.progess.setValue(97)
        self.app.processEvents()

        if self.save_raw:
            # Save as lossless compressed mp4 file using h264 codec
            try:
                io.mimwrite(fn_base+"_lossless.mp4",
                            ims,
                            codec='libx264rgb',
                            pixelformat='rgb24',
                            output_params=['-crf', '0',
                                        '-preset', 'ultrafast'])

                saved.append("Movie (lossless) [mp4]")
            except Exception as e:
                sys.stderr.write("Lossless compressed movie could not be saved.\n{}".format(e))

        # Show progress at 100% for saving the video lossless
        self.progess.setValue(100)
        self.app.processEvents()

        if self.verbose:
            print("Done with saving!")

        # Reset buttons
        self.saveButton.setText("Save data.")
        self.progess.setEnabled(False)

        QMessageBox.information(self,
            "Data saved.",
            "Data was successfully saved here: \n{}\n\n{}".format(folder_name, "\n".join(saved)))

    def close(self):
        self.im.close()
        super().close()



