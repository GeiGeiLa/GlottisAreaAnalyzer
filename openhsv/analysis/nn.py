from xml.etree.ElementPath import prepare_self
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QDialog, QPushButton
from numpy import pad
import pyqtgraph as pg
from skimage.color import rgb2gray
import numpy as np

from ..analysis.midline import Midline
from ..analysis.parameters import GAW, Audio
from ..gui.table import Table
from ..analysis.audio import sync
from pathlib import Path
import logging
import csv

_TF_MODEL_PATH = Path(__file__) / Path(r"..\..\cnn\GlottisSegmentation.h5")

def _divpad(im, multiple_of=32, cval=0):
    """preprocesses an cropped image for feeding into neural network.
    In most convolutional neural networks, images need to have a specific minimum
    size that it can be processed by the network. In a U-Net-like architecture,
    image dimensions should be a multiple of 32.
    
    :param im: cropped input image (grayscale or RGB)
    :type im: numpy.ndarray
    :param multiple_of: number image dimensions should be a multiple of, defaults to 32
    :type multiple_of: int, optional
    :param cval: value that should be used for padded columns, defaults to 0
    :type cval: int, optional
    :return: padded input image
    :rtype: numpy.ndarray
    """
    needed_padding = []
    real_padding = []

    for sh in im.shape:
        if sh > 3 and sh % multiple_of:
            needed_padding.append(multiple_of - sh % multiple_of)
        else:
            needed_padding.append(0)

    real_padding.append([needed_padding[0] // 2,
                         needed_padding[0] // 2 + needed_padding[0] % 2])

    real_padding.append([needed_padding[1] // 2,
                         needed_padding[1] // 2 + needed_padding[1] % 2])

    return pad(im, real_padding, 'constant', constant_values=cval)

class Analysis(QWidget):
    """ Class for analysis.

    Attributes:
        GAW (list[float]): A list of detected area sorted by the order of the original frame.    
    """
    def __init__(self, app=None):
        """Analysis widget that shows the segmentation process of the neural network.
        

        """
        super().__init__()

        self.setWindowTitle("Analysis - Full automatic glottis segmentation")

        self.app = app
        self.model = None
        self.segmentations = []
        self.GAW = []
        self.audio = None
        self.synced_audio = None

        self._initUI()
        self._initTensorflow()

    def _initUI(self):
        """inits the user interface. In particular, it prepares the preview window for
        the endoscopic image, the segmentation map and the glottal area waveform (GAW).
        """
        self.layout = QGridLayout(self)
        self.setGeometry(50, 50, 1800, 600)

        pen = pg.mkPen("y", width=2)

        # Preview Endoscopic image
        self.im = pg.ImageView()
        # Preview segmentation
        self.seg = pg.ImageView()
        # Preview GAW
        self.plt = pg.PlotWidget()
        self.plt.setMaximumWidth(400)
        self.curve = self.plt.plot(pen=pen, symbolBrush=(255, 255, 255), symbolPen="w", symbolSize=8, symbol="o")

        self.im.setImage(np.random.randint(0, 100, (200, 200)))
        self.seg.setImage(np.random.randint(0, 100, (200, 200)))

        self.layout.addWidget(QLabel("Endoscopy image"), 0, 0, 1, 1)
        self.layout.addWidget(QLabel("Segmentation map"), 0, 2, 1, 1)
        self.layout.addWidget(QLabel("Glottal area waveform (GAW)"), 0, 4, 1, 1)

        self.layout.addWidget(self.im, 1, 0, 1, 1)
        self.layout.addWidget(self.seg, 1, 2, 1, 1)
        self.layout.addWidget(self.plt, 1, 4, 1, 1)

    def _initTensorflow(self):
        """Initializes tensorflow and loads glottis segmentation neural network
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(str(_TF_MODEL_PATH), compile=False)

    @property
    def GlobalAreaWave(self) -> list:
        return self.GAW

    def GetAllAreaPredictions(self, ims, normalize=True, reinit=True):
        logging.info("Seg")
        print("Seg")
        """segments an image sequence, such as a video, frame by frame.
        
        :param ims: collection of images
        :type ims: list of numpy.ndarray, or numpy.ndarray
        :param normalize: normalize 0..255 to -1..1, defaults to True
        :type normalize: bool, optional
        :param reinit: deletes any previous segmentation information, defaults to True
        :type reinit: bool, optional
        """
        if reinit:
            self.GAW = []
            self.segmentations = []

        for im in len(ims):
            if normalize:
                im = im.astype(np.float32) / 127.5 - 1

            # Segment frame
            self.segment(im)

            # Ensure that the GUI is response
            if self.app:
                app.processEvents()
        return self.GAW

    def segment(self, im):
        """Segments an endoscopic image using a deep neural network

        :param im: np.ndarray (HxWx3)
        :return:
        """
        # Process image to fit the DNN
        processed = _divpad(rgb2gray(im).astype(np.float32))

        # Run neural network
        pr = self.model.predict(processed[None, ..., None]).squeeze()
        # Save segmentation and GAW
        self.segmentations.append(pr)
        prsum = pr.sum()
        area_value = np.float32(prsum).item()
        print(area_value)
        self.GAW.append(area_value)

        # Transpose image if RGB
        if im.ndim == 3:
            im = im.transpose((1, 0, 2))

        # Transpose image if grayscale
        elif im.ndim == 2:
            im = im.transpose((1, 0))

        # Show image, show segmentation, show GAW
        self.im.setImage(im)
        self.seg.setImage(pr.transpose((1, 0)))
        self.curve.setData(self.GAW)

    def setAudio(self, audio):
        self.audio = audio

    def syncAudio(self, start_frame, end_frame, total_frames, debug=False):
        if self.audio is None:
            return
        
        reference_signal = self.audio[..., 0]
        audio_signal = self.audio[..., 1]

        synced_audio = sync(reference_signal=reference_signal,
            audio_signal=audio_signal,
            start_frame=start_frame,
            end_frame=end_frame,
            total_frames=total_frames,
            debug=debug)

        self.synced_audio = synced_audio

    def computeParameters(self, dt_audio=1/80000, dt_video=1/4000, debug=False):
        """Compute parameters from GAW

        :param dt_audio: audio sampling time in seconds, defaults to 1/80000
        :type dt_audio: float, optional
        :param dt_video: video sampling time in seconds, defaults to 1/4000
        :type dt_video: float, optional
        :param debug: shows debugging information and plots, defaults to False
        :type debug: bool, optional
        """
        # Convert raw segmentations to binary masks
        seg = np.asarray(self.segmentations).round().astype(np.bool)

        # Predict midline from segmentation
        M = Midline(seg)
        M.predict()

        # Use midline for left and right GAW
        gaws = M.side()
        left_gaw  = gaws[..., 0]
        right_gaw = gaws[..., 1]

        # Compute and show values
        gaw = GAW(seg.sum((1,2)), 
            use_filtered_signal=False, 
            use_hanning=False, 
            dt=dt_video)

        gaw.setLeftRightGAW(left_gaw, right_gaw)
        params_GAW = gaw.computeParameters()

        # # Create summary table for parameters
        # self.t = Table(params_GAW, title="GAW parameters")
        # self.t.show()

        if debug:
            # Show complete segmentation with midline
            im = pg.image(seg.transpose(0, 2, 1), 
                        title="Segmentation with midline")

            line = pg.LineSegmentROI([M.coordinates[0, :2],
                                    M.coordinates[0, 2:],],
                                    pen="y")

            im.getView().addItem(line)

            # Show complete GAW plot with detected cycles
            gaw_plot = pg.plot(gaw.t, gaw.raw_signal,
                title="GAW with cycles")

            cs = [(241, 196, 15), (231, 76, 60)]
            i = 0

            for o, c in zip(gaw.opening, gaw.closing):
                i1 = pg.PlotCurveItem(gaw.t[o:c], np.zeros_like(gaw.t[o:c]))
                i2 = pg.PlotCurveItem(gaw.t[o:c], gaw.raw_signal[o:c])
                between = pg.FillBetweenItem(i1, i2, brush=cs[i % len(cs)])
                gaw_plot.getPlotItem().addItem(between)
                i += 1

            # Show left and right gaw
            LR_plot = pg.plot(title="Left and right GAW")
            LR_plot.plot(gaw.t, left_gaw)
            LR_plot.plot(gaw.t, -right_gaw)

        # If audio data is available
        if type(self.synced_audio) != type(None):
            if debug:
                pg.plot(self.synced_audio, 
                    title="Synchronized audio")

            a = Audio(self.synced_audio,
                dt=dt_audio)

            params_Audio = a.computeParameters()

            
            self.t2 = Table(params_Audio, title="Audio parameters")
            self.t2.show()   
        else:
            params_Audio = None 

        return dict(GAW=params_GAW, Audio=params_Audio)

    def get(self):
        """returns GAW and segmentation maps for video
        
        :return: GAW and segmentations
        :rtype: tuple(list, list(numpy.ndarray))
        """
        return dict(gaw=self.GAW, segmentation=self.segmentations)



class ROIDialog(QDialog):
    def __init__(self, ims):
        super().__init__()

        self.l = QGridLayout(self)
        self.imv = pg.ImageView()
        self.imv.setImage(ims.transpose(0, 2, 1, 3))
        self.l.addWidget(self.imv)

        self.roi = pg.RectROI((0, 0), (10, 10))
        self.imv.addItem(self.roi)

        self.b = QPushButton("Save ROI and close window")
        self.b.clicked.connect(self._close_and_save)
        self.l.addWidget(self.b)

        self.setWindowTitle("Select an ROI around the glottis")
        self.setMinimumWidth(800)
        self.setMinimumHeight(800)

        self.save = False
        self.exec_()
        
    def _close_and_save(self):
        self.save = True
        self.close()

    def __getitem__(self, i):
        pos = self.roi.pos()
        size = self.roi.size()

        arr = int(pos[1]), int(pos[1]+size[1]), int(pos[0]), int(pos[0]+size[0])
        return arr[i]
