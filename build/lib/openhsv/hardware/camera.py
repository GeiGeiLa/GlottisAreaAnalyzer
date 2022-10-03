from openhsv.hardware import XsCamera
import numpy as np
import sys
import ctypes
from abc import ABC, abstractmethod
import imageio as io

class Camera(ABC):
    @abstractmethod
    def __init__(self, verbose=True):
        """An abstract camera class
        
        :param ABC: abstract class
        :type ABC: object
        :param verbose: prints to console, defaults to True
        :type verbose: bool, optional
        """
        self.verbose = verbose

    @abstractmethod
    def openCamera(self):
        """opens camera
        """
        pass

    @abstractmethod
    def configCam(self, *args, **kwargs):
        """configures camera
        """
        pass 

    @abstractmethod
    def setSettings(self, exposure, fps, *args, **kwargs):
        """sets camera settings
        
        :param exposure: exposure time
        :type exposure: int
        :param fps: frames per second
        :type fps: int
        """
        pass

    @abstractmethod
    def isIdle(self):
        """returns if camera is recording or not (boolean)
        """
        pass

    @abstractmethod
    def startGrab(self):
        """Starts acquisition on camera
        """
        pass 

    @abstractmethod
    def stopGrab(self):
        """stops acquisition on camera
        """
        pass

    @abstractmethod
    def live(self):
        """returns live image preview as numpy array
        """
        pass 

    @abstractmethod
    def updateTriggerPosition(self):
        """Updates the trigger position for internal memory view
        """
        pass

    @abstractmethod
    def getMemoryFrame(self, frame_index, by_trigger):
        """gets memory frame from camera onboard memory
        
        :param frame_index: frame index in memory
        :type frame_index: int
        """
        pass

    @abstractmethod
    def closeCamera(self):
        """closes camera connection
        """
        pass

class DummyCamera(Camera):
    def __init__(self, is_color=True, verbose=True):
        self.verbose = verbose 
        self.is_color = is_color
        self._idle = True
        self.i = 0

        # Needed for saving metadata
        self.frames_to_record = 4000
        self.frames_before_trigger = 4000

        self.ims = io.mimread("./openhsv/examples/oscillating_vocal_folds.mp4", memtest=False)

    def openCamera(self):
        if self.verbose:
            print("Dummy camera successfully opened!")
        return True 

    def configCam(self):
        if self.verbose:
            print("Dummy camera successfully configured!")
        return True

    def setSettings(self, *args, **kwargs):
        if self.verbose:
            print("Dummy camera settings were set.")

        return True 

    def isIdle(self):
        return self._idle

    def startGrab(self):
        if self.verbose:
            print("Start grabbing dummy camera")
        self.i = 0
        self._idle = False
        return True 

    def stopGrab(self):
        if self.verbose:
            print("Stop grabbing dummy camera")
        self._idle = True
        return True

    def live(self):
        self.i += 1

        return self.ims[self.i % len(self.ims)]

    def getMemoryFrame(self, frame_index, by_trigger=True):
        return self.ims[frame_index % len(self.ims)]

    def closeCamera(self):
        return True

    def updateTriggerPosition(self):
        return True
    

class IdtCamera(Camera):
    """The IdtCamera class uses the abstract ``Camera`` class to interact with the IDT high-speed camera API. 
    In particular it starts and stops recording, fetches frames from the internal camera memory,
    and sets settings, such as exposure time and framerate.

    :param verbose: Additional information is printed to the command window.
        Maybe important for debugging purposes. Defaults to True.
    :type verbose: bool, optional
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.is_color = 0

        # Get ROI size etc
        self.width = None
        self.height = None
        self.nAddLo = None
        self.nAddHi = None
        self.frame_size = 0

        self.frames_to_record = 4000
        self.frames_before_trigger = 4000

        self.triggerFrameIndex = 0

        self.live_frame = None
        self.live_buffer = None

    def openCamera(self):
        """Searches for attached cameras and opens the first found one
        
        :return: success in opening the camera
        :rtype: bool
        """
        # Shows available cameras, here especially IDT CCmini
        enum_list = list(XsCamera.XsEnumCameras(XsCamera.XS_ENUM_FLT.XS_EF_GE_NO | XsCamera.XS_ENUM_FLT.XS_EF_PCI_X))

        if len(enum_list) == 0:
            print("No cameras found.", file=sys.stderr)
            return False
            # sys.exit(1)

        # Use first camera in list (unlikely that there are more...)
        enum_item = enum_list[0]

        # Open Camera, get info and model
        cam = XsCamera.XsOpenCamera(enum_item.nCameraId)
        CamSerial = XsCamera.XsGetCameraInfo(cam, XsCamera.XS_INFO.XSI_SERIAL)
        CamModel = XsCamera.XsGetCameraInfo(cam, XsCamera.XS_INFO.XSI_CAMERA_MODEL)

        if self.verbose:
            print("CamSerial : ", CamSerial)
            print("CamModel  : ", CamModel)

        self.cam = cam
        self.CamSerial = CamSerial
        self.CamModel = CamModel

        return True

    def configCam(self, px_gain=1, camera_gain=1):
        """Basic camera configuration, such as gain.
        
        :param px_gain: Pixel gain (selects 8 from 10 bits, lower (0), middle (1) and upper (2) 8 bits), defaults to 1
        :type px_gain: int, optional
        :param camera_gain: Camera gain, defaults to 1
        :type camera_gain: int, optional
        """
        # Load default camera settings
        cfg = XsCamera.XsReadDefaultSettings(self.cam)  # Load default settings

        # Set Pixel and camera gain        
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PIX_GAIN, 1)  # Set to upper 8 bits, default: 0
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_GAIN, 2)  # Set to gain = 1, default: 0
        
        # Check colormode of camera
        is_color = XsCamera.XsGetCameraInfo(self.cam, XsCamera.XS_INFO.XSI_SNS_TYPE)[0]
        self.is_color = is_color

        # Dependend on color mode, set different pixel format
        if is_color:
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_IMG_FORMAT, XsCamera.XS_IMG_FMT.XS_IF_BGR24)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PIX_DEPTH, 24)
        else:
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_IMG_FORMAT, XsCamera.XS_IMG_FMT.XS_IF_GRAY8)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PIX_DEPTH, 8)

        # Write camera settings to camera
        XsCamera.XsRefreshCameraSettings(self.cam, cfg)

        if self.verbose:
            sys.stdout.flush()
            print("Camera contains color: ", self.is_color)
            print("Configured camera.")

    def setSettings(self, exposure, fps, roi=(1024, 1024), rec_mode=XsCamera.XS_REC_MODE.XS_RM_CIRCULAR, sync=True):
        """Sets camera settings.
        
        :param exposure: Exposure time in us (microseconds)
        :type exposure: int
        :param fps: Camera sampling rate in frames per second
        :type fps: int
        :param roi: if camera image should cropped to ROI (i.e. may run faster).
            If not desired, set roi=None, otherwise (height, width). Defaults to (1024, 1024)
        :type roi: tuple or None, optional
        :param rec_mode: Recording mode, defaults to XsCamera.XS_REC_MODE.XS_RM_CIRCULAR
        :type rec_mode: XsCamera.XS_REC_MODE, optional
        :param sync: If recording should be synchronized to trigger, defaults to True
        :type sync: bool, optional
        """
        cfg = XsCamera.XsReadCameraSettings(self.cam)
        e = 0.5 # For some reason this is written in the SDK manual...

        # Get dimensions from camera chip, important for buffer size!
        width = XsCamera.XsGetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH)
        height = XsCamera.XsGetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIHEIGHT)

        # Crop to ROI
        if width == 1440 and height == 1024 and roi is not None:
            desired_height, desired_width = roi

            # Width
            # Center it
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIX, (width-desired_width)//2)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH, desired_width)
            width = desired_width

            # Height
            # Center it
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIY, (height-desired_height)//2)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH, desired_height)
            height = desired_height

        # Calculate frame size important for buffer size
        frame_size = width * height * (3 if self.is_color else 1)

        self.width = width
        self.height = height
        self.frame_size = frame_size

        if self.verbose:
            print("Image size: {}x{}x{}".format(height, width, (3 if self.is_color else 1)))

        # Set Frame Rate, Exposure and recording mode
        # In HSV, we record on a circular buffer and trigger the end of acquisition
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PERIOD, int(1000000000 / fps + e))
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_EXPOSURE, int(1000 * exposure))  # Exp in nS
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_REC_MODE, rec_mode)

        # you need to set bayer mode if you want mono images
        # Already set in the ConfigCam...
        # XsCamera.XsSetParameter(cam, cfg, XsCamera.XS_PARAM.XSP_PIX_DEPTH, 24)  # 8 bit, RGB

        # Stops with external trigger (e.g. foot switch)
        if sync:
            XsCamera.XsSetParameter(self.cam,
                cfg,
                XsCamera.XS_SYNCIN_CFG.XS_SIC_EXT_EDGE_LO,
                XsCamera.XS_SYNCIN_CFG.XS_SIC_EXT_EDGE_LO)

        # Update Camera Settings...
        XsCamera.XsRefreshCameraSettings(self.cam, cfg)

        # Create live image buffer
        self.live_buffer = ctypes.create_string_buffer(frame_size)

        # Create frame containing buffer
        # used to stream data live from camera for preview
        self.live_frame = XsCamera.XS_FRAME()
        self.live_frame.nBufSize = frame_size
        self.live_frame.pBuffer = ctypes.addressof(self.live_buffer)
        self.live_frame.nImages = 1

        self._getBufferSize()

    def _getBufferSize(self):
        nAddLo, nAddHi = XsCamera.XsGetCameraInfo(self.cam, XsCamera.XS_INFO.XSI_LIVE_BUF_SIZE)
        self.nAddLo = nAddLo
        self.nAddHi = nAddHi

    def getStatus(self):
        """Returns camera status. Status indicates if the camera is recording on a circular buffer
        or the camera is in idle mode or the recording is done.
        
        :return: business and status
        :rtype: bool, XsCamera.XS_STATUS
        """
        busy, status, _, _ = XsCamera.XsGetCameraStatus(self.cam)
        return busy, status

    def isIdle(self):
        """Determines if camera is not recording via XS_STATUS"""
        busy, status = self.getStatus()

        if status == XsCamera.XS_STATUS.XSST_IDLE or status == XsCamera.XS_STATUS.XSST_REC_DONE:
            if self.verbose:
                print("Status: ", status)

            return True

        else:
            return False

    def startGrab(self):
        """Starts recording images on camera using previous set settings
        """
        self._getBufferSize()

        XsCamera.XsMemoryStopGrab(self.cam)
        XsCamera.XsMemoryStartGrab(self.cam, 
            self.nAddLo, # Internal buffer start
            self.nAddHi, # Internal buffer stop
            self.frames_to_record,  
            self.frames_before_trigger, 
            0, # Callbacks 
            0, 
            0)

    def stopGrab(self):
        """Stops data acquisition
        
        :return: number of recorded frames
        :rtype: int
        """
        recorded_frames = XsCamera.XsMemoryStopGrab(self.cam)

        if self.verbose:
            print("Recorded frames : ", recorded_frames)

        return recorded_frames

    def _bufferToArray(self, buffer, dtype=np.uint8):
        """Converts ctypes buffer to numpy array
        
        :param buffer: string buffer object
        :type buffer: ctypes buffer
        :param dtype: numpy data type, defaults to np.uint8
        :type dtype: numpy.dtype, optional
        """
        im = np.frombuffer(buffer, dtype)
        im = im.reshape(self.height,
                self.width,
                3 if self.is_color else 1)[..., ::-1]  # BGR to RGB

        return im

    def live(self):
        """Returns live image using XsMemoryPreview
        :return:
        """
        XsCamera.XsMemoryPreview(self.cam, self.live_frame)
        im = self._bufferToArray(self.live_buffer)

        return im

    def correctForTrigger(self, frame_index):
        """Correct frame_index for trigger index, otherwise indexing is not historically

        :param frame_index: absolute frame index
        :type frame_index: int
        :return: frame index relative to trigger
        :rtype: int
        """
        i_ = np.arange(self.frames_to_record, dtype=np.int32)
        i_ = np.roll(i_, -self.triggerFrameIndex)

        return int(i_[frame_index])

    def getMemoryFrame(self, frame_index, by_trigger=True):
        """Creates a buffer and retrieves a camera frame by index from the camera's onboard memory.
        
        :param frame_index: The frame index
        :type frame_index: int
        :param by_trigger: If the frame index should be relative to the trigger, defaults to True
        :type by_trigger: bool, optional
        :return: the camera frame in RGB (height, width, 3) or grayscale (height, width, 1)
        :rtype: numpy.ndarray
        """
        frame = ctypes.create_string_buffer(self.frame_size)

        if by_trigger:
            frame_index = self.correctForTrigger(frame_index)

        XsCamera.XsMemoryReadFrame(self.cam, 
            self.nAddLo, # Internal buffer location start 
            self.nAddHi, # Intenral buffer location end
            frame_index, # Frame index (relative to trigger by default)
            frame) # ctypes string buffer

        # Create numpy array from ctypes buffer (UINT8)
        im = self._bufferToArray(frame)

        return im

    def _getTriggerPosition(self):
        """Gets trigger position from camera
        
        :return: trigger position
        :rtype: int
        """
        return XsCamera.XsMemoryReadTriggerPosition(self.cam)

    def updateTriggerPosition(self):
        """updates internally trigger position
        
        :return: trigger position
        :rtype: int
        """
        _, idx, _ = self._getTriggerPosition()
        self.triggerFrameIndex = idx

        if self.verbose:
            print("New trigger index: ", idx)

        return idx

    def closeCamera(self):
        """Closes camera handle"""
        XsCamera.XsCloseCamera(self.cam)
