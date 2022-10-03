import numpy as np
from scipy.signal import find_peaks
import cv2
from skimage.measure import moments
from sklearn.decomposition import PCA
from openhsv.analysis.pvg import compute_pvg, get_labels, _create_maps

class Midline:
    def __init__(self, seg, maxima=None):
        """Midline prediction module.

        Midline is predicted based on segmentation from neural net
        for each peak. Midline is interpolated between peaks. Computations
        are based on `Kist et al., biorxiv 2020 <https://www.biorxiv.org/content/10.1101/2020.08.20.257428v1.abstract>`_.

        :param seg: segmentation masks (T x H x W)
        :type seg: numpy.ndarray
        :param maxima: detected maxima in GAW, defaults to None
        :type maxima: list, optional
        """
        self.seg = seg.astype(np.bool)
        self.gaw = self.seg.sum((1,2))
        self.coordinates = np.zeros((len(self.gaw), 4))   
        self.line_properties = np.zeros((len(self.gaw), 2))
        self.side_gaws = np.zeros((len(self.gaw), 2))
        
        self.maxima = None if maxima is None else maxima
    
    def predict(self, method='pca', time_range=5):
        """Predicts midline with given method on each GAW peak.

        :param method: 'pca' or 'moments', defaults to 'pca'
        :type method: str, optional
        :param time_range: time range around peak to improve prediction, defaults to 5
        :type time_range: int, optional
        """
        # if no peaks were provided through different method
        if self.maxima is None:
            self.maxima = find_peaks(self.gaw)[0]
            
        # Iterate over peaks
        for p in self.maxima:
            # Set lower and higher bounds around each peak
            low = p-time_range if p-time_range >= 0 else 0
            high = p+time_range+1 if p+time_range+1 <= len(self.gaw) else len(self.gaw)
            
            # Create sum image
            im = self.seg[low:high].sum(0)
            
            # Find midline at peak
            a, b = _midline(im, method=method)
            
            # Find intersection coordinates
            posterior, anterior = _intersection(im, a, b)
            
            self.coordinates[p, :2] = posterior
            self.coordinates[p, 2:] = anterior
            self.line_properties[p] = a, b
            
        # Linear interpolate between A and P point estimates
        for i in range(4):
            self.coordinates[:, i] = np.interp(np.arange(len(self.gaw)),
                                        self.maxima,
                                        self.coordinates[self.maxima, i])
            
        for i in range(2):
            self.line_properties[:, i] = np.interp(np.arange(len(self.gaw)),
                                        self.maxima,
                                        self.line_properties[self.maxima, i])    
        
    def side(self):
        """Returns left and right GAW based on midline in each frame

        :return: left and right GAW as array T x 2
        :rtype: numpy.ndarray
        """
        # Iterate over frames
        for frame in range(self.side_gaws.shape[0]):
            # Get midline for frame
            coef, intercept = self.line_properties[frame]

            # Create Left/Right map for frame
            LR = _create_maps(self.seg.shape[1:], coef, np.array([intercept]))[0]

            # Compute segmented pixels for left and right side, respectively
            self.side_gaws[frame, 0] = ((LR >= 0) & self.seg[frame]).sum()
            self.side_gaws[frame, 1] = ((LR < 0) & self.seg[frame]).sum()

        return self.side_gaws

            
    def pvg(self, steps=64):
        """Computes PVG in discrete steps for each side.

        :param steps: resolution along each axis, defaults to 64
        :type steps: int, optional
        :return: phonovibrogram as T x steps*2
        :rtype: numpy.ndarray
        """
        labels = np.zeros_like(self.seg, dtype=np.int32)
        
        for frame in range(len(self.gaw)):
            labels[frame] = get_labels(self.coordinates[frame, 0], 
                              self.coordinates[frame, 2],
                              self.line_properties[frame, 0],
                              self.line_properties[frame, 1],
                              self.seg.shape[1:],
                              steps=steps)
            
        self.labels = labels
        _pvg = compute_pvg(self.seg, labels, steps=steps)
            
        return _pvg


####################
## Helper functions
####################

def _intersection(im, a, b, upsample=10):
    """find intersection between line defined by a*x+b=y and
    the binary image im. For higher accuracy, upsampling of the image
    is encouraged. 

    Parameters
    ----------
    im : numpy.ndarray
        binary image
    a : float
        slope of linear function
    b : float
        intercept of linear function
    upsample : int, optional
        upsampling of binary image, by default 10


    Returns
    -------

    Posterior and anterior point coordinates.
    """    
    if abs(a) > 2000:
        y, x = np.where(im)

        top_y = min(y)
        top_x = np.mean(np.where(im[top_y])[0])

        bottom_y = max(y)
        bottom_x = np.mean(np.where(im[bottom_y])[0])

        return (top_x, top_y), (bottom_x, bottom_y)

    im1 = cv2.resize(im.astype(np.uint8).copy(),
                    (0,0),
                    fx=upsample,
                    fy=upsample).astype(np.bool)
    
    
    bottom_x = 0
    bottom_y = 0
    top_x = 0
    top_y = 0
    
    # Iterate over x values of the image, left to right
    for x in range(im1.shape[1]):
        # Calculate discrete y values
        y = int(a * x  + b * upsample)
        
        # If y is out of the image scene, continue
        if y < 0 or y >= im1.shape[0]:
            continue
            
        # If we enter the first time the binary shape
        # set this as first point
        if im1[y, x]:
            top_x = x
            top_y = y
            break

    # Iterate over x values of the image, right to left
    for x in range(im1.shape[1]-1, -1, -1):
        # Calculate discrete y values
        y = int(a * x  + b * upsample)
        
        # If y is out of the image scene, continue
        if y < 0 or y >= im1.shape[0]:
            continue
            
        # If we enter the first time the binary shape
        # set this as first point
        if im1[y, x]:
            bottom_x = x
            bottom_y = y
            break
    
    # Check for top:
    if top_y > bottom_y:
        top_x, top_y, bottom_x, bottom_y = bottom_x, bottom_y, top_x, top_y
            
    return (top_x/upsample, top_y/upsample), (bottom_x/upsample, bottom_y/upsample)

def imageMoments(im, transpose=True, angle_correction=+np.pi/2):
    """Predicts midline using image moments.

    :param im: [description]
    :type im: [type]
    :param transpose: [description], defaults to True
    :type transpose: bool, optional
    :param angle_correction: [description], defaults to +np.pi/2
    :type angle_correction: [type], optional
    :return: [description]
    :rtype: [type]
    """
    if transpose:
        im = im.T

    if im.dtype == np.float64  or \
        im.dtype == np.float32 or \
        im.dtype == np.int32 or \
        im.dtype == np.int64:

        # ensure that value range -1 to 1
        im = (im / im.max()).astype(np.float64)

    # calculate moments
    M = moments(im)
    x_c = M[1, 0] / M[0, 0]
    y_c = M[0, 1] / M[0, 0]

    # central moments
    mu11 = M[1,1] - x_c * M[0,1]
    mu20 = M[2,0] - x_c * M[1,0]
    mu02 = M[0,2] - y_c * M[0,1]

    A = 2*mu11
    B = mu20-mu02

    # Calculate orientation vector
    xd = (0.5*(1+B/np.sqrt(A**2+B**2)))**0.5
    yd = (0.5*(1-B/np.sqrt(A**2+B**2)))**0.5

    # Calculate major axis of rotation
    angle = 0.5 * np.arctan(2* mu11 / (mu20-mu02+1e-9)) # + angle_correction

    # If B is below zero, add offset
    if B < 0:
        angle += np.pi/2

    # Ensure that angle stays in -pi/2 to pi/2
    if angle > np.pi/2:
        angle -= np.pi

    intercept = y_c - np.tan(angle) * x_c

    return np.tan(angle), intercept

def principalComponents(im, use_2nd=False):
    """Midline prediction using principal component analysis.

    :param im: input image
    :type im: numpy.ndarray
    :param use_2nd: use second principal component, defaults to False
    :type use_2nd: bool, optional
    :return: slope and intercept of midline
    :rtype: tuple(float, float)
    """
    y, x = np.where(im)

    # Compute PCA
    X = np.array([x,y], dtype=np.float64)
    pca = PCA().fit(X.T)


    if pca.components_[0,0] == 0:
        angle = np.pi/2

    else:
        if use_2nd:
            angle = np.arctan(pca.components_[1,1]/pca.components_[1,0])
        else:
            angle = np.arctan(pca.components_[0,1]/pca.components_[0,0])


    intercept = pca.mean_[1] - np.tan(angle) * pca.mean_[0]

    return np.tan(angle), intercept

def _midline(im, method='pca'):
    """Helper function to compute midline from given image

    :param im: image
    :type im: numpy.ndarray
    :param method: method used for midline prediction (pca, moments), defaults to 'pca'
    :type method: str, optional
    :return: slope and intercept of midline
    :rtype: tuple(float, float)
    """
    if method.lower() == 'pca':
        a, b = principalComponents(im)
        if abs(a) < 1:
            a, b = principalComponents(im, use_2nd=True)

    else: # moments
        a, b = imageMoments(im)
    
    return a, b
    
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import pyqtgraph as pg
    import imageio as io

    app = QApplication([])
    
    # Load an example segmentation
    seg = io.mimread(r"./openhsv/examples/segmentation.mp4",
        memtest=False)

    seg = (np.asarray(seg)[..., 0] > 128)

    # Create analysis class and show widget
    M = Midline(seg)
    M.predict()

    # Show segmentation with predicted midline
    im = pg.image(seg.transpose(0, 2, 1), 
                    title="Segmentation with midline")

    line = pg.LineSegmentROI([M.coordinates[0, :2],
                              M.coordinates[0, 2:],],
                              pen="y")

    im.getView().addItem(line)

    # Compute and show phonovibrogram
    pvg = M.pvg()
    pg.image(pvg, title="Phonovibrogram")

    app.exec_()