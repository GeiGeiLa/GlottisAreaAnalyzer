import numpy as np
from numba import njit, prange

def _find_orthogonal_points(x_low, x_high, coef, intercept, steps=64):
    """
    Find orthogonal points equally spaced along the main anterior-posterior axis.
    Returns intercept and coefficient for linear equation.
    :param x: ndarray, x from AP axis
    :param y: ndarray, y from AP axis
    :param coef: float, coef from linear regression of AP axis
    :param intercept: float, intercept from linear regresiion of AP axis
    :param steps: int, sections in between AP
    :return: tuple of xs, ys, coef of orthogonal axis (float), ndarray of intercepts
    """
    # Calculate the angle of the slope
    # slope = tan(angle)
    _alpha = np.arctan(coef)

    # Calculate the orthogonal slope (alpha - 90°)
    coef_orth = np.tan(_alpha - np.pi / 2)

    # Get points that should intersect the AP axis
    xs = np.linspace(x_low, x_high, steps)#[1:-1]
    ys = coef * xs + intercept
    
    intercepts_orth = ys - coef_orth * xs

    return xs, ys, coef_orth, intercepts_orth


@njit
def _create_maps(im_shape, c, i):
    """
    Creates distance maps for all orthogonal axes
    :param im_shape: tuple, image shape of video
    :param c: float, coefficient
    :param i: ndarray, intercepts
    :return: ndarray, amount of intercepts x image shape
    """
    labels = np.zeros((i.shape[0],) + im_shape, dtype=np.float32)

    for j in range(i.shape[0]):
        for y in range(im_shape[0]):
            for x in range(im_shape[1]):
                labels[j, y, x] = (y - (c * x + i[j]))

    return labels


@njit
def _find_parts(im_shape, labels_parallel, labels_LR):
    """
    Creates a label map depending on distance to axis and left or right
    axis of anterior posterior axis.
    :param im_shape: tuple, image shape
    :param labels_parallel: ndarray, from create_maps
    :param labels_LR: ndarray, from create_maps, but only image shape!
    :return: ndarray, labels from -steps to +steps, no 0
    """
    labels = np.zeros(im_shape, dtype=np.int8)

    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            sign = -1

            if labels_LR[i, j] >= 0:
                sign = 1

            closest_to = np.argmin(np.sqrt(labels_parallel[:, i, j] ** 2))

            #if labels_parallel[closest_to, i, j] < 0:
            #    closest_to += 1

            labels[i, j] = sign * (closest_to + 1)

    return labels


def get_labels(x_low, x_high, coef, intercept, image_shape, steps=64):
    r"""Function for getting left/right step-id'd labels based on the AP axis

    :param x_low: lower x coordinate 
    :type x_low: float
    :param x_high: higher x coordinate
    :type x_high: float
    :param coef: coefficient of linear regression AP
    :type coef: float
    :param intercept: intercept of linear regression AP
    :type intercept: float
    :param image_shape: image size (HxW) for label
    :type image_shape: tuple(int, int)
    :param steps: steps between A and P
    :type steps: int
    :return: label map
    :rtype: numpy.ndarray
    """
    xs, ys, coef_orth, intercepts_orth = _find_orthogonal_points(x_low, x_high, coef, intercept, steps=steps)
    labels_parallel = _create_maps(image_shape, coef_orth, intercepts_orth)
    labels_LR = _create_maps(image_shape, coef, np.array([intercept]))[0]

    labels = _find_parts(image_shape, labels_parallel, labels_LR)

    return labels
    
def compute_pvg(s, labels, steps=64):
    """Calculates Phonovibrogram based on labels.

    :param s: segmented area (TxYxX)
    :type s: numpy.ndarray
    :param labels: labelled image
    :type labels: numpy.ndarray
    :param steps: PVG resolution, defaults to 64
    :type steps: int, optional
    :return: PVG, time x 2*steps
    :rtype: numpy.ndarray
    """
    pvg = np.zeros((s.shape[0], steps * 2))
    l = np.zeros((steps*2, )+labels.shape[1:], dtype=np.bool_)
    
    all_steps = np.arange(1, steps*2+1)
    all_steps[:steps] = all_steps[:steps][::-1]
    all_steps[steps:] = -(all_steps[steps:] - steps)
    
    for frame in range(s.shape[0]):
        f = s[frame]
        
        for i, j in enumerate(all_steps):
            l[i] = labels[frame] == j
        
        for i in range(steps*2):
            pvg[frame, i] = (f & l[i]).sum()

    return pvg