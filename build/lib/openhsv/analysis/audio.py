import numpy as np 
from numba import njit
from scipy.signal import find_peaks
from scipy.stats.mstats import zscore
import matplotlib.pyplot as plt

@njit
def _rolling_std_numba(x, window=20):
    """Rolling standard deviation, faster with numba

    :param x: array
    :type x: numpy.ndarray
    :param window: window size for std, defaults to 20
    :type window: int, optional
    :return: filtered signal
    :rtype: numpy.ndarray
    """
    std = np.zeros_like(x)
    for i in range(window, x.shape[0]):
        std[i-window//2] = np.std(x[i-window:i])
        
    return std

def _findTriggerEnd(reference_signal, window=101, prominence=1, zscoring=True):
    """Uses z-scoring of rolling standard deviation to
    find the end trigger from the camera to synch audio and video.

    :param reference_signal: reference signal in audio data
    :type reference_signal: numpy.ndarray
    :param window: window size, defaults to 101
    :type window: int, optional
    :param prominence: prominence for peak finding, defaults to 1
    :type prominence: int, optional
    :param zscoring: enables z-scoring of data before peak finding, defaults to True
    :type zscoring: bool, optional
    :return: peak location and std signal
    :rtype: tuple(int, numpy.ndarray)
    """
    std = _rolling_std_numba(reference_signal, window)

    if zscoring:
        std = zscore(std)
        
    else:
        std = (std - std.min()) / (std.max() - std.min())
    
    peaks = find_peaks(std, prominence=prominence)[0]
    
    if len(peaks):
        return peaks[0], std
    
    else:
        return False

def sync(reference_signal, audio_signal, start_frame, end_frame, total_frames, debug=False):
    """Synchronizes audio signal with video data using the reference signal.

    .. note::
        This works highly reliable when the reference signal is strong enough.
        Ensure that the gain of the audio interface is set appropriately.

    :param reference_signal: reference signal from camera
    :type reference_signal: numpy.ndarray
    :param audio_signal: recorded audio signal
    :type audio_signal: numpy.ndarray
    :param start_frame: start frame of region selection
    :type start_frame: int
    :param end_frame: end frame of region selection
    :type end_frame: int
    :param total_frames: total amount of recorded frames
    :type total_frames: int
    :param debug: sets debug mode with additional information and plots, defaults to False
    :type debug: bool, optional
    :return: synchronized signal corresponding to video data
    :rtype: numpy.ndarray
    """
    # Find trigger
    trigger_end, std = _findTriggerEnd(reference_signal)

    # Show found trigger and reference signal
    if debug:
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(reference_signal, c='k', alpha=.4, label='raw trace')
        plt.ylabel("audio [au]")

        ax2 = ax.twinx()
        plt.plot(std, c='g', alpha=1, label='z-scored, rolling std')
        plt.scatter(trigger_end, std[trigger_end], marker="x", color='magenta')
        
        fig.legend(loc='best')
        plt.xlabel("time")
        plt.ylabel("std")

    # crop audio
    cropped_reference = reference_signal[:trigger_end]
    cropped_audio     = audio_signal[:trigger_end]

    # find single frames
    # Zscore signal because of varying input amplitude
    z = -zscore(cropped_reference)
    # At least 1.5 STDs, better 2...
    frame_idx = find_peaks(z, height=1.5)[0]

    # Use the last X recorded frames
    recorded_frames = frame_idx[-total_frames:]
    
    # Retrieve the indices from first and last selected frame
    start_frame_idx = recorded_frames[start_frame-1]
    end_frame_idx = recorded_frames[end_frame-1]

    # Show cropped audio signal, acquired video footage section,
    # and selected, downloaded frames
    if debug:
        plt.figure() 
        xc = np.arange(0, cropped_audio.shape[0])
        plt.axvspan(xc[recorded_frames[0]], 
            xc[-1], 
            color=(0,0,0,.2), 
            label="recorded frames")
        plt.plot(xc, cropped_audio, alpha=.2, label="cropped audio")
        plt.plot(xc[start_frame_idx:end_frame_idx],
            cropped_audio[start_frame_idx:end_frame_idx],
            label="saved footage, in sync")
        
        plt.legend(loc='best')

    # Return audio synchronized
    return cropped_audio[start_frame_idx:end_frame_idx]

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    from scipy.io.wavfile import read, write
    from openhsv.analysis.parameters import Audio
    from openhsv.gui.table import Table

    path_to_audio = r"./openhsv/examples/audio.wav"

    # Set some settings
    start_frame = 1234
    end_frame = 2345
    total_frames = 4000

    # Read file
    freq, a = read(path_to_audio)
    reference_signal = a[..., 0]
    audio_signal = a[..., 1]
    x = np.arange(0, audio_signal.shape[0]/freq, 1/freq)

    # Plot raw signal
    plt.figure()
    plt.plot(x, reference_signal, label="reference")
    plt.plot(x, audio_signal, alpha=.5, label="audio")
    plt.legend(loc='best')
    plt.title("Audio recording")
    # plt.show()

    # Compute audio in sync
    cr = sync(reference_signal,
        audio_signal,
        start_frame,
        end_frame,
        total_frames,
        debug=True)

    analysis = Audio(cr, 1/freq, debug=True)
    table = Table(analysis.computeParameters(), title="Audio")
    table.show()

    plt.show()
