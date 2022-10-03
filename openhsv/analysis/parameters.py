from PyQt5.QtWidgets import QWidget, QGridLayout
import pyqtgraph as pg
import numpy as np
from scipy.signal import find_peaks, filtfilt, butter
from scipy.optimize import curve_fit
from datetime import datetime
from scipy.signal import medfilt
from numba import njit
import matplotlib.pyplot as plt

"""
    ***************
    Helper functions
    ***************
"""

@njit
def movingAverage(x, n=3):
    """computes moving average from x with a window of n

    :param x: input array
    :type x: numpy.ndarray
    :param n: window, defaults to 3
    :type n: int, optional
    :return: filtered array
    :rtype: numpy.ndarray
    """
    y = np.zeros_like(x, dtype=np.float32)

    for i in range(x.shape[0]):
        low = i-n if i > n else 0

        y[i] = np.mean(x[low:i+1])

    return y

@njit
def _find_bottom(x, t=0.02):
    """Finds bottom of a decaying function
    
    :param x: input array
    :type x: numpy.ndarray
    :param t: threshold deviation from minimum, defaults to 0.02
    :type t: float, optional
    :return: index of bottom
    :rtype: int
    """
    the_min = np.min(x)
    
    for i in range(len(x)):
        if abs(x[i]-the_min) > t:
            continue
        else:
            break
            
    return i

def _lin(x, m, b):
    """Linear function

    .. math:
        f(x) = m \cdot x + b

    :param x: x
    :type x: int, float or numpy.ndarray
    :param m: slope
    :type m: int, float
    :param b: intercept
    :type b: int, float
    :return: eval
    :rtype: int, float or numpy.ndarray
    """    
    return m*x+b

"""
    ***************
    Event detection
    ***************
"""

def detectMaximaMinima(s, distance=5, rel_height=.01, use_prominence=True, clean_f0=None):
    """Detect maxima and minima from a signal ``s``.
    
    :param s: signal
    :type s: numpy.ndarray
    :param distance: distance between two peaks in samples, defaults to 5
    :type distance: int, optional
    :param rel_height: minimum relative height of a found peak, defaults to .35
    :type rel_height: float, optional
    :param use_prominence: uses peak prominence for peak detection, defaults to True
    :type use_prominence: bool, optional
    :return: tuple of maxima and minima locations
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    
    if use_prominence:
        width = None if clean_f0 is None else clean_f0


        p_max = find_peaks(s, prominence=.5, width=width)[0]
        p_min = find_peaks(-s, prominence=.5, width=width)[0]

    else:
        h_max = s.max() - s.min()
        h_min = (-s).max() - (-s).min()

        p_max = find_peaks(s, 
            distance=distance, 
            height=rel_height*h_max)[0]
        
        p_min = find_peaks(-s, 
            distance=distance, 
            height=rel_height*h_min)[0]

    return p_max, p_min

def detectOpeningAndClosingEvents(signal, p_max, t=0.02):
    """Detects glottis opening and closing events relative to 
    maximum opening events.
    
    :param signal: glottal area waveform
    :type signal: numpy.ndarray
    :param p_max: maxima of glottal area waveform
    :type p_max: list of indexes
    :param t: threshold for finding signal bottom, defaults to 0.02
    :type t: float, optional
    :return: opening and closing point for each maximum
    :rtype: tuple(list(int), list(int))
    """
    opening = []
    closing = []

    for i in range(len(p_max)):
        # Search for bottom in local boundary, max. to adjacent peak
        # or to the beginning or end of the signal sequence
        lower_bound = p_max[i-1] if i else 0
        higher_bound = p_max[i+1] if i < len(p_max)-1 else len(signal)-1

        # Run down the signal hill until bottom is found
        starts_opening = _find_bottom(signal[lower_bound:p_max[i]][::-1], t=t)
        stops_closing  = _find_bottom(signal[p_max[i]:higher_bound], t=t)

        # Correct for location
        starts_opening = p_max[i]-starts_opening-1
        stops_closing  = p_max[i]+stops_closing

        opening.append(starts_opening)
        closing.append(stops_closing)

    return np.asarray(opening), np.asarray(closing)

def computeOpenAndClosedIntervals(t, opening, closed):
    """computes the opened and closed intervals during each cycle.

    :param t: time
    :type t: numpy.ndarray
    :param opening: indices of opening points
    :type opening: list(int)
    :param closed: indices of closing points
    :type closed: list(int)
    :return: duration of opening and closed phases
    :rtype: tuple(list(float), list(float))
    """    
    open_t = [t[c]-t[o] for o, c in zip(opening, closed)]
    closed_t = [t[o]-t[c] for o, c in zip(opening[1:], closed)]
    closed_t.append(t[-1]-t[closed[-1]]) # why?

    return np.asarray(open_t), np.asarray(closed_t)

def computeOCandCOTransitions(t, opening, closed, p_max):
    """Computes Open->Closed (OC) and Closed-Open (CO) transitions.

    :param t: time
    :type t: numpy.ndarray
    :param opening: indices of opening points
    :type opening: list(int)
    :param closed: indices of closing points
    :type closed: list(int)
    :param p_max: indices of cycle maxima
    :type p_max: numpy.ndarray
    :return: CO and OC durations
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """    
    CO = np.array([t[p]-t[o]  for o, p in zip(opening, p_max)])
    OC = np.array([t[c]-t[p]  for c, p in zip(closed, p_max)])

    return CO, OC

"""
    ***************
    Parameters
    ***************
"""

def F0fromCycles(T, verbose=False, epsilon=1e-9):
    r"""determine fundamental frequency (F0) based on period lengths
    
    :param T: periods
    :type T: numpy.ndarray
    :param verbose: prints F0 mean and standard deviation, defaults to False
    :type verbose: bool, optional
    :return: mean of F0 and std of F0
    :rtype: tuple(float, float)
    """
    freq_est = 1/(T+epsilon)
    f0 = freq_est.mean()
    f0_std = freq_est.std()

    if verbose:
        print("F0: {:.2f} ± {:.2f} Hz".format(f0, f0_std))
        
    return f0, f0_std

def F0fromFFT(fft, freqs, freq_lower=75, freq_higher=500):
    """fundamental frequency from power spectrum determined
    assuming that fundamental frequency is equal to the dominant frequency.
    Frequency is determined in physiological range (75-500 Hz).

    :param fft: fast fourier transformation of signal
    :type fft: numpy.ndarray
    :param freqs: frequencies corresponding to FFT
    :type freqs: numpy.ndarray
    :param freq_lower: lower border for F0 prediction, defaults to 75
    :type freq_lower: int, optional
    :param freq_higher: higher border for F0 prediction, defaults to 500
    :type freq_higher: int, optional
    :return: fundamental frequency (F0)
    :rtype: float
    """
    f0 = np.argmax(abs(fft[(freqs > freq_lower) & (freqs < freq_higher)]))
    f0 = freqs[f0+len(freqs[freqs <= freq_lower])]

    return f0

def F0fromAutocorrelation(signal, freq=40000):
    """Fundamental frequency using autocorrelation.

    Steps:
        1) rFFT of signal
        2) Corresponding frequencies
        3) Autocorrelation
        4) Find peaks in autocorrelation
        5) Remove first unrelevant peaks
        6) Use first peak as fundamental frequency

    :param signal: Signal (e.g. audio trace or GAW)
    :type signal: numpy.ndarray
    :param freq: sampling frequency, defaults to 40000 Hz
    :type freq: int
    :return: fundamental frequency (F0)
    :rtype: float
    """
    # rFFT from Signal
    fft = np.fft.rfft(signal)

    # Corresponding frequencies
    freqs = np.fft.rfftfreq(len(signal), 1/freq)
    
    # Autocorrelation using fft
    R = np.fft.irfft(fft.conj()*fft)

    # Find peaks in autocorrelation
    p = find_peaks(R)[0]
    # Remove first peak artifacts
    p = p[1:] if p[0] < 10 else p

    # f0 = np.argmax(R[p])
    f0 = 0

    return freqs[p[f0]]


"""
    +++++++
    GAW
    +++++++
"""

def asymmetryQuotient(CO, OC):
    r"""Asymmetry Quotient (AsyQ)

    .. math::
        \text{AsyQ} = \frac{1}{N} \sum_{i=1}^{N} \frac{\frac{CO_i}{OC_i}}{1+\frac{CO_i}{OC_i}}

    :param CO: Closed->Open transitions
    :type CO: numpy.ndarray
    :param OC: Open->Closed transitions
    :type OC: numpy.ndarray
    :return: asymmetry quotient (AQ), a.u.
    :rtype: float
    """    
    aq = np.mean((CO / OC) / (1 + (CO / OC)))

    return aq

def closingQuotient(CO, t_open):
    r"""Closing Quotient (CQ)

    .. math::
        \text{CQ} = \frac{1}{N} \sum_{i=1}^{N} \frac{CO_i}{t_{open,i}}

    :param CO: Closed->Open transitions
    :type CO: numpy.ndarray
    :param t_open: Open interval
    :type t_opent: numpy.ndarray
    :return: closing quotient (CQ), a.u.
    :rtype: float
    """    
    cq = np.mean(CO/t_open)

    return cq

def openQuotient(t_open, t_closed):
    r"""Open Quotient (OQ)

    .. math::
        \text{OQ} = \frac{1}{N} \sum_{i=1}^{N} \frac{t_{open,i}}{t_{open,i}+t_{closed,i}}

    :param t_open: Open intervals
    :type t_open: numpy.ndarray
    :param t_closed: Closed intervals
    :type t_closed: numpy.ndarray
    :return: open quotient (OQ), a.u.
    :rtype: float
    """    
    oq = np.mean(t_open / (t_open+t_closed))

    return oq

def rateQuotient(CO, OC, t_closed):
    r"""Rate Quotient (RQ)

    .. math::
        \text{RQ} = \frac{1}{N} \sum_{i=1}^{N} \frac{t_{closed,i} - CO_i}{OC_i}

    :param CO: Closed->Open transitions
    :type CO: numpy.ndarray
    :param OC: Open->Closed transitions
    :type OC: numpy.ndarray
    :param t_closed: closed intervals
    :type t_closed: numpy.ndarray
    :return: rate quotient (RQ), a.u.
    :rtype: float
    """    
    rq = (t_closed + CO) / OC

    return np.mean(rq), np.std(rq) 

def speedIndex(CO, OC, t_open):
    r"""Speed Index (SI)

    .. math::
        \text{SI} = \frac{1}{N} \sum_{i=1}^{N} \frac{CO_i - OC_i}{t_{open,i}}

    :param CO: Closed->Open transitions
    :type CO: numpy.ndarray
    :param OC: Open->Closed transitions
    :type OC: numpy.ndarray
    :param t_open: open intervals
    :type t_open: numpy.ndarray
    :return: speed index (SI), a.u.
    :rtype: float
    """    
    si = (CO-OC) / t_open

    return np.mean(si), np.std(si)

def speedQuotient(CO, OC):
    r"""Speed Quotient (SQ)

    .. math::
        \text{SQ} = \frac{1}{N} \sum_{i=1}^{N} \frac{CO_i}{OC_i}

    :param CO: Closed->Open transitions
    :type CO: numpy.ndarray
    :param OC: Open->Closed transitions
    :type OC: numpy.ndarray
    :return: speed quotient (SQ), a.u.
    :rtype: float
    """    
    sq = CO / OC

    return np.mean(sq), np.std(sq)

def meanJitter(T):
    r"""Calculating the mean jitter in ms from signal periods

    .. math::
        \text{mean-Jitter} = \frac{\sum_{i=1}^{N-1}|T_i - T_{i-1}|}{N-1}
    
    :param T: The signal periods
    :type T: numpy.ndarray or list
    :return: mean jitter in ms
    :rtype: float
    """
    return np.mean(np.abs(np.diff(T))) * 1000

def jitterPercent(T): 
    r"""Calculating the jitter in percent from signal periods.
    
    .. math::
        \text{Jitter[%]} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1}|T_i - T_{i-1}|}{\frac{1}{N}\sum_{i=0}^{N-1}T_i} \cdot 100.
    
    :param T: The signal periods
    :type T: numpy.ndarray or list
    :return: jitter in percent
    :rtype: float
    """
    numerator   = np.mean(np.abs(T[1:]-T[:-1]))
    denominator = np.mean(T)
    
    return numerator / denominator * 100

def meanShimmer(A, epsilon=1e-9):
    r"""Calculating the mean shimmer in dB from the signal amplitude maxima.

    .. math::
        \text{mean-Shimmer [db]} = \frac{20}{N-1}\sum_{i=0}^{N-2}|\log_{10} \left[ \frac{A_i}{A_{i+1}} \right]|.

    :param A: The signal amplitude maxima
    :type A: numpy.ndarray or list
    :return: mean shimmer in db
    :rtype: float
    """    
    meanShimmer = 20 * np.mean([np.abs(np.log10(A[i]/A[i+1]+epsilon)) for i in range(len(A)-1)])
    
    return meanShimmer

def shimmerPercent(A, e=1e-5):
    r"""Calculating the shimmer in percent from the signal amplitude maxima.
    
    .. math::
        \text{Shimmer[%]} = \frac{\frac{20}{N-1}\sum_{i=0}^{N-2}|\log_{10} \left[ \frac{A_i}{A_{i+1}} \right]|}{\frac{20}{N}\sum_{i=0}^{N-1}|\log_{10} A_i|} \cdot 100.

    :param A: The signal amplitude maxima
    :type A: numpy.ndarray or list
    :return: shimmer in percent
    :rtype: float
    """
    numerator   = np.mean([np.abs(np.log10(A[i]/A[i+1])) for i in range(len(A)-1)])
    denominator = np.mean(np.abs(np.log10(A)))

    return numerator / (denominator+e) * 100

def periodPerturbationFactor(T):
    r"""Calculating the Period Perturbation Factor (PPF) in arbitrary units using the signal periods.

    .. math::
        \text{PPF} = \frac{1}{N-1} \sum_{i=1}^{N-1}|\frac{T_i-T_{i-1}}{T_i}| \cdot 100
    
    :param T: periods
    :type T: list
    :return: PPF in percent
    :rtype: float
    """
    PPT = np.abs(np.diff(T)/T[1:])
    return np.mean(PPT) * 100, np.std(PPT) * 100

def glottalGapIndex(signal, opening, epsilon=1e-9):
    r"""Glottal Gap Index (GGI) that computes the relation between 
    minimum and maximum glottal area in each glottal cycle.

    .. math::
        GGI = \frac{1}{N} \sum_i^N \frac{\min(a_i)}{\max(a_i)}

    :param signal: glottal area waveform
    :type signal: numpy.ndarray
    :param opening: points where the glottis opens
    :type opening: list
    :param epsilon: numerical stability, defaults to 1e-9
    :type epsilon: float, optional
    :return: glottal gap index mean and std
    :rtype: tuple(float, float)
    """
    ggi = []
    
    for os, oe in zip(opening, opening[1:]):
        ggi.append(
            np.min(signal[os:oe+1]) / (np.max(signal[os:oe+1]) + epsilon)
        )

    return np.mean(ggi), np.std(ggi)

def amplitudePerturbationFactor(A):
    r"""Amplitude perturbation factor.

    .. math::
        APF = \frac{1}{N} \sum_{i=1}^{N-1}|\frac{A_i-A_{i-1}}{A_i}| \cdot 100

    :param A: amplitudes in each cycle
    :type A: list or numpy.ndarray
    :return: APF mean and std
    :rtype: tuple(float, float)
    """
    APF = []
    
    for i in range(1, len(A)):
        APF.append(abs((A[i] - A[i-1]) / A[i]))

    return np.mean(APF) * 100, np.std(APF) * 100

def amplitudePerturbationQuotient(A, k=3):
    r"""Amplitude perturbation quotient

    .. math::
        l = \frac{k-1}{2}
    .. math::
        APQ_k = \frac{1}{N-k} \sum_{i=l}^{N-l-1}|1-\frac{k \cdot A_i}{\sum_{j=-l}^{l}A_{i+j}}| \cdot 100

    :param A: amplitudes in each cycle
    :type A: list or numpy.ndarray
    :param k: range, defaults to 3
    :type k: int, optional
    :return: APQ mean and std
    :rtype: tuple(float, float)
    """
    APQ = []
    lim = int((k-1)/2)

    for i in range(lim, len(A)-lim):
        numer = k * A[i]
        denom = A[i-lim:i+lim+1].sum()

        APQ.append(abs(1-numer/denom))

    return np.mean(APQ) * 100, np.std(APQ) * 100

def amplitudeQuotient(signal, opening):
    r"""Amplitude Quotient

    .. math::
        AQ = \frac{A_i}{|\min_j \frac{d}{dj}f_i(j)|}

    :param signal: audio or GAW signal
    :type signal: numpy.ndarray
    :param opening: indices where glottis starts to open
    :type opening: list
    :return: AQ mean and std
    :rtype: tuple(float, float)
    """
    AQ = []

    dSignal = np.insert(np.diff(signal), 0, 0)

    for i, j in zip(opening, opening[1:]):
        m = abs(min(dSignal[i:j]))
        Ai = max(signal[i:j]) - min(signal[i:j])

        AQ.append(Ai/m)


    return np.mean(AQ), np.std(AQ)


def stiffness(signal, opening):
    r"""Stiffness

    .. math::
        Stiffness = \frac{|\max_j |\frac{d}{dj}f_i(j)||}{A_i}

    :param signal: audio or GAW signal
    :type signal: numpy.ndarray
    :param opening: indices where glottis starts to open
    :type opening: list
    :return: stiffness mean and std
    :rtype: tuple(float, float)
    """
    S = []

    dSignal = np.insert(np.diff(signal), 0, 0)

    for i, j in zip(opening, opening[1:]):
        m = max(dSignal[i:j])
        Ai = max(signal[i:j]) - min(signal[i:j])

        S.append(m/Ai)

    return np.mean(S), np.std(S)


def harmonicNoiseRatio(signal, freq, freq_lower=50, freq_higher=450, filter_autocorrelation=False, epsilon=1e-9):
    r"""Computes Harmonic-Noise-Ratio  (HNR) using autocorrelation approximation.
    First, it computes the fundamental frequency using the power spectrum of ``signal``.
    Next, it computes the autocorrelation in Fourier space. 
    Then, local maxima in the autocorrelation are found, the HNR computed and the maximum HNR
    and the corresponding frequency is returned.

    .. math::
        R_{xx} = \frac{1}{N} \sum_{k=l}^{N-1} x[k]x[k-l]

    .. math::
        HNR = \frac{R_{xx}[T_0]}{R_{xx}[0]-R_{xx}[T_0]}

    :param signal: audio signal
    :type signal: numpy.ndarray
    :param freq: sampling rate/frequency, e.g. 44100
    :type freq: int
    :param freq_lower: lower frequency cut-off, defaults to 50
    :type freq_lower: int, optional
    :param freq_higher: higher frequency cut-off, defaults to 350
    :type freq_higher: int, optional
    :return: HNR [dB], F0_FFT [Hz], F0_Autocorr [Hz]
    :rtype: tuple(float, float, float)
    """    
    # Create timestamps for signal ``s`` 
    time = np.arange(0, len(signal)/freq, 1/freq)
    
    # rFFT from Signal
    fft = np.fft.rfft(signal)
    
    # Corresponding frequencies
    freqs = np.fft.rfftfreq(len(signal), 1/freq)
    
    # Autocorrelation using fft
    R = np.fft.irfft(fft.conj()*fft)

    # Remove higher harmonics for peak detection
    if filter_autocorrelation:
        R_fft = np.fft.fft(R)
        R_freq = np.fft.fftfreq(R.size, d=1/freq)
        R_fft[abs(R_freq) > freq_higher] = 0
        Rp = np.fft.ifft(R_fft).real
    else:
        Rp = R.copy()
    
    # Find peaks in autocorrelation
    p = find_peaks(Rp, width=20)[0]
    # Remove first peak artifacts
    p = p[1:] if p[0] < 10 else p
    # Remove unnatural peaks (minima and frequencies > 350)
    p = [i for i in p if 1/time[i] < freq_higher and R[i] > 0]

    # Compute HNR for peaks that are higher than minimum frequency (freq_lower)
    hnr = [10 * np.log10(R[pi] / (R[0]-R[pi])) for pi in p if 1/time[pi] > freq_lower]

    return np.max(hnr), 1/time[p[np.argmax(hnr)]]

def cepstralPeakProminence(signal, freq, freq_lower=70, freq_higher=350, plot=False):
    r"""Computes cepstral peak prominence from signal using Fourier transformations.

    Steps:
        1) Compute FFT from signal
        2) Compute fundamental frequency from power spectrum
        3) Compute cepstrum from FFT, filter with moving average (window = 3)
        4) Find maximum peak in cepstrum
        5) Find corresponding quefrency
        6) Fit line to cepstrum
        7) Compute distance from peak to line --> Cepstral Peak Prominence

    :param signal: audio signal
    :type signal: numpy.ndarray
    :param freq: sampling rate/frequency, e.g. 44100
    :type freq: int
    :param freq_lower: lower frequency cut-off, defaults to 70
    :type freq_lower: int, optional
    :param freq_higher: higher frequency cut-off, defaults to 350
    :type freq_higher: int, optional
    :param plot: plots the cepstrum, the line and the peak prominence, defaults to False
    :type plot: bool, optional
    :return: CPP [dB], F0_FFT [Hz], F0_Cepstrum [Hz]
    :rtype: tuple(float, float, float)
    """    
    # Create timestamps for signal ``s`` 
    time = np.arange(0, len(signal)/freq, 1/freq)
    
    # rFFT from Signal
    fft = np.fft.rfft(signal)
    
    # Corresponding frequencies
    freqs = np.fft.rfftfreq(len(signal), 1/freq)
    
    # Find fundamental frequency in region
    f0 = np.argmax(abs(fft[(freqs > freq_lower) & (freqs < freq_higher)]))
    f0 = freqs[f0+len(freqs[freqs <= freq_lower])] 
    
    # Compute quefrencies and cepstrum
    epsilon = 1e-9
    cepstrum = np.fft.irfft(np.log10(np.abs(fft)+epsilon)).real
    cepstrum = movingAverage(cepstrum, 3)
    cepstrum = cepstrum[:len(cepstrum)//2]
    quefrencies = time[:len(time)//2]
    
    # Remove artifacts from beginning
    cepstrum[quefrencies < 0.002] = 0
    # Remove higher real quefrencies
    cepstrum[quefrencies > 0.02] = 0
    
    # Fit a line to cepstrum
    m, b = curve_fit(_lin, quefrencies, cepstrum)[0]
    
    # Find cepstrum peak around f0:
    p = np.argmax(cepstrum)

    # Find fundamental frequency from cepstrum
    quefrency = time[p]
    f0_q = 1/quefrency
    
    # Compute CPP (distance cepstrum peak to fitted line)
    cpp = cepstrum[p]-_lin(time[p],m,b)
    
    if plot:
        plt.figure()
        plt.plot(quefrencies, cepstrum, label="cepstrum")
        plt.plot(quefrencies, _lin(quefrencies, m, b), c='k', label="est. baseline")
        plt.plot([quefrency, quefrency], 
            [_lin(time[p],m,b), cepstrum[p]],
            label="CPP")

        plt.legend(loc='best')
        # plt.xlim([quefrency-.002, quefrency+.002])
        plt.title("Cepstral Peak Prominence")
    
    return -10 * np.log10(cpp+epsilon), f0, f0_q

def phaseAsymmetryIndex(left_gaw, right_gaw, opening):
    r"""Phase Asymmetry Index (PAI)

    .. math::
        PAI = \frac{|argmax_j GA_i^L(j) - argmax_j GA_i^R(j)|}{N_i}

    .. note::
        We use here the maximum instead of the minimum, as it is more
        likely that there are multiple time points where the value
        is minimal or close to the minimum (i.e. when the glottis
        is closed for prolonged time).  

    :param left_gaw: GAW of left vocal fold
    :type left_gaw: numpy.ndarray
    :param right_gaw: GAW from right vocal fold
    :type right_gaw: numpy.ndarray
    :param opening: indices where glottis starts to open (i.e. new cycle)
    :type opening: list
    :return: PAI mean and std
    :rtype: tuple(float, float)
    """
    PAI = []

    for i0, i1 in zip(opening, opening[1:]):
        L = left_gaw[i0:i1+1]
        R = right_gaw[i0:i1+1]

        PAI.append(
            abs(np.argmax(L) - np.argmax(R)) / len(L)
        )

    return np.mean(PAI), np.std(PAI)

def amplitudeSymmetryIndex(left_gaw, right_gaw, opening, epsilon=1e-5):
    r"""Amplitude Symmetry Index (ASI)

    .. math:: 
        ASI = \frac{\min(\max_j(GA_i^L(j)), \max_j(GA_i^R(j)))}{\max(\max_j(GA_i^L(j)), \max_j(GA_i^R(j)))}

    :param left_gaw: GAW of left vocal fold
    :type left_gaw: numpy.ndarray
    :param right_gaw: GAW from right vocal fold
    :type right_gaw: numpy.ndarray
    :param opening: indices where glottis starts to open (i.e. new cycle)
    :type opening: list
    :param epsilon: numerical stability, defaults to 1e-5
    :type epsilon: float, optional
    :return: ASI mean and std
    :rtype: tuple(float, float)
    """
    ASI = []

    for i0, i1 in zip(opening, opening[1:]):
        L = max(left_gaw[i0:i1+1])
        R = max(right_gaw[i0:i1+1])

        ASI.append(
            min(L, R) / (max(L, R)+epsilon)
        )

    return np.mean(ASI), np.std(ASI)


class Signal:
    def __init__(self, raw_signal, dt=1/4000, debug=True):
        r"""Inits the signal class with the raw signal, e.g. audio data or glottal area waveform.
        
        :param raw_signal: Audio signal, GAW or alike
        :type raw_signal: numpy.ndarray
        """
        self.raw_signal = raw_signal
        self.dt = dt

        # Time
        # First sample is acquired after dt,
        # in total N dt-spaced samples
        self.t = np.linspace(dt, dt*len(raw_signal), len(raw_signal))
        self.debug = debug

        # Filtered signal to remove noise
        self.filtered_signal = raw_signal

        # Prepare feature vectors
        # Raw peaks [maxima, minima]
        # T: Periods from maxima
        # A: Amplitude maxima
        self.raw_peaks = None
        self.clean_peaks = None
        self.T = None
        self.A = None

        # FFT of raw signal
        self.fft = None
        # Corresponding frequencies to FFT
        self.fftfreq = None
        # Peaks of the power spectrum
        self.fft_peaks = None

        # Cepstrum, FFT of FFT
        self.cepstrum = None
        self.cepstrumfreq = None

    def computeFFT(self, use_filtered_signal=True, use_hanning=True, lowpass_filter=20):
        signal = (self.filtered_signal if use_filtered_signal else self.raw_signal).copy()

        if use_hanning:
            signal *= np.hanning(len(signal))

        self.fft = np.fft.rfft(signal)
        self.fftfreq = np.fft.rfftfreq(len(signal), d=self.dt)

        if lowpass_filter:
            self.fft[self.fftfreq < lowpass_filter] = 0

    def computeCepstrum(self):
        assert self.fft is not None, "before calculating the cepstrum, first calculate the FFT"
        epsilon = 1e-9
        self.cepstrum = np.fft.ifft(np.log10(np.abs(self.fft)+epsilon)).real
        return self.cepstrum

    def filterSignal(self, cutoff_frequency=.1):
        b, a = butter(3, cutoff_frequency)
        self.filtered_signal = filtfilt(b, a, self.raw_signal)

    def detectCycles(self, method='peaks', peak='max', use_filtered_signal=True):
        """Detects cycles using different methods.
        
        :param method: 
            method to detect cycles, defaults to 'peaks'

            - *peaks*
                Using a peak finding algorithm on the raw signal
            - *autocorrelation*
                Detects cycles and period using autocorrelation
        :type method: str, optional
        :param use_filtered_signal: uses filtered signal, if available.
        :type use_filtered_signal: bool, optional
        """
        assert method in ('peaks', 'autocorrelation'), "selected method ({}) not available!".format(method)
        assert peak in ('max', 'min'), "selected peak should be either `min` or `max`"

        signal = self.filtered_signal if use_filtered_signal else self.raw_signal
        peak = 0 if peak == 'max' else 1

        if method == 'peaks':
            self.raw_peaks = detectMaximaMinima(signal)
            
        self.Tp = np.diff(self.t[self.raw_peaks[peak]])
        self.A = signal[self.raw_peaks[peak]]

    def detectPhases(self, use_filtered_signal=True):
        """Detects opening and close phase in each cycle.

        :param use_filtered_signal: Event detection on raw (False) or filtered (True) signal.
        :type use_filtered_signal: bool, optional
        """
        assert self.raw_peaks, "first you need to find local minima and maxima (--> detectCycles)"

        signal = self.filtered_signal if use_filtered_signal else self.raw_signal
        self.opening, self.closing = detectOpeningAndClosingEvents(signal, self.raw_peaks[0])

        self.T = np.diff(self.t[self.opening])

        self.t_open, self.t_closed = computeOpenAndClosedIntervals(self.t, self.opening, self.closing)

        self.CO, self.OC = computeOCandCOTransitions(self.t, self.opening, self.closing, self.raw_peaks[0])

    def getPowerSpectrum(self):
        """Returns power spectrum from signal

        :return: Frequencies and Amplitude
        :rtype: tuple(np.ndarray, np.ndarray)
        """        
        return self.fftfreq, np.abs(self.fft)

    def getCepsturm(self):
        """Returns cepstrum from signal

        :return: quefrencies and cepstrum
        :rtype: tuple(np.ndarray, np.ndarray)
        """        
        return self.t[:len(self.cepstrum)], self.cepstrum

class Audio(Signal):
    def __init__(self, raw_signal, dt=1/80000, use_filtered_signal=True, use_hanning=True, debug=False):
        """The **Audio** class handles audio data to compute respective parameters.

        :param raw_signal: the raw audio signal
        :type raw_signal: numpy.ndarray
        :param dt: time between samples in seconds, defaults to 1/80000
        :type dt: float, optional
        :param use_filtered_signal: use filtered signal for computations, defaults to True
        :type use_filtered_signal: bool, optional
        :param use_hanning: use hanning window for FFT, defaults to True
        :type use_hanning: bool, optional
        :param debug: enable debugging mode, defaults to False
        :type debug: bool, optional
        """
        
        super().__init__(raw_signal=raw_signal, dt=dt, debug=debug)
        self.median_signal = None
        self.F0 = None

        self.computeFFT(use_filtered_signal=False, 
            use_hanning=use_hanning)

        self.computeCepstrum()

        self.filterSignal()

        self.detectCycles(use_filtered_signal=use_filtered_signal, 
            peak='max')

    def _A(self, real_zp, use_median_filtered_signal=True):
        A = np.zeros(real_zp.shape[0]-1, dtype=np.float32)
        x = 0
        s = self.median_signal if use_median_filtered_signal else self.raw_signal
        
        # Iterate over cycles identified by Zero-crossing of fundamental frequency
        for i, j in zip(real_zp, real_zp[1:]):
            if i < 0:
                i = 0
            # Compute amplitude (min signal to max signal)
            A[x] = abs(np.max(s[i:j+1])-np.min(s[i:j+1]))
            x += 1
            
        return A

    def detectCycles(self, method='peaks', peak='max', use_filtered_signal=True):
        """Detects cycles using different methods.
         
        :param method: 
            method to detect cycles, defaults to 'peaks'

            - *peaks*
                Using a peak finding algorithm on the raw signal
            - *autocorrelation*
                Detects cycles and period using autocorrelation
        :type method: str, optional
        :param use_filtered_signal: uses filtered signal, if available.
        :type use_filtered_signal: bool, optional
        """
        assert method in ('peaks', 'autocorrelation'), "selected method ({}) not available!".format(method)
        assert peak in ('max', 'min'), "selected peak should be either `min` or `max`"

        signal = self.filtered_signal if use_filtered_signal else self.raw_signal
        peak = 0 if peak == 'max' else 1

        # Identify zero-crossings on (filtered!) signal
        zc = np.diff(np.sign(signal-.5), prepend=np.sign(signal[0]))

        # Identify only cycle starts
        zp = np.where(zc>0)[0]

        # Clean cycle starts
        real_zc = np.zeros_like(zc)
        real_zp = []

        # Iterate over estimated cycle starts
        # Use median filtered signal to improve cycle estimate
        for p in zp:
            # Set current estimate
            cur_p = p
            
            # If half-cycle start, always happening if zc>0
            if zc[p] > 0:    
                while self.median_signal[cur_p] > 0:
                    # move to left
                    cur_p -= 1

            # If half-cycle end, if zc < 0
            if zc[p] < 0:
                while self.median_signal[cur_p] > 0:
                    cur_p += 1
                    
            real_zp.append(cur_p)
            real_zc[cur_p] = zc[p] 

        real_zp = np.asarray(real_zp)

        self.raw_peaks = real_zp

        # Compute amplitudes
        self.A = self._A(real_zp)

        # Compute cycle durations
        self.T = np.diff(self.t[real_zp])
        self.T = self.T[self.T > 0]

    def filterSignal(self, freq_range=20):
        self.F0 = F0fromFFT(self.fft, self.fftfreq)

        # F0 filter signal
        fft_ = self.fft.copy()
        fft_[abs(self.fftfreq) < self.F0-freq_range] = 0
        fft_[abs(self.fftfreq) > self.F0+freq_range] = 0
        self.filtered_signal = np.fft.irfft(fft_)

        # Normalize signal to min/max
        self.filtered_signal = (self.filtered_signal - self.filtered_signal.min()) / (self.filtered_signal.max() - self.filtered_signal.min())

        # Median filter original signal
        self.median_signal = medfilt(self.raw_signal, kernel_size=7)

    def computeParameters(self, use_filtered_signal=False):
        params = {}
        s = self.filtered_signal if use_filtered_signal else self.raw_signal

        params['Mean Jitter'] = meanJitter(self.T)
        params['Jitter%'] = jitterPercent(self.T)
        params['Mean Shimmer'] = meanShimmer(self.A)
        params['Shimmer%'] = shimmerPercent(self.A)

        HNR = harmonicNoiseRatio(s, 1/self.dt)

        if self.debug:
            print(HNR)

        params['HNR'] = HNR[0]

        CPP = cepstralPeakProminence(s, 1/self.dt, plot=self.debug)

        if self.debug:
            print(CPP)

        params['CPP'] = CPP[0]

        params['F0_Cycles'] = F0fromCycles(self.T)[0]
        params['F0_Spectrum'] = F0fromFFT(self.fft, self.fftfreq)
        params['F0_Autocorr'] = HNR[1] # F0 from autocorrelation, maybe direct function?
        params['F0_Cepstrum'] = CPP[2] # F0 from quefrency peak
        params['APF'] = amplitudePerturbationFactor(self.A)
        params['APQ3'] = amplitudePerturbationQuotient(self.A, k=3)
        params['APQ5'] = amplitudePerturbationQuotient(self.A, k=5)
        params['APQ11'] = amplitudePerturbationQuotient(self.A, k=11)

        return params


class GAW(Signal):
    def __init__(self, raw_signal, dt=1/4000, use_filtered_signal=False, \
            use_hanning=True, debug=False):
        """The **GAW** class handles the glottal area waveform (GAW) to compute respective parameters.

        :param raw_signal: raw GAW signal
        :type raw_signal: numpy.ndarray
        :param dt: time between samples in seconds, defaults to 1/4000
        :type dt: float, optional
        :param use_filtered_signal: use filtered signal for computations, defaults to False
        :type use_filtered_signal: bool, optional
        :param use_hanning: use hanning window for FFT, defaults to True
        :type use_hanning: bool, optional
        :param debug: enable debugging mode, defaults to False
        :type debug: bool, optional
        """
        
        super().__init__(raw_signal=raw_signal, dt=dt, debug=debug)
        if use_filtered_signal:
            self.filterSignal(cutoff_frequency=.1)

        self.detectCycles(use_filtered_signal=use_filtered_signal)
        self.detectPhases(use_filtered_signal=use_filtered_signal)
        self.computeFFT(use_filtered_signal=use_filtered_signal, use_hanning=use_hanning)
        self.computeCepstrum()

        self.left_gaw = None
        self.right_gaw = None

    def setLeftRightGAW(self, left_gaw, right_gaw):
        self.left_gaw = left_gaw
        self.right_gaw = right_gaw

    def computeParameters(self):
        params = {}

        params['Mean Jitter'] = meanJitter(self.T)
        params['Jitter%'] = jitterPercent(self.T)
        params['Mean Shimmer'] = meanShimmer(self.A)
        params['Shimmer%'] = shimmerPercent(self.A)

        params['F0_Cycles'] = F0fromCycles(self.T)[0]
        params['F0_Spectrum'] = F0fromFFT(self.fft, self.fftfreq, 75, 500)
        params['F0_Autocorr'] = F0fromAutocorrelation(self.raw_signal)

        params['Open Quotient'] = openQuotient(self.t_open, self.t_closed)
        params['Closing Quotient'] = closingQuotient(self.CO, self.t_open)
        params['Speed Quotient'] = speedQuotient(self.CO, self.OC)
        params['Asymmetry Quotient'] = asymmetryQuotient(self.CO, self.OC)
        params['Rate Quotient'] = rateQuotient(self.CO, self.OC, self.t_closed)
        params['Amplitude Quotient'] = amplitudeQuotient(self.raw_signal, self.opening)
        params['Speed Index'] = speedIndex(self.CO, self.OC, self.t_open)
        params['Glottal Gap Index'] = glottalGapIndex(self.raw_signal, self.opening)
        params['Stiffness'] = stiffness(self.raw_signal, self.opening)

        # Symmetry measures if left and right GAWs are available
        if type(self.left_gaw) != type(None) and type(self.right_gaw) != type(None):
            params['Amplitude Symmetry Index'] = amplitudeSymmetryIndex(self.left_gaw, self.right_gaw, self.opening)
            params['Phase Asymmetry Index'] = phaseAsymmetryIndex(self.left_gaw, self.right_gaw, self.opening)
            
        return params


class AnalysisPlatform(QWidget):
    def __init__(self, raw_signal, dt):
        super().__init__()

        self._logs = []
        self.l = QGridLayout(self)

        ####################
        ### Create Signal
        ####################
        self.signal = Signal(raw_signal=raw_signal, dt=dt)
        self.signalPlot = pg.PlotWidget()
        self.signalCurve1 = self.signalPlot.plot(self.signal.t, self.signal.raw_signal)

        ####################
        ### Filter Signal 
        ####################
        self.signal.filterSignal()
        self.signalCurve2 = self.signalPlot.plot(self.signal.t, 
            self.signal.filtered_signal, 
            pen=pg.mkPen('m'))

        ####################
        ### Detect cycles
        ####################
        self.signal.detectCycles()
        self.signalPlot.plot(self.signal.t[self.signal.raw_peaks[0]], 
            self.signal.filtered_signal[self.signal.raw_peaks[0]],
            pen=pg.mkPen('y'))

        ####################
        ### Detect opening and closing phases
        ####################
        self.signal.detectPhases(use_filtered_signal=False)

        i = 0
        cs = [(241, 196, 15), (231, 76, 60)]

        for o, c in zip(self.signal.opening, self.signal.closing):
            print(o, c)
            i1 = pg.PlotCurveItem(self.signal.t[o:c], np.zeros_like(self.signal.t[o:c]))
            i2 = pg.PlotCurveItem(self.signal.t[o:c], self.signal.raw_signal[o:c])
            between = pg.FillBetweenItem(i1, i2, brush=cs[i % len(cs)])
            self.signalPlot.addItem(between)
            i += 1

        ####################
        ### FFT, Cepstrum
        ####################
        self.l.addWidget(QLabel("Signal"))
        self.l.addWidget(self.signalPlot)

        self.signal.computeFFT(use_hanning=True, use_filtered_signal=True)

        self.l.addWidget(QLabel("Power Spectrum"))
        self.powerSpectrum = pg.PlotWidget()
        self.l.addWidget(self.powerSpectrum)
        self.powerSpectrum.plot(*self.signal.getPowerSpectrum())

        self.signal.computeCepstrum()

        self.l.addWidget(QLabel("Cepstrum"))
        self.Cepstrum = pg.PlotWidget()
        self.l.addWidget(self.Cepstrum)
        self.Cepstrum.plot(*self.signal.getCepsturm())

        #######################
        ### Compute some parameters
        #######################
        print(F0fromCycles(self.signal.T))
        print(jitterPercent(self.signal.T))
        print(shimmerPercent(self.signal.A))  
        print(cepstralPeakProminence(self.signal.raw_signal, 1/self.signal.dt))      

        self.setWindowTitle("Analysis Platform")

    def _log(self, msg):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._logs.append(
            "{}: {}".format(now, msg)
        )


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QInputDialog, QLabel
    from openhsv.gui.table import Table
    app = QApplication([])

    freq, ok = QInputDialog.getInt(QWidget(), 
        "Frequency", 
        "Enter frequency to test:")

    if not ok or freq <= 0:
        freq = 100

    x = np.arange(0, .200, 1/4000)
    y = np.sin(x * 2 * np.pi * freq - np.pi/2)
    y[y<0] = 0 # GAW!
    
    # Compute some parameters for 
    gaw = GAW(y, use_filtered_signal=False)
    params = gaw.computeParameters()
    t = Table(params)
    t.show()

    AP = AnalysisPlatform(y, # Signal
        np.diff(x).mean()) # dt

    AP.show()

    app.exec_()
