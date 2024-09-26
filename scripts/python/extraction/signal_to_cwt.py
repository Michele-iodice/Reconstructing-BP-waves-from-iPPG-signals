import numpy as np
from scipy import interpolate, sparse
import pywt
import matplotlib.pyplot as plt


def signal_to_cwt(signal, overlap, norm, detrend, fps):
    """
    signal: full iPPG or BP signal (sampling frequency=fps)
    overlap: 0 for no overlap; N for an overlap on N samples
    norm: 0 for no standardization (BP); 1 for standardization (iPPG)
    detrend: 0 for no detrending (BP) 1 for detrending (iPPG)
    fps: sampling frequency of the signal
    """

    # COMPUTE SCALES
    sc_min = -1
    sc_max = -1
    sc = np.arange(0.2, 1000.01, 0.01)
    MorletFourierFactor = 4 * np.pi / (6 + np.sqrt(2 + 6**2))
    freqs = 1 / (sc * MorletFourierFactor)
    for dummy in range(len(freqs)):
        if freqs[dummy] < 0.6 and sc_max == -1:
            sc_max = sc[dummy]
        elif freqs[dummy] < 8 and sc_min == -1:
            sc_min = sc[dummy]
    sc = np.array([sc_min, sc_max])

    # RESAMPLING (100 Hz)
    time = np.arange(0, len(signal)/fps, 1/100)
    interp_func = interpolate.interp1d(np.arange(0, len(signal)/fps, 1/fps), signal, kind='linear')
    signal = interp_func(time)
    fps = 100

    # DETRENDING (Tarvainen et al., 2002)
    if detrend:
        lambda_ = 470
        T = len(signal)
        I = sparse.eye(T)
        D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T)).toarray()
        signal = (I - np.linalg.inv(I + lambda_**2 * D2.T @ D2) @ signal)

    # OVERLAPPING
    if overlap == 0:
        overlap = 256

    # WINDOWING
    CWT = []
    i = 0
    while (i + 255) < len(signal):
        signal_window = signal[i:i+256]
        time_window = time[i:i+256]

        # Standardization
        if norm:
            signal_window = (signal_window - np.mean(signal_window)) / np.std(signal_window)

        # Compute CWT
        scales = np.arange(sc[0], sc[1], 0.00555)
        cwt_result, _ = pywt.cwt(signal_window, scales, 'cmor', sampling_period=1/fps)
        CWT.append(cwt_result)

        i += overlap

    return CWT


def plotCWT(cwt_sig):
    joy = np.linspace(1, 5, len(cwt_sig[0][0]))
    time = np.linspace(0, 2.5, len(cwt_sig[0][0][0]))
    scalogram = np.mean(cwt_sig, axis=0)
    scalogram = np.mean(scalogram, axis=0)
    scalogram_real = np.abs(scalogram)
    plt.figure(figsize=(10, 5))
    plt.imshow(scalogram_real, aspect='auto', extent=[time.min(), time.max(), joy.min(), joy.max()],
               origin='lower', cmap='jet')
    plt.title('CWT signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Amplitude')
    plt.show()


# Example usage
# signal = np.random.randn(1000)
# CWT = signal_to_cwt(signal, overlap=0, norm=1, detrend=1, fps=100)
# plotCWT(CWT)

