import numpy as np
from scipy import interpolate, sparse
import pywt
import matplotlib.pyplot as plt


def compute_scales():
    """
    COMPUTE SCALES
    :return: scales
    """
    sc_min = -1
    sc_max = -1
    sc = np.arange(0.2, 1000.01, 0.01)
    MorletFourierFactor = 4 * np.pi / (6 + np.sqrt(2 + 6 ** 2))
    freqs = 1 / (sc * MorletFourierFactor)
    for dummy in range(len(freqs)):
        if freqs[dummy] < 0.6 and sc_max == -1:
            sc_max = sc[dummy]
        elif freqs[dummy] < 8 and sc_min == -1:
            sc_min = sc[dummy]

    sc = np.array([sc_min, sc_max])
    scales = np.arange(sc[0], sc[1], 0.00555)

    return scales


def signal_to_cwt(signal, overlap, norm, detrend, recover, fps):
    """
    signal: full iPPG or BP signal (sampling frequency=fps)
    overlap: 0 for no overlap; N for an overlap on N samples
    norm: 0 for no standardization (BP); 1 for standardization (iPPG)
    detrend: 0 for no detrending (BP) 1 for detrending (iPPG)
    recover: 0 for no mean recovery (iPPG), 1 to add mean back to CWT (BP)
    fps: sampling frequency of the signal
    """

    # RESAMPLING (100 Hz)
    time = np.arange(0, len(signal)/fps, 1/100)
    scales = compute_scales()
    x = np.arange(0, len(signal) / fps, 1 / fps)
    interp_func = interpolate.interp1d(x, signal, kind='linear')
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
        cwt_result, _ = pywt.cwt(signal_window, scales, 'cmor', sampling_period=1/fps)

        if recover==1:
            cwt_result = cwt_result + np.mean(signal_window)
        CWT.append(cwt_result)

        i += overlap

    return CWT, scales


def inverse_cwt(CWT, fps):
    """
    Approximate the inverse CWT using a summation over scales and time.

    CWT: Coefficients of the Continuous Wavelet Transform
    scales: Scales used in the CWT
    time: Array of time points corresponding to the original signal
    wavelet_function: The mother wavelet function psi(t)
    C_psi: The admissibility constant C_psi
    """

    time = np.arange(0, len(CWT) / fps, 1 / 100)
    wavelet = pywt.ContinuousWavelet('cmor', dtype='float64')
    wavelet_function, _ = wavelet.wavefun(level=10)  # morlet function psi(t)
    C_psi = 0.776  # approximation of C_psi for cmor wavelet
    scales = compute_scales()
    reconstructed_signal = np.zeros(len(time))

    # Loop over each scale
    for idx, scale in enumerate(scales):
        for tau in range(CWT.shape[1]):
            # Compute the contribution of each coefficient CWT_x(tau, scale)
            wavelet_contribution = wavelet_function((time - tau) / scale) / np.sqrt(np.abs(scale))
            reconstructed_signal += (CWT[idx, tau] * wavelet_contribution) / (scale ** 2)

    # Normalize by the admissibility constant C_psi
    reconstructed_signal /= C_psi

    return reconstructed_signal


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


def plotComparison(original_signal, reconstructed_signal):
    time_len= max(len(original_signal[0][0][0]), len(reconstructed_signal[0][0][0]))
    time = np.linspace(0, 2.5, time_len)
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_signal, label='Original Signal', linestyle='--', color='red')
    plt.plot(time, reconstructed_signal, label='Reconstructed Signal', color='blue')
    plt.legend()
    plt.show()

# Example usage
# signal = np.random.randn(1000)
# CWT, scales = signal_to_cwt(signal, overlap=256, norm=0, detrend=0, fps=100, recover=0)
# print(np.array(CWT).shape)
# plotCWT(CWT)
# Calcola il segnale ricostruito
# reconstructed_signal = inverse_cwt(CWT, fps=100)
# plotComparison(CWT, reconstructed_signal, time)

