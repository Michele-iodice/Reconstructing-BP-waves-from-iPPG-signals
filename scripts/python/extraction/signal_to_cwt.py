import numpy as np
import pywt
import matplotlib.pyplot as plt


def compute_scales(range_freq, num_scales, fps):
    """
    COMPUTE SCALES
    :return: scales
    """
    sc_min = range_freq[0]
    sc_max = range_freq[1]
    freqs = np.linspace(sc_max, sc_min, num_scales)
    MorletFourierFactor = 4 * np.pi / (6 + np.sqrt(2 + 6 ** 2))
    delta = 1 / fps
    scales = MorletFourierFactor / (freqs * delta)

    return scales


def signal_to_cwt(signal, range_freq:[float], num_scales:int, overlap, norm, recover,fps=100, verbose=False):
    """
    signal: full iPPG or BP signal (sampling frequency=fps)
    overlap: 0 for no overlap; N for an overlap on N samples
    norm: 0 for no standardization (BP); 1 for standardization (iPPG)
    detrend: 0 for no detrending (BP) 1 for detrending (iPPG)
    recover: 0 for no mean recovery (iPPG), 1 to add mean back to CWT (BP)
    fps: sampling frequency of the signal
    """
    if verbose:
        if norm:
            print("-post-filter applied: Standardization")
        if recover:
            print("-post-filter applied: Recovery")

        print("CWT extraction...")

    scales = compute_scales(range_freq, num_scales, fps)

    # OVERLAPPING
    if overlap == 0:
        overlap = 256

    # WINDOWING
    CWT = []
    sig_windows=[]
    windowing = 256
    i = 0
    while (i + windowing-1) < len(signal):
        signal_window = signal[i:i+windowing]

        # Standardization
        if norm:
            signal_window = (signal_window - np.mean(signal_window)) / np.std(signal_window)

        # Compute CWT
        cwt_result, _ = pywt.cwt(signal_window, scales, 'cmor1.5-1.0', sampling_period=1/fps)


        if recover==1:
            cwt_result = cwt_result + np.mean(signal_window)

        cwt_tensor = np.stack([np.real(cwt_result), np.imag(cwt_result)], axis=0)
        CWT.append(cwt_tensor)
        sig_windows.append(signal_window)

        i += overlap

    return CWT, sig_windows


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

