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


def inverse_cwt(CWT, f_min=0.6, f_max=4.5, num_scales=256, C_psi=0.776, fps=100, recover=False):
    """
    Approximate the inverse CWT using a summation over scales and time.

    CWT: Coefficients of the Continuous Wavelet Transform.
    f_min: Minimum frequency of scales.
    f_max: Maximum frequency of scales.
    scales: Scales used in the CWT.
    time: Array of time points corresponding to the original signal.
    wavelet_function: The mother wavelet function psi(t).
    C_psi: The admissibility constant C_psi.
    """

    range_freq = [f_min, f_max]
    real_part = CWT[0]
    imag_part = CWT[1]
    scales = compute_scales(range_freq, num_scales, fps)
    coeffs = real_part + 1j * imag_part

    num_scales, num_samples = coeffs.shape
    time = np.arange(num_samples) / fps

    psi, x = pywt.ContinuousWavelet('cmor1.5-1.0').wavefun(level=10)

    reconstructed = np.zeros(num_samples, dtype=np.float64)

    for idx, scale in enumerate(scales):
        for tau in range(num_samples):
            t_scaled = (time - time[tau]) / scale
            wavelet_val = np.interp(t_scaled, x, np.real(psi), left=0, right=0)
            contribution = (coeffs[idx, tau] * wavelet_val) / (scale ** 1.5)
            reconstructed += np.real(contribution)

    reconstructed /= C_psi
    if recover:
        reconstructed = reconstructed + np.mean(real_part)
    return reconstructed


def plotCWT(cwt_sig,fps=100):
    scales = compute_scales([0.6,4.5],256,fps)
    cwt_complex = cwt_sig[0] + 1j * cwt_sig[1]
    power = np.abs(cwt_complex) ** 2

    time = np.arange(cwt_complex.shape[1]) / fps
    freqs = pywt.scale2frequency('cmor1.5-1.0', scales) * fps

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(time, freqs, power, shading='auto', cmap='jet')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('CWT Scalogram')
    plt.colorbar(label='Power')
    plt.tight_layout()
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

