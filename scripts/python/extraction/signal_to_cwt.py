import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pykalman import KalmanFilter


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

def spline_interpolation(x):
    nans = np.isnan(x)
    if np.all(nans): return x
    f = interp1d(np.flatnonzero(~nans), x[~nans], kind='cubic', fill_value="extrapolate")
    x[nans] = f(np.flatnonzero(nans))
    return x

def kalman_interpolation(x):
    nans = np.isnan(x)
    x[nans] = 0  # inizializza
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=x[~nans][0],
                      observation_covariance=1,
                      transition_covariance=0.01)
    x_smoothed, _ = kf.em(x).smooth(x)
    return x_smoothed

def smart_interpolation(x):
    n_nan = np.sum(np.isnan(x))
    frac_nan = n_nan / len(x)
    if n_nan == 0:
        return x
    elif frac_nan < 0.1:
        return spline_interpolation(x)
    else:
        return kalman_interpolation(x)


def signal_to_cwt(signal, range_freq:[float], num_scales:int, fps=100, nan_threshold=0.3, verbose=False):
    """
    signal: full iPPG or BP signal (sampling frequency=fps)
    overlap: 0 for no overlap; N for an overlap on N samples
    norm: 0 for no standardization (BP); 1 for standardization (iPPG)
    detrend: 0 for no detrending (BP) 1 for detrending (iPPG)
    recover: 0 for no mean recovery (iPPG), 1 to add mean back to CWT (BP)
    fps: sampling frequency of the signal
    """
    if verbose:
        print("-post-filter applied: Standardization")
        print("CWT extraction...")

    scales = compute_scales(range_freq, num_scales, fps)

    # WINDOWING
    CWT = []
    sig_windows=[]
    i = 0
    for signal_window in signal:

        if signal_window.shape[0] == 0 or signal_window.shape[1] <= 1:
            continue

        frac_nan = np.sum(np.isnan(signal_window)) / len(signal_window)
        if frac_nan >= nan_threshold:
            if verbose:
                print(f"Discarded window (std ~0): index {i}")
            continue

        # SIGNAL CLEANING
        if np.any(np.isnan(signal_window)):
            signal_window = smart_interpolation(signal_window)

        # Standardization
        mean = np.mean(signal_window)
        std = np.std(signal_window)
        if std < 1e-6:
            std = 1e-6

        signal_window = (signal_window - mean) / std

        # Compute CWT
        cwt_result, _ = pywt.cwt(signal_window, scales, 'cmor1.5-1.0', sampling_period=1/fps)

        if np.any(np.isnan(cwt_result)) or np.any(np.isinf(cwt_result)):
            if verbose:
                print(f"DISCARDED: CWT produced NaN/Inf at index {i}")
            continue

        cwt_tensor = np.stack([np.real(cwt_result), np.imag(cwt_result)], axis=0)
        CWT.append(cwt_tensor)
        sig_windows.append(signal_window)

        i=i+1

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

    # params
    delta = 1 / fps
    range_freq = [f_min, f_max]
    real_part = CWT[0]
    imag_part = CWT[1]
    scales = compute_scales(range_freq, num_scales, fps)
    coeffs = real_part + 1j * imag_part

    num_scales, num_samples = coeffs.shape
    time = np.arange(num_samples) * delta
    dt = delta

    wavelet = pywt.ContinuousWavelet('cmor1.5-1.0')
    psi, x = wavelet.wavefun(level=10)

    reconstructed = np.zeros(num_samples, dtype=np.float64)

    for idx, scale in enumerate(scales):
        t_scaled = x * scale
        psi_scaled = psi / np.sqrt(scale)

        wavelet_vals = np.interp(time[:, None] - time, t_scaled, np.real(psi_scaled), left=0, right=0)
        contributions = np.real(coeffs[idx, :] @ wavelet_vals.T) / (scale ** 1.5)
        reconstructed += contributions

    reconstructed *= dt / C_psi

    if recover:
        reconstructed += np.mean(real_part)

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

