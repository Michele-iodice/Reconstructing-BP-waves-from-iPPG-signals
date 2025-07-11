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
    f_min, f_max = range_freq
    frequencies = np.linspace(f_min, f_max, num_scales)
    scales = pywt.central_frequency('cmor1.5-1.0') * fps / frequencies
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
    range_freq: range of frequencies to use to compute the scales
    num_scales: number of scales to use
    fps: sampling frequency of the signal
    nan_threshold: threshold to use for limit the nan values
    """
    if verbose:
        print("CWT extraction...")

    scales = compute_scales(range_freq, num_scales, fps)

    # WINDOWING
    CWT = []
    sig_windows=[]
    i = 0
    for signal_window in signal:

        if signal_window.ndim == 2:
            signal_window = np.ravel(signal_window)

        if signal_window.shape[0] == 0:
            if verbose:
                print(f"DISCARDED: Signal with NaN/Inf at index {i}")
            continue

        frac_nan = np.sum(np.isnan(signal_window)) / len(signal_window)
        if frac_nan >= nan_threshold:
            if verbose:
                print(f"Discarded window (std ~0): index {i}")
            continue

        # SIGNAL CLEANING
        if np.any(np.isnan(signal_window)):
            signal_window = smart_interpolation(signal_window)

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


def inverse_cwt(CWT, f_min=0.6, f_max=4.5, num_scales=256, C_psi=0.776, fps=100):
    """
    Approximate the inverse CWT using a complex Morlet wavelet cmor1.5-1.0.

    :ARGS: CWT: Coefficients of the Continuous Wavelet Transform.
           f_min: Minimum frequency of scales.
           f_max: Maximum frequency of scales.
           num_scales: number of Scales used in the CWT.
           C_psi: The admissibility constant C_psi.
           fps: Frequency of the wavelet.
    :return: reconstructed BP signal.
    """

    range_freq = [f_min, f_max]
    real_part = CWT[0]
    imag_part = CWT[1]
    scales = compute_scales(range_freq, num_scales, fps)
    coeffs = real_part + 1j * imag_part
    num_scales, num_samples = coeffs.shape
    dt = 1.0 / fps
    time = np.arange(num_samples) * dt

    wavelet = pywt.ContinuousWavelet('cmor1.5-1.0')
    psi, t_psi = wavelet.wavefun(level=10)

    wavelet_peak_idx = np.argmax(np.abs(psi))
    shift_wavelet = t_psi[wavelet_peak_idx]

    reconstructed = np.zeros(num_samples, dtype=np.float64)

    for i, scale in enumerate(scales):
        t_scaled = t_psi * scale
        psi_scaled = psi / np.sqrt(scale)
        interp_wavelet = interp1d(t_scaled, np.real(psi_scaled), bounds_error=False, fill_value=0.0)

        temp_sum = np.zeros(num_samples, dtype=np.float64)

        for b in range(num_samples):
            shifted_time = time - time[b] + shift_wavelet * scale
            wavelet_vals = interp_wavelet(shifted_time)
            # Correzione: divisione per scale^2 come da formula
            temp_sum += (np.real(coeffs[i, b]) * wavelet_vals) / (scale ** 2)

        reconstructed += temp_sum

    reconstructed *= dt / C_psi

    return reconstructed


def plotCWT(cwt_sig,fps=100, title="X"):
    scales = compute_scales([0.6,4.5],256,fps)
    cwt_complex = cwt_sig[0] + 1j * cwt_sig[1]
    power = np.abs(cwt_complex) ** 2

    time = np.arange(cwt_complex.shape[1]) / fps
    freqs = pywt.scale2frequency('cmor1.5-1.0', scales) * fps

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(time, freqs, power, shading='auto', cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    s="CWT Scalogram" + title
    plt.title(s)
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

