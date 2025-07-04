import numpy as np
from scipy.signal import stft
import plotly.graph_objects as go
import pywt
from extraction.signal_to_cwt import compute_scales
from importlib import import_module
from PPG.filters import *
from scipy.signal import resample


class BPsignal:
    """
    Manage (multi-channel, row-wise) BP signals.
    """
    # nFFT = 2048  # freq. resolution for STFTs
    step = 1       # step in seconds
    renderer = 'colab'  # or 'notebook'

    def __init__(self, data, fs, startTime=0, minHz=0.75, maxHz=4., verb=False):
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fs = fs                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz
        nyquistF = self.fs/2
        fRes = 0.5
        self.nFFT = max(2048, (60*2*nyquistF) / fRes)
        self.spect = None
        self.freqs = None
        self.times = None
        self.bpm = None

    def getSig(self):
        """
        :return: the ground truth Blood Pressure (BP) signal
        """
        return self.data

    def getSigValue(self):
        """
        :return: a single value for the ground truth Blood Pressure (BP) signal
        """
        mean = np.mean(self.data[0])
        return mean

    def getSigFps(self):
        """
        :return: the fps of the ground truth BP signal
        """
        return self.fs

    def spectrogram(self, winsize=5):
        """
        Compute the BP signal spectrogram restricted to the
        band 42-240 BPM by using winsize (in sec) samples.
        """

        # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=self.fs*winsize,
                       noverlap=self.fs*(winsize-self.step),
                       boundary='even',
                       nfft=self.nFFT)
        Z = np.squeeze(Z, axis=0)

        # -- freq subband (0.65 Hz - 4.0 Hz)
        minHz = 0.65
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        self.spect = np.abs(Z[band, :])     # spectrum magnitude
        self.freqs = 60*F[band]            # spectrum freq in bpm
        self.times = T                     # spectrum times

        # -- BPM estimate by spectrum
        self.bpm = self.freqs[np.argmax(self.spect, axis=0)]

    def displaySpectrum(self, display=False, dims=3):
        """Show the spectrogram of the BP signal"""

        # -- check if bpm exists
        try:
            bpm = self.bpm
        except AttributeError:
            self.spectrogram()
            bpm = self.bpm

        t = self.times
        f = self.freqs
        S = self.spect

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=S, x=t, y=f, colorscale="viridis"))
        fig.add_trace(go.Scatter(
            x=t, y=bpm, name='Frequency Domain', line=dict(color='red', width=2)))

        fig.update_layout(autosize=False, height=420, showlegend=True,
                          title='Spectrogram of the BVP signal',
                          xaxis_title='Time (sec)',
                          yaxis_title='BPM (60*Hz)',
                          legend=dict(
                              x=0,
                              y=1,
                              traceorder="normal",
                              font=dict(
                                family="sans-serif",
                                size=12,
                                color="black"),
                              bgcolor="LightSteelBlue",
                              bordercolor="Black",
                              borderwidth=2)
                          )

        fig.show(renderer=self.renderer)

    def getCWT(self, sig, range_freq:[float], num_scales:int, winsize, overlap=0, fps=100):
        sig = np.copy(sig)

        scales = compute_scales(range_freq, num_scales, fps)

        # OVERLAPPING
        if overlap == 0:
            overlap = winsize
        overlap = int(overlap * fps)

        # WINDOWING
        CWT = []
        sig_windows = []
        windowing = round(winsize * fps)
        i = 0
        while (i + windowing - 1) < len(sig):
            signal_window = sig[i:i + windowing]


            # Compute CWT
            cwt_result, _ = pywt.cwt(signal_window, scales, 'cmor1.5-1.0', sampling_period=1 / fps)

            cwt_result = cwt_result + np.mean(signal_window)

            real = np.real(cwt_result)
            imag = np.imag(cwt_result)

            n_resample= round(winsize * 100)
            real_resampled = resample(real, n_resample, axis=1)
            imag_resampled = resample(imag, n_resample, axis=1)

            cwt_tensor = np.stack([real_resampled, imag_resampled], axis=0)
            CWT.append(cwt_tensor)
            sig_windows.append(signal_window)

            i += overlap

        return CWT, sig_windows
