import numpy as np
from scipy.fft import fft


def get_stft_columns(x, nx, n_win, noverlap, fs):
    hop_size = int(n_win - noverlap)
    n_col = int((nx - noverlap) / hop_size)
    xin = np.zeros((n_win, n_col)).astype(object)
    t = np.zeros(n_col)
    for iCol in range(0, n_col):
        x = (np.array(x))
        start = int(1 + hop_size * iCol) - 1
        end = int(n_win + hop_size * iCol)
        xin[:, iCol] = np.round_(x[start: end].flatten(), decimals=15)
        t[iCol] = ((iCol * hop_size)+(n_win/2))/fs
    return [xin, t]


def custom_frequency_grid(fs, npts):
    freq_res = fs / npts
    w = np.ones(npts)
    for iCol in range(0, npts):
        w[iCol] = iCol * freq_res
    nyq = fs / 2
    half_npts = int(npts / 2) + 1
    w[half_npts] = nyq
    w[npts-1] = fs - freq_res
    w = np.transpose(w[0:half_npts])
    return w


def custom_format_spectrogram(f_out, y):
    return y[0:len(f_out), :]


def get_spectrogram_options(song_data, window, NOverlap, nfft):
    xw = np.transpose(song_data)
    nx = len(xw)
    nwin = len(window)
    Fs = 8000
    [xin, t] = get_stft_columns(xw, nx, nwin, NOverlap, Fs)
    y1 = np.multiply(xin.astype(float), (window[:, None]).astype(float))
    S = fft(y1.astype(float), nfft,  axis=0, norm='backward')
    npts = 512
    f = custom_frequency_grid(Fs, npts)
    S = custom_format_spectrogram(f, S)
    return [S, f, t]


def custom_spectrogram(song_data, window, NOverlap, nfft):
    return get_spectrogram_options(song_data, window, NOverlap, nfft)
