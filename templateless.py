import librosa
import librosa.display
from librosa.display import specshow
import matplotlib.pyplot as plt
import scipy
from scipy.signal import blackmanharris, correlate
import numpy as np
from numpy import argmax, mean, diff, log, nonzero
from numpy.fft import rfft
import sys
from numpy.linalg import norm
from dtw import dtw

template_m = "002_M.wav"
template_k = "001_K.wav"

def load(path):
    sig, fs = librosa.load(path)
    sig = filterFreq(sig)
    return (sig, fs)

def filterFreq(sig):
    freqs = np.abs(np.fft.rfft(sig))
    freqs = [f for index, f in enumerate(freqs) if index < 280]
    return np.fft.irfft(freqs)

def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def funfreq(sig, fs):
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start - 1
    px, py = parabolic(corr, peak)

    return fs / px

def calcMeanFunFreq(sig, fs):
    split = np.split(sig, 50)
    freqs = [funfreq(s, fs) for s in split]
    return np.mean(freqs)

def calcMeanfun(sig, fs):
    freqs = funfreq(sig, fs)
    #filtered = [f for f in freqs if f <= 280]
    return np.mean(freqs)

def calcIqr(sig, fs):
    spec = np.abs(np.fft.rfft(sig))
    freq = np.fft.rfftfreq(len(sig), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    amp_cumsum = np.cumsum(amp)
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    return IQR


def detectGender(path):
    sig, fs = load(path)
    meanfun = calcMeanfun(sig, fs)
    iqr = calcIqr(sig, fs)
    print("Meanfun: " + str(meanfun));
    print("IQR: " + str(iqr));
    if meanfun > 140:
        if iqr > 700:
            return 'M'
        else:
            return 'K'
    else:
        return 'K'

windowMFCC = 512
windowFFT = 512

for i in range(1, len(sys.argv)): 
    path = sys.argv[i]
    label = path + ' ' +  detectGender(path)
    print(label)