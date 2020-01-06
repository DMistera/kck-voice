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
import os.path
from numpy.linalg import norm
from dtw import dtw

def load(path):
    sig, fs = librosa.load(path)
    #sig = filterFreq(sig)
    return (sig, fs)

def filterFreq(sig):
    freqs = np.abs(np.fft.rfft(sig))
    freqs = [f for index, f in enumerate(freqs) if index < 280]
    return np.fft.irfft(freqs)

def parabolic(f, x):
    try:
        xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
        yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
        return (xv, yv)
    except:
        return (x, f)

def funfreq(sig, fs):
    # Calculate autocorrelation and throw away the negative lags
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def calcMeanFunFreq(sig, fs):
    # split = np.array_split(sig, len(sig)*fs/100)
    # freqs = [funfreq(s, fs) for s in split]
    # return np.mean(freqs)
    return funfreq(sig, fs)

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
    meanfun = calcMeanFunFreq(sig, fs)
    iqr = calcIqr(sig, fs)
    if meanfun > 160:
        return 'K'
    else:
        if iqr > 700:
            return 'M'
        else:
            return 'K'

windowMFCC = 512
windowFFT = 512

if(len(sys.argv) > 1):
    print(detectGender(sys.argv[1]))
else:
    #Configurable
    startIndex = 5
    endIndex = 93

    success = 0
    total = 0
    fail_m = 0
    fail_k = 0
    for i in range(startIndex, endIndex):
        prefix = str(i).zfill(3) + '_'
        if os.path.isfile(prefix + 'M.wav'):
            result = detectGender(prefix + 'M.wav')
            if result == 'M': success += 1
            else: fail_m += 1
        else:
            result = detectGender(prefix + 'K.wav')
            if result == 'K': success += 1
            else: fail_k += 1
        total += 1
        print('Detected: ' + result + ' Total: ' + str(total) + ' Success: ' + str(success) + ' Accuracy: ' + str(success/total))
    print(str(fail_m) + ' men detected as women')
    print(str(fail_k) + ' women detected as man')