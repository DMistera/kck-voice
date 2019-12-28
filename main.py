import librosa
import librosa.display
from librosa.display import specshow
import matplotlib.pyplot as plt
import scipy
import sys
from numpy.linalg import norm
from dtw import dtw
import os.path

template_m = ["002_M.wav", "004_M.wav"]
template_k = ["001_K.wav", "003_K.wav"]

def getMFCC(path):
    samples, sample_rate = librosa.load(path)
    return librosa.feature.mfcc(samples, n_mfcc=12, hop_length=windowMFCC,n_fft = windowFFT)

def diff(path1, path2):
    mfcc1 = getMFCC(path1)
    mfcc2 = getMFCC(path2)
    return dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))[0]

def detectGender(path):
    minDiff = 99999999
    for tm in template_m:
        diffm = diff(tm, path)
        if diffm < minDiff:
            minDiff = diffm
            result = 'M'
    for tk in template_k:
        diffk = diff(tk, path)
        if diffk < minDiff:
            minDiff = diffk
            result = 'K'
    return result


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
    for i in range(startIndex, endIndex):
        prefix = str(i).zfill(3) + '_'
        if os.path.isfile(prefix + 'M.wav'):
            result = detectGender(prefix + 'M.wav')
            if result == 'M': success += 1
        else:
            result = detectGender(prefix + 'K.wav')
            if result == 'K': success += 1
        total += 1
        print('Detected: ' + result + ' Total: ' + str(total) + ' Success: ' + str(success) + ' Accuracy: ' + str(success/total))

for i in range(1, len(sys.argv)): 
    path = sys.argv[i]
    label = path + ' ' +  detectGender(path)
    print(label)