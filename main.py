import librosa
import librosa.display
from librosa.display import specshow
import matplotlib.pyplot as plt
import scipy
import sys
from numpy.linalg import norm
from dtw import dtw

template_m = "002_M.wav"
template_k = "001_K.wav"

def getMFCC(path):
    samples, sample_rate = librosa.load(path)
    return librosa.feature.mfcc(samples, n_mfcc=12, hop_length=windowMFCC,n_fft = windowFFT)

def diff(path1, path2):
    mfcc1 = getMFCC(path1)
    mfcc2 = getMFCC(path2)
    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    return dist

def detectGender(path):
    diffm = diff(template_m, path)
    diffk = diff(template_k, path)
    if diffm > diffk: return 'Kobieta'
    else: return 'Mezczyzna'

windowMFCC = 512
windowFFT = 512

for i in range(1, len(sys.argv)): 
    path = sys.argv[i]
    print(path)
    samples, sample_rate = librosa.load(path)
    label = path + ' ' +  detectGender(path)
    ax = plt.subplot(1, len(sys.argv) - 1, i)
    ax.title.set_text(label)
    mfcc1 = librosa.feature.mfcc(samples, n_mfcc=12, hop_length=windowMFCC,n_fft = windowFFT)
    specshow(mfcc1)
plt.show()