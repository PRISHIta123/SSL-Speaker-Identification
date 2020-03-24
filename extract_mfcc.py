import numpy as np
from scipy.signal import spectrogram, hamming
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def LMS(x):
    M=1024
    NFFT=M
    win=hamming(M)
    overlap=0.5
    overlap_samples= int(round(M*overlap))
    f, t, S= spectrogram(x, window=win, nperseg=M, noverlap=overlap_samples, nfft=NFFT)
    avg_S= np.mean(S,axis=1)

    return f, t, S, avg_S


def MFCC(x):
    freq, time, spgram, avg= LMS(x)
    #12 cepstral coefficients are chosen, except the 0th coefficient- as they represent fast changes, do not contribute to ASR
    num_ceps=12
    #Calculate Discrete Cosine Transform of spectrogram
    mfcc= dct(spgram, type=2, axis=1, norm='ortho')[:,1:(num_ceps + 1)]
    return mfcc
