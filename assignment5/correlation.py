import numpy as np
from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt

def corr(arr1,arr2,n):
    fft1 = rfft(arr1)
    fft2 = rfft(arr2)
    return irfft(np.conj(fft1)*fft2,n)