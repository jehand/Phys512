import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt

def corr(arr1,arr2):
    fft1 = fft(arr1)
    fft2 = fft(arr2)
    return ifft(fft1*np.conj(fft2))