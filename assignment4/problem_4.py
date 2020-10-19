import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt

#To get past the wrapping around nature of a Fourier transform, we use zero padding.
def conv(arr1,arr2):
    arr1pad, arr2pad = np.pad(arr1,[0,len(arr1)]), np.pad(arr2,[0,len(arr2)])
    f1 = fft(arr1pad)
    f2 = fft(arr2pad)
    transform = ifft(f1*f2)
    return transform[:len(arr1)]
