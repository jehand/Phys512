import numpy as np
from numpy.fft import fft,ifft

def conv(arr1,arr2):
    f1 = fft(arr1)
    f2 = fft(arr2)
    return np.real(ifft(f1*f2))

