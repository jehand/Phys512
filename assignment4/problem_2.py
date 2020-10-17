import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt

def corr(arr1,arr2):
    fft1 = fft(arr1)
    fft2 = fft(arr2)
    return np.real(ifft(fft1*np.conj(fft2)))

x = np.linspace(-5,5,100)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2) #We define a Gaussian centered at 0 with sigma=1.

#plt.plot(x,y,label="Original Gaussian")
plt.plot(x,corr(y,y),label="Correlation function")
plt.legend()
plt.show()