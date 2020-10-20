import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt

def corr(arr1,arr2):
    fft1 = fft(arr1)
    fft2 = fft(arr2)
    return ifft(fft1*np.conj(fft2))

x = np.linspace(-5,5,100)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2) #We define a Gaussian centered at 0 with sigma=1.

plt.plot(x,np.real(corr(y,y)),color="teal",label="Correlation function")
plt.xlabel("x",fontsize=14)
plt.ylabel("Correlation Function",fontsize=14)
plt.savefig("problem_2.png",bbox_inches="tight",dpi=500)
plt.show()