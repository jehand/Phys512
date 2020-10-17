import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt

#Defining the Gaussian with mean=0 and sigma=1
x = np.linspace(-5,5,100)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

#A shit in a fourier transform can be thought of as a adding a phase to each term
def shift(arr,x): #arr is the initial array and x is the x-amount to be shifted by
    f = fft(arr)
    n = len(arr)
    vec = np.linspace(0,n-1,n)
    exp = np.exp(-2*complex(0,1)*np.pi*vec*x/n)
    return np.real(ifft(f*exp))

plt.plot(x,y,label="Original Gaussian")
plt.plot(x,shift(y,len(y)/2),label="Shifted Gaussian")
plt.legend()
plt.show()