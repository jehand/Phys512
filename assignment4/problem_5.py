import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

#We arbitrarly choose a non-integer sine wave of f=0.8
f = 0.8
x = np.linspace(-5,5,1000)
y = np.sin(2*np.pi*f*x)
print(y)

#Determining the fft from numpy
fftsine = fft(y)
plt.plot(np.abs(fftsine))

#Analytic estimate of the dft
N = len(y)
k = f
analytic = (1-np.exp(-2*np.pi*complex(0,1)*k))/(1-np.exp(-2*np.pi*complex(0,1)*k/N))
plt.plot(np.abs(y*analytic))
plt.show()

#Windowing: choose the window function to be 0.5-0.5cos(2πx/N)
ywind = 0.5-0.5*np.cos(2*np.pi*x/N)
ynew = y*ywind
fftnew = fft(ynew)
plt.plot(np.abs(fftnew))
plt.show()

#Fourier transform of the window
ywindfft = fft(ywind)
print("Showing the Fourier Transform of the Window=",np.abs(ywindfft)/N)
