import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

#We arbitrarly choose a non-integer sine wave of w=0.7
w = 0.8
x = np.linspace(-10,10,100)
y = np.sin(w*x)

#Determining the fft from numpy
fftsine = fft(y)
#plt.plot(np.abs(fftsine))

#Analytic estimate of the dft
N = len(y)
k = 0.2
analytic = (1-np.exp(-2*np.pi*complex(0,1)*k))/(1-np.exp(-2*np.pi*complex(0,1)*k/N))
plt.plot(np.abs(y*analytic))
plt.show()