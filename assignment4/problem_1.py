import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt

#Defining the Gaussian with mean=0 and sigma=1
x = np.linspace(-5,5,100)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

#A shit in a fourier transform can be thought of as a adding a phase term (shift theorem)
def shift(arr,delx): #arr is the initial array and x is the x-amount to be shifted by
    f = fft(arr)
    N = len(arr)
    k = np.linspace(0,N-1,N)
    exp = np.exp(-2*complex(0,1)*np.pi*k*delx/N)
    return np.real(ifft(f*exp))

plt.plot(x,y,"k-",label="Original Gaussian")
plt.plot(x,shift(y,len(y)/2),"r-",label="Shifted Gaussian")
plt.xlabel("x",fontsize=14)
plt.ylabel("y",fontsize=14)
plt.legend(fontsize=11)
plt.savefig("problem_1.png",bbox_inches="tight",dpi=500)
plt.show()