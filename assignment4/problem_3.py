import numpy as np
import matplotlib.pyplot as plt
from problem_1 import shift
from problem_2 import corr

#Defining Gaussians
x = np.linspace(-5,5,100)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

#Defining a routine to take the correlation with an arbitrary shift
def corshift(arr,x):
    arrshift = shift(arr,x)
    return corr(arr,arrshift)

plt.plot(x,corshift(y,len(y)/10))
plt.legend()
plt.show()