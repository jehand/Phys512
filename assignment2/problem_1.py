import numpy as np
from integrator import integrate_step #Old integrator
from integrator import integrator #New integrator


#Let's write out some common functions that we can integrate
def Gauss(x, u=0, sig=1):
    return (1/np.sqrt(2*np.pi*sig**2))*np.exp((-(x-u)**2)/(2*sig**2))

def quad(x):
    return x**2 + 2*x + 1

def log(x):
    return np.log(x)

#Just a check to see if the function is integrating correctly
print(integrate_step(Gauss,1,5,0.01), integrator(Gauss,1,5,0.01))