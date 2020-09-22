import numpy as np
from integrator import integrate_step #Old integrator
from integrator import integrator #New integrator

#Let's write out some common functions that we can integrate
def Gauss(x, u=0, sig=1):
    return (1/np.sqrt(2*np.pi*sig**2))*np.exp((-(x-u)**2)/(2*sig**2))

def sqrt(x):
    return np.sqrt(x)

def abs(x):
    return np.abs(x)

def log(x):
    return x*np.log(x)

def semi_circle(x,r=1):
    return np.sqrt(r-x**2)

#Comparing all these functions
print("Gaussian number of reduced calls =", integrate_step(Gauss,-5,5,0.001)[1]-integrator(Gauss,-5,5,0.001)[1])
print("√x number of reduced calls =", integrate_step(sqrt,0,5,0.001)[1]-integrator(sqrt,0,5,0.001)[1])
print("|x| number of reduced calls =", integrate_step(sqrt,0,2,0.001)[1]-integrator(sqrt,0,2,0.001)[1])
print("xln(x) number of reduced calls =", integrate_step(log,0.5,5,0.001)[1]-integrator(log,0.5,5,0.001)[1])
print("Semi-circle of radius 1 number of reduced calls =", integrate_step(semi_circle,-1,1,0.001)[1]-integrator(semi_circle,-1,1,0.001)[1])