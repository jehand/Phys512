import numpy as np

data = np.loadtxt("dish_zenith.txt") #Importing data

"""
We now carry out our fit. First, we define our predicted function with our new parameters. 
"""

def para(x,y,x0,y0,z0,a,r):
    return a(x**2+y**2-x0*x-y0*x+r)+z0 #returning z-values

def linear(d,pars,N):
    vec = d[:,:2] #We want to just use the x,y columns.
    Ninv = np.linalg.inv(N) #Calculating the inverse to use later
    A = np.empty(shape=(len(data), len(pars)))
    A[:,0] = 1.0
    for i in range(1,len(pars)):
        A[:,i] = vec @ A[:,i-1]
    cov = np.linalg.inv(A.transpose() @ Ninv @ A)
    m = cov @ (A.transpose() @ Ninv @ vec)
    return m

N = np.diag()
pars = linear(data,5,N)
