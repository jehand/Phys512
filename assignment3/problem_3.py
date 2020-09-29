import numpy as np
from wmap_camb_example import get_spectrum
from newton import newton

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""Our variables are written in the form [H0, ombh2, omch2, tau, As, ns]. We begin by writing a script that utilizes Newton's method to find the best 
fit parameters for our code. This script can be found in "newton.py". Our noise matrix is a matrix with err**2 on the diagonal. Our initial
parameters are the parameters we used in problem2. 
"""
pars = [65,0.02,0.1,0.05,2e-9,0.96]
dx = 0.001*pars #as the values of pars vary drastically, our dx value cannot be the same for all, hence arbitrarily is 0.1% of each parameter.
N = np.diag(err**2)


