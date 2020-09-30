import numpy as np
from wmap_camb_example import get_spectrum
from newton import newton
from newton import newton_chi

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""Our variables are written in the form [H0, ombh2, omch2, tau, As, ns]. We begin by writing a script that utilizes Newton's method to find the best 
fit parameters for our code. This script can be found in "newton.py". Our noise matrix is a matrix with err**2 on the diagonal. Our initial
parameters are the parameters we used in problem2. 
"""
parsi = [65,0.02,0.1,0.05,2e-9,0.96]
dx = 0.001 #as the values of pars vary drastically, our dx value cannot be the same for all. Hence, choose dx=0.1% of param.
N = np.diag(err**2)

model = get_spectrum(parsi,l)
new_params = newton_chi(pow,l,err,parsi,model,dx,N,0.01)
print(new_params)