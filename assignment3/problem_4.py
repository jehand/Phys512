import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum
from newton import newton
from newton import newton_chi
from markov import mcmc

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""
We begin by copying our Newton program over so that we can use the co-variance matrix given by Newton's method.
"""

parsi = [65,0.02,0.1,0.05,2e-9,0.96]
dx = 0.001 #as the values of pars vary drastically, our dx value cannot be the same for all. Hence, choose dx=0.1% of param.
N = np.diag(err**2)

newton = np.load("newton.npy", allow_pickle=True) #Let us load our Newton data from problem 3 for floating tau
print("Parameters =", newton[0], "\n"+"Errors =", np.sqrt(np.diag(newton[1])), "\n"+"Chi =", newton[2])

"""
We now run our markov chain from 'markov.py' using the covariance given by Newtons method.
"""

markov = mcmc(pow,l,newton[-1],err,newton[0],newton[1],newton[2],5000)
print(markov)