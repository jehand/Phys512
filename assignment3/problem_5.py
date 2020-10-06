import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum
from markov import mcmc

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""
We begin by recognising that we are adding a new interval of tau as a prior in our matrix. Hence, we must adapt our 'markov.py' file (which we already
have). Furthermore, we use the covariance matrix given by newtons method without tau and calculate the cholesky matrix with it to take correlated
steps. We then add our interval of tau as a prior using a random number fit.
"""

#Importing newton data
newton = np.load("newtontauconst.npy")

n = 50 #number of steps
tau = 0.0544
utau = 0.0073

markov = mcmc(pow,l,err,newton[0],newton[1],newton[2],n,utau)
