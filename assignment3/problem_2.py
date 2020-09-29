import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values
vars = [65, 0.02, 0.1, 0.05, 2e-9, 0.96] #Defining our variables in the form [H0, ombh2, omch2, tau, As, ns]

#We can calculate chi^2 fairly easily taking err to be our sigma and pow to be our x values. The predicted values can be found from the model
chisq = np.sum(((pow-get_spectrum(vars,l))**2)/(err**2))
print("chisq =", chisq)