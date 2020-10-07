import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum
from markov import mcmc
from scipy import signal

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""
We begin by recognising that we are adding a new interval of tau as a prior in our matrix. Hence, we must adapt our 'markov.py' file (which we already
have). Furthermore, we use the covariance matrix given by newtons method without tau and calculate the cholesky matrix with it to take correlated
steps. We then add our interval of tau as a prior using a random number fit.
"""

#Importing newton data
newton = np.load("newtontauconst.npy", allow_pickle=True)

n = 20000 #number of steps
tau = 0.0544
utau = 0.0073
newton[0][3] = tau
print("Initial Pars=", newton[0])

#We also wanna recalculate chi^2 as we now have new pars
chi = np.sum((pow-get_spectrum(newton[0],l))**2/err**2)
print("Starting Chi^2=", chi)

markov = mcmc(pow,l,err,newton[0],newton[1],chi,n,tau,utau)
np.save("markovtau2",markov)

print("Tau avg=", np.mean(markov[0][:,3]))
print("Acceptance Rate=",markov[-1])
plt.plot(np.linspace(1,n,n),markov[0][:,0],label="H0")
plt.plot(np.linspace(1,n,n),markov[0][:,3],label="Tau")
plt.legend()
plt.show()

#We can also calculate the Fourier transform to see if we have reached convergence and determine independent samples
chainfour = signal.periodogram(markov[1],scaling="spectrum",fs=2) #fs=2 to scale the fourier transform so that its max=1
plt.plot(chainfour[0],chainfour[1],color="crimson")
plt.ylim(1e-8,10) #Arbitrarily set to allow for better viewing
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Sample Frequencies Scaled", fontsize=14)
plt.ylabel("Power Spectrum", fontsize=14)
#plt.savefig("problem_5_fourier.png", bbox_inches="tight", dpi=500)
plt.show()