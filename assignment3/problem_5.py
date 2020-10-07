import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum
from markov import mcmc
from scipy import signal

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""
We begin by recognising that we are no longer stepping tau using the results of our Cholesky matrix. Hence, we must adapt our 'markov.py' file (which
we already have). Begin by using the covariance matrix given by newtons method without tau and calculate the cholesky matrix with it to take
correlated steps for the other parameters. We then randomly sample our value of tau from a normal distribution centered at the value of tau with a
standard deviation equal to the uncertainty in tau (taking, by convention, the uncertainty to be the 1 sigma deviation).
"""

#Importing newton data
newton = np.load("newtontauconst.npy", allow_pickle=True)

n = 20000 #number of steps
tau = 0.0544
utau = 0.0073
newton[0][3] = tau
print("Initial Pars=", newton[0])

#We also wanna recalculate chi^2 as we now have new pars from Planck
chi = np.sum((pow-get_spectrum(newton[0],l))**2/err**2)
print("Starting Chi^2=", chi)

markov = mcmc(pow,l,err,newton[0],newton[1],chi,n,tau,utau)
np.save("markovtau",markov)

#We load in the data so we do not need to run the above code all the time
markov = np.load("markovtau.npy", allow_pickle=True)
burnindex = 100
n = len(markov[0][:,0])-burnindex

print("Tau avg=", np.mean(markov[0][:,3]))
print("Acceptance Rate=",markov[-1])
plt.plot(np.linspace(1,n,n),markov[1][burnindex:],color="mediumblue")
plt.xlabel("Step #", fontsize=14)
plt.ylabel(r"$\chi^2$", fontsize=14)
#plt.savefig("problem_5_plot.png", bbox_inches="tight", dpi=500)
plt.show()

#We can also calculate the Fourier transform to see if we have reached convergence and determine independent samples
chainfour = signal.periodogram(markov[1][burnindex:],scaling="spectrum",fs=2) #fs=2 to scale the fourier transform so that its max=1
plt.plot(chainfour[0],chainfour[1],color="crimson")
plt.ylim(1e-9,10) #Arbitrarily set to allow for better viewing
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Sample Frequencies Scaled", fontsize=14)
plt.ylabel("Power Spectrum", fontsize=14)
#plt.savefig("problem_5_fourier.png", bbox_inches="tight", dpi=500)
plt.show()

#Calculating parameter values and errors
indeptrials = 0.016*n #number of independent trials
print("Independent Trials=",indeptrials)
chainpars = np.mean(markov[0][burnindex:,:],axis=0)
chainerrs = np.std(markov[0][burnindex:,:],ddof=1,axis=0)/np.sqrt(indeptrials)
print("Pars=",chainpars,"\nErrs=",chainerrs)

#Final chi^2 using these new pars
chi2final = np.sum((pow-get_spectrum(chainpars,l))**2/err**2)
print("Final chi^2=",chi2final)