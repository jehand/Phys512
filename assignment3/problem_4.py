import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum
from markov import mcmc
from scipy import signal

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""
We begin by copying our Newton program over so that we can use the co-variance matrix given by Newton's method.
"""

parsi = [65,0.02,0.1,0.05,2e-9,0.96]
dx = 0.001 #as the values of pars vary drastically, our dx value cannot be the same for all. Hence, choose dx=0.1% of param.
N = np.diag(err**2)

newton = np.load("newton.npy", allow_pickle=True) #Let us load our Newton data from problem 3 for floating tau

#We now run our markov chain from 'markov.py' using the covariance given by Newtons method.
n = 5000 #number of steps to use
markov = mcmc(pow,l,err,newton[0],newton[1],newton[2],n)

"""
#For our next chains we use the covariance matrix of the previous chain in order to determine our step size
chain0 = np.load("markov.npy", allow_pickle=True)
chain0[0] = chain0[0][500:,];chain0[1] = chain0[1][500:,]
parsc = np.mean(chain0[0], axis=0)
cov = np.cov(chain0[0].transpose())
markov = mcmc(pow,l,err,parsc,cov,chain0[1].mean(),n)
np.save("markov1", markov)

chain1 = np.load("markov1.npy", allow_pickle=True)
chain1[0] = chain1[0][4000:,];chain1[1] = chain1[1][4000:,]
parsc1 = np.mean(chain1[0], axis=0)
cov1 = np.cov(chain1[0].transpose())
markov1 = mcmc(pow,l,err,parsc1,cov1,chain1[1].mean(),n)
np.save("markov2", markov1)
print(markov1)"""

#We load in our previously saved results
chain = np.load("markov.npy",allow_pickle=True)
chain1 = np.load("markov1.npy",allow_pickle=True)
chain2 = np.load("markov2.npy",allow_pickle=True)

burnindex = 100 #the determined burn in index for each chain, chain1=500, chain2=4000, chain3=100.
n = len(chain[1][burnindex:])
plt.plot(np.linspace(1,n,n),chain2[1][burnindex:],color="mediumblue")
plt.xlabel("Step #", fontsize=14)
plt.ylabel(r"$\chi^2$", fontsize=14)
#plt.savefig("problem_4_chain3.png", bbox_inches="tight", dpi=500)
plt.show()

#We can now calculate the Fourier transform for each of our chains to see if we have reached convergence and determine independent samples
chainfour = signal.periodogram(chain2[1][burnindex:],scaling="spectrum",fs=2) #fs=2 to scale the fourier transform so that its max=1
plt.plot(chainfour[0],chainfour[1],color="crimson")
plt.ylim(1e-8,10) #Arbitrarily set to allow for better viewing
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Sample Frequencies Scaled", fontsize=14)
plt.ylabel("Power Spectrum", fontsize=14)
#plt.savefig("problem_4_chain3fourier.png", bbox_inches="tight", dpi=500)
plt.show()

#Qualitative independent samples calculation
chain1samples = 8000*0.025
chain2samples = 4900*0.021

#Calculating parameter values and errors
chain1pars = np.mean(chain1[0][4000:,:],axis=0)
chain1errs = np.std(chain1[0][4000:,:],ddof=1,axis=0)/np.sqrt(chain1samples)
chain2pars = np.mean(chain2[0][100:,],axis=0)
chain2errs = np.std(chain2[0][100:,],ddof=1,axis=0)/np.sqrt(chain2samples)
pars = (chain1pars+chain2pars)/2 #Calculating pars by taking an average
parserrs = np.sqrt(chain1errs**2+chain2errs**2)/2 #Calculating error in pars by taking the average
print("pars=",pars,"\nerrs=",parserrs)

#Final chi^2 using these new pars
chi2final = np.sum((pow-get_spectrum(pars,l))**2/err**2)
print("Final chi^2=",chi2final)