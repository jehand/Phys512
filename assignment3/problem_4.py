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
#print("Parameters =", newton[0], "\n"+"Errors =", np.sqrt(np.diag(newton[1])), "\n"+"Chi =", newton[2])

"""
We now run our markov chain from 'markov.py' using the covariance given by Newtons method.
"""

n = 5000 

#markov = mcmc(pow,l,err,newton[0],newton[1],newton[2],n)
#markov = mcmc(pow,l,err,parsi,newton[1],1588.42568,n)
"""
chain0 = np.load("markov.npy", allow_pickle=True)
chain0[0] = chain0[0][400:,];chain0[1] = chain0[1][400:,]
parsc = np.mean(chain0[0], axis=0)
cov = np.cov(chain0[0].transpose())
markov = mcmc(pow,l,err,parsc,cov,chain0[1].mean(),n)"""

chain1 = np.load("markov1.npy", allow_pickle=True)
chain1[0] = chain1[0][4000:,];chain1[1] = chain1[1][4000:,]
parsc1 = np.mean(chain1[0], axis=0)
cov1 = np.cov(chain1[0].transpose())
markov1 = mcmc(pow,l,err,parsc1,cov1,chain1[1].mean(),n)

np.save("markov3", markov1)
print(markov1)
plt.plot(np.linspace(1,n,n),markov1[0][:,0],label="H0")
plt.plot(np.linspace(1,n,n),markov1[0][:,1],label="wh")
plt.plot(np.linspace(1,n,n),markov1[0][:,2],label="wc")
plt.plot(np.linspace(1,n,n),markov1[0][:,3],label="tau")
plt.plot(np.linspace(1,n,n),markov1[0][:,4],label="As")
plt.plot(np.linspace(1,n,n),markov1[0][:,5],label="slope")
plt.legend()
plt.show()