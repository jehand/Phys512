import numpy as np
import camb
from wmap_camb_example import get_spectrum

"""
We can take the 'mcmc_class_wnewton.py' from lectures and adapt it for our needs. Print statements have been added just so that we are aware of where
in the program we currently are. Furthermore, we add a parameter tau_interv that checks whether tau has been given as a prior or not. If it has, it
steps tau by the random sampling of a normal distribution centered at 0 with sig=1, multiplyed by our interval (i.e. tau's error). We also have a
parameter tau that sets the mean value of tau, so each time we are iterating about the mean of tau, and hence our chain does not drag too far away
from the origin.
"""

def mcmc(d,l,err,pars,cov,chi_cur,nstep=5000,tau=0,tau_interv=0):
    print("Starting Markov")
    npar = len(pars)
    n_success = 0 #so we can determine acceptance rate
    r = np.linalg.cholesky(cov) #Calculating cholesky matrix
    chain = np.zeros([nstep,npar])
    chivec = np.zeros(nstep)
    for i in range(nstep):
        print("\t" + "Step #" + str(i+1))
        par_step = np.dot(r,np.random.randn(r.shape[0]))
        pars_trial = pars.copy()
        if tau_interv !=0: #If it is not the default, then we know our covariance matrix if pars-1 dimensional so we add another column for tau.
            par_step = np.insert(par_step,3,tau_interv*np.random.randn()) #tau uncert is a 1sig unertainty, hence we step with a normal distribution.
            pars_trial[3] = tau #Always setting tau to be what was determined by Planck and then taking a step from there.
        pars_trial += par_step*0.5 #Our step size is too large -> arbitrarily adjust for each chain
        new_model = get_spectrum(pars_trial,l)
        chi_trial = np.sum((d-new_model)**2/(err**2))
        accept_prob = np.exp(-0.5*(chi_trial-chi_cur)) #decide if we take the step
        if (np.random.rand(1)<accept_prob) & (pars_trial[3]>0): #accept the step with appropriate probability + don't accept if tau<0. 
            pars = pars_trial
            chi_cur = chi_trial
            n_success += 1
            print("\t\t"+"Success! Pars=", pars, "Chi=", chi_cur)
        else:
            print("\t\t"+"Fail :(")
        chain[i,:] = pars
        chivec[i] = chi_cur
    return chain, chivec, (n_success/nstep)*100