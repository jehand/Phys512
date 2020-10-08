import numpy as np
import camb
from wmap_camb_example import get_spectrum

"""
We can take the 'mcmc_class_wnewton.py' from lectures and adapt it for our needs. Print statements have been added just so that we are aware of where
in the program we currently are. Furthermore, we add a parameter tau that checks whether tau has been given as a prior or not, and a parameter tau_std
which is the uncertainty in tau. If tau has been given as a prior, we chooses a value for tau from the random sampling of a normal distribution
centered at tau with sig=tau_std (i.e. tau's error).
"""

def mcmc(d,l,err,pars,cov,chi_cur,nstep=5000,tau=0,tau_std=0):
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
        if tau !=0: #If it is not the default, then we know our covariance matrix is pars-1 dimensional so we insert our tau step.
            par_step = np.insert(par_step,3,tau_std*np.random.randn()) #tau uncert is a 1sig unertainty, hence we step with a normal distribution.
            pars_trial[3] = tau #Always setting tau to be what was determined by Planck and then taking a step from there. i.e. tau' ~ N(tau,tau_std).
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