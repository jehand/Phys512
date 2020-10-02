import numpy as np
import camb
from wmap_camb_example import get_spectrum

"""
We can take the 'mcmc_class_wnewton.py' from lectures and adapt it for our needs. Print statements have been added just so that we are aware of where
in the program we currently are.
"""

def mcmc(d,l,pred,err,pars,cov,chi_cur,nstep=5000):
    print("Starting Markov")
    npar=len(pars)
    n_success = 0
    r = np.linalg.cholesky(cov) #Calculating cholesky matrix
    chain = np.zeros([nstep,npar])
    chivec = np.zeros(nstep)
    for i in range(nstep):
        print("\t" + "Step #" + str(i+1))
        par_step = np.dot(r,np.random.randn(r.shape[0]))*0.4 #Our step size is too large -> arbitrarily adjust by a factor of 0.4 (=35% success rate)
        pars_trial = pars+par_step
        new_model = get_spectrum(pars_trial,l)
        chi_trial = np.sum((d-new_model)**2/(err**2))
        #we now have chi^2 at our current location and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        if (np.random.rand(1)<accept_prob) & (pars_trial[3]>=0): #accept the step with appropriate probability + don't accept if tau<0. 
            pars = pars_trial.copy()
            chi_cur = chi_trial.copy()
            n_success+=1
            print("\t\t"+"Success! Pars=", pars, "Chi", chi_cur)
        else:
            print("\t\t"+"Fail :(")
        chain[i,:] = pars
        chivec[i] = chi_cur
    return chain, chivec, (n_success/nstep)*100