import numpy as np
import camb
from wmap_camb_example import get_spectrum

def mcmc(d,l,err,pars,par_step,cov,chi,nstep=5000):
    

