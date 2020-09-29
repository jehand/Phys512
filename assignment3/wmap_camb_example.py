import numpy as np
import camb
from matplotlib import pyplot as plt
import time

def get_spectrum(pars,l):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(int(max(l)),lens_potential_accuracy=0) #Edited so that the function calculates the maximum of l inside
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[int(min(l)):int(max(l)+1),0]    #Edited so that the function outputs the results between the minimum and maximum l value
    return tt