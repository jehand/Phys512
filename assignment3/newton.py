import numpy as np
import camb
from wmap_camb_example import get_spectrum

"""
We take inspiration from the "newton.py" file used in lectures to write our own Newton's method best fit finder. The main difference is that we no
longer have an estimate for the derivatives as CAMB does not give us derivatives with respect to the model. A simple solution to this would be to
calculate that function at a point (x+∆x) where ∆x is small and to calculate the derivative by the usual method: f' = f(x+∆x)-f(x)/∆x. However, this
is only for taking the limit from the left side however, for both sides we also need to calculate f(x-∆x) therefore f' = f(x+∆x)-f(x-∆x)/2∆x. This is
a large number of calculations! Besides this, we carry out the normal Newton's method as described in class. 
"""

def newton(d,l,pars,pred,delx,N,tau=True): #tau=True when we already know the value of tau
    r = d-pred #residual array
    N = np.linalg.inv(N) #calculating inverse N
    delx[3] = 0 if tau else delx[3] #Since we are setting this to 0, we cannot divide for our derivative. 
    #We next look to calculate all the derivatives
    dA = np.empty(shape=(len(d),len(pars)))
    for i in range(len(pars)): #Calculating the derivatives for each parameter
        if i==3 and tau: #To avoid the divide by 0
            continue
        print("\t\t"+"Parameter " + str(i+1))
        pars1, pars2 = list(pars), list(pars)
        pars1[i] += delx[i]
        pars2[i] -= delx[i]
        fxplus = get_spectrum(pars1,l)
        fxminus = get_spectrum(pars2,l)
        dA[:,i] = (fxplus-fxminus)/(2*delx[i])
    if tau:
        dA = np.delete(dA, 3, axis=1)
    #We now have our derivatives, however if we are not concerned with tau, we do not need that column of the matrix. Proceeding with Newton's
    cov = np.linalg.inv(dA.transpose() @N @dA) #to calculate the errors
    dm = cov @ (dA.transpose() @N @r)

    #If we already have the value for tau
    if tau:
        dm = np.insert(dm,3,0,axis=0)
    return pars+dm, cov

"""
We now write our function that will cycle through values until chi2 is changing by less than chimin, depending on the accuracy we desire. We will also
include our code to calculate the errors in our parameters here so that it is all calculated at the same time.
"""

def newton_chi(d,l,err,pars,pred,dx,N,chimin,tau=True):
    delchi = 10
    covbef = 0 #Defining so that previous covariance can be outputted if delchi<0
    n = 1
    while delchi>chimin and n<10: #Condition that gives us the accuracy that we desire, yet also does not allow the loop to continue forever
        print("Chi " + str(n) + ":")
        delx = [i*dx for i in pars] #Calculating the new delx for each iteration
        powbef = get_spectrum(pars,l)
        chibef = np.sum((d-powbef)**2/err**2) #Calculating chi2 before change in pars
        newtonpars, cov = newton(d,l,pars,powbef,delx,N,tau) #Calling our previous function to calculate new pars
        powaft = get_spectrum(newtonpars,l)
        chiaft = np.sum((d-powaft)**2/err**2) #Calulating chi2 after change in pars
        delchi = chibef-chiaft #chibef-chiaft should always be positive. However, if it's negative, we should return the previous pars.
        print("\t\t" + "chi" + str(n) + "=", chiaft, "\n")
        if delchi<0: #Returning previous pars if negative
            return pars, covbef, chibef, delchi, powbef
        pars = list(newtonpars)
        n += 1 #Just so that we do not loop forever
        covbef = cov.copy() #taking the previous covariance so that it can be outputted if delchi<0
    return pars, cov, chiaft, delchi, powaft

    """
    This is not the most efficient method as it repeats calculations for powbef and chibef multiple times when it has already been calculated as
    chiaft and powaft in the previous iteration. However, it gets the job done.
    """