import numpy as np
import camb
from wmap_camb_example import get_spectrum

"""
We take inspiration from the "newton.py" file used in lectures to write our own Newton's method best fit finder. The main difference is that we no
longer have an estimate for the derivatives as CAMB does not give us derivatives with respect to the model. A simple solution to this would be to
calculate that function at a point (x+∆x) where ∆x is small and to calculate the derivative by the usual method: f' = f(x+∆x)-f(x)/∆x. This is for
taking the limit from the left side however, for both sides we also need to calculate f(x-∆x) therefore f' = f(x+∆x)-f(x-∆x)/2∆x. This is a large
number of calculations! Besides this, we carry out the normal Newton's method as described in class. 
"""

def newton(d,l,pars,pred,delx,N,tau=True): #tau=True when we already know the value of tau
    r = d-pred #residual array
    N = np.linalg.pinv(N) #calculating inverse N
    #We next look to calculate all the derivatives
    dA = [] #np.zeros(shape=(len(d),len(pars)))
    for i in range(len(pars)): #Calculating the derivatives for each parameter
        fxplus = get_spectrum(pars[i]+delx[i],l)
        fxminus = get_spectrum(pars[i]-delx[i],l)
        dA[:,i] = (fxplus-fxminus)/(2*delx[i])

    #We now have our derivatives, however if we are not concerned with tau, we do not need that column of the matrix.
    #dA = np.delete(dA,3,axis=1) if tau else dA
    #Proceeding with Newton's since we now have everything
    dm = np.linalg.pinv(dA.transpose() @N @dA) @ (dA.transpose() @N @r)

    #If we already have the value for tau
    dm = np.insert(dm,3,0,axis=0) if tau else dm
    return pars+dm

"""
We now write our function that will cycle through values to give us the minimum value of 
"""
def newton_chi(d,l,pars,pred,delx,N,tau=True,chimin):
    while delchi2>chimin:
        newton = newton(d,l,pars,pred,delx,N,tau=True)

    return chi2, 