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
    delx[3] = 0 if tau else delx[3] #Since we are setting this to 0, we cannot divide for our derivative. 

    #We next look to calculate all the derivatives
    dA = np.empty(shape=(len(d),len(pars)))
    for i in range(len(pars)): #Calculating the derivatives for each parameter
        if i!=3: #To avoid the divide by 0
            print("Parameter "+str(i+1))
            pars1, pars2 = pars,pars; pars1[i] += delx[i]; pars2[i] += delx[i]
            fxplus = get_spectrum(pars1,l)
            fxminus = get_spectrum(pars2,l)
            dA[:,i] = (fxplus-fxminus)/(2*delx[i])

    if tau:
        dA = np.delete(dA, 3, axis=1)
    #We now have our derivatives, however if we are not concerned with tau, we do not need that column of the matrix. Proceeding with Newton's
    dm = np.linalg.pinv(dA.transpose() @N @dA) @ (dA.transpose() @N @r)

    #If we already have the value for tau
    if tau:
        dm = np.insert(dm,3,0,axis=0)
    print(dm)
    return pars+dm

"""
We now write our function that will cycle through values until chi2 is changing by less than chimin, depending on the accuracy we desire. 

def newton_chi(d,l,pars,pred,delx,N,tau=True,chimin):
    while delchi2>chimin:
        newton = newton(d,l,pars,pred,delx,N,tau=True)

    return chi2, """