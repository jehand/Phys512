import numpy as np

"""
We first recognise that the range [0.5,1] is not the range of the Chebyshev polynomials. Hence, we must adjust the range of our function. This can be done fairly easily by taking the range of x of our function i.e. from 0.5-1 and rescaling it to be -1,1. We do not have to do this generally, as we are only concerned with log_2(x) from 0.5-1, hence we shall do it specifically for this case. Essentially our x values are scaled up by 4 (since we're going from a range of 0.5 to a range of 2) and a factor of -3 is added such that the x values match up (since multiplying 0.5 by 4 gives 2 and we want it to be at -1). After this we must find the cheby matrix which we can take directly from the lectures. 
"""

def cheby(x,y,ord):
    x = 4*x - 3 #Changing the range of our x-values to match
    nx = len(x)
    mat=np.zeros([nx,ord+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(1,ord):
        mat[:,i+1]=2*x*mat[:,i]-mat[:,i-1]
    coeffs = np.linalg.pinv(mat)@y
    return np.polynomial.chebyshev.chebval(x,coeffs),coeffs