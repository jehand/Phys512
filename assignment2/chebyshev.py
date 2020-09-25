import numpy as np

"""
We first recognise that the range [0.5,1] is not the range of the Chebyshev polynomials. Hence, we must adjust the range of our function (log_2x) to match that of the Chebyshev polynomials which is [-1,1]. This can be done fairly easily by taking the range of x of our function i.e. from 0.5->1 and rescaling it to be -1->1. We do not have to do this in a general format as we are only concerned with log_2(x) from 0.5->1, hence we shall do it specifically for this case. Essentially our x values are scaled up by 4 (since we're going from a range of 0.5 to a range of 2) and a factor of -3 is added such that the x values match up (since multiplying 0.5 by 4 gives 2 and we want the range to start at -1). After this we must find the cheby matrix and find the coefficients which we can take directly from the lectures. We can then create our polynomial using chebval from np.polynomial.chebyshev. Furthermore, we can add another feature to our function "trunc" which determines the Chebyshev polynomial up to an order "ord" and then truncates the number of terms used at "trunc" in order to give us our truncated Chebyshev polynomial.  
"""

def cheby(x,y,ord,trunc):
    x = 4*x - 3 #Changing the range of our x-values to match
    nx = len(x)
    mat=np.zeros([nx,ord+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(1,ord):
        mat[:,i+1]=2*x*mat[:,i]-mat[:,i-1]
    coeffs = np.linalg.pinv(mat)@y #We use pinv such that there are no possible issues with singular matrices
    return np.polynomial.chebyshev.chebval(x,coeffs[:trunc+1])