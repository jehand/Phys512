import numpy as np
from scipy import integrate

"""
To solve for the decay products of U238, we copy the code used in class for the implicit case. The only adjustment that will need to be made is to include more half lives in the evaluation.
"""

def radioactivity(x,y,half_life):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    dydx[1]=y[0]/half_life[0]-y[1]/half_life[1]
    dydx[2]=y[1]/half_life[1]
    return dydx