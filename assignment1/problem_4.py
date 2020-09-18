import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.constants as con

e0 = con.epsilon_0
d = np.linspace(0,10,101)
#We can always align our axis in such a way that z represents the distance from the shell.
#Let's use the function given by Griffiths Problem 2.7; we can use the substituted integral with respect to u.
#However, we can write the function in terms of z/R and therefore, the distance z can be written in terms of R.
#We can also write the field in terms of sigma, i.e. E/sigma where sigma is the charge per unit area (surface charge density).
#Hence, our x-axis is in units of R and our y-axis is in units of sigma.
def E(u,z):
    coeff = 1/(2*e0)
    integrand = (z-u)/((1+z**2-2*z*u)**(3/2))
    return coeff*integrand

#Integrator


#scipy.quad
Eq = []
for z in d: #integrating for each distance z
    Eq.append(integrate.quad(E,-1,1,args=(z))[0])
plt.plot(d,Eq,label="scipy.quad")
plt.xlabel(r"$\frac{z}{R}$", fontsize=18)
plt.ylabel(r"$\frac{E}{\sigma}$", fontsize=18)
plt.legend()
plt.show()

"""
There is a singularity in the integral around z=R (or z/R=1 on the plot). Quad takes E to be a very large number, however
it does not care about this and still integrates correctly over the bounds. 
"""