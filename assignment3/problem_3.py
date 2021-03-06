import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum
from newton import newton
from newton import newton_chi

d = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt", skiprows=20, usecols=(0,1,2)) #Importing the data
l, pow, err = d[:,0], d[:,1], d[:,2] #Splitting the data into its appropriate values

"""Our variables are written in the form [H0, ombh2, omch2, tau, As, ns]. We begin by writing a script that utilizes Newton's method to find the best 
fit parameters for our code. This script can be found in "newton.py". Our noise matrix is a matrix with err**2 on the diagonal. Our initial
parameters are the parameters we used in problem2. 
"""
parsi = [65,0.02,0.1,0.05,2e-9,0.96]
dx = 0.001 #as the values of pars vary drastically, our dx value cannot be the same for all. Hence, choose dx=0.1% of param.
N = np.diag(err**2)

model = get_spectrum(parsi,l)
result = newton_chi(pow,l,err,parsi,model,dx,N,0.001,True) #lets arbitrarily set the difference between our chi2 values to be less than 0.001
np.save("newtontauconst.npy", result) #Saving the results to be used for later fits
print("Parameters =", result[0], "\n"+"Errors =", np.sqrt(np.diag(result[1])), "\n"+"Chi =", result[2])

plt.xlabel(r"$l$", fontsize=14)
plt.ylabel("Power Spectrum", fontsize=14)
plt.errorbar(l, pow, yerr=err, fmt="kx", ms=4, elinewidth=0.5, capsize=1, alpha=0.2, label="Data")
plt.plot(l, result[-1], "r-", label="Newton's Fit")
plt.legend()
#plt.savefig("problem_3_taufloat.png", bbox_inches="tight", dpi=500)
plt.show()

#Residual plot
plt.errorbar(l, pow-result[-1], yerr=err, fmt="kx", ms=4, elinewidth=0.5, capsize=1, alpha=0.2, label="Residuals")
plt.ylabel("Residuals", fontsize=14)
plt.xlabel(r"$l$", fontsize=14)
#plt.savefig("problem_3_taufloatres.png", bbox_inches="tight", dpi=500)
plt.show()

#Derivative plot
newtonchi1 = newton(pow,l,parsi,model,[i*dx for i in parsi],N,True)
plt.plot(l,np.abs(newtonchi1[2][:,0])*dx*parsi[0], label=r"$H_0$")
plt.plot(l,np.abs(newtonchi1[2][:,1])*dx*parsi[1], label=r"$\omega_b h^2$")
plt.plot(l,np.abs(newtonchi1[2][:,2]*dx*parsi[2]), label=r"$\omega_c h^2$")
plt.plot(l,np.abs(newtonchi1[2][:,3]*dx*parsi[3]), label=r"$\tau$")
plt.plot(l,np.abs(newtonchi1[2][:,4]*dx*parsi[4]), label=r"$A_s$")
plt.plot(l,np.abs(newtonchi1[2][:,5]*dx*parsi[5]), label="Slope")
plt.xlabel("l", fontsize=14)
plt.ylabel(r"$|f'(m)|dx$", fontsize=14)
plt.yscale("log")
plt.legend()
#plt.savefig("problem_3_derivstaufloat.png",bbox_inches="tight",dpi=500)
plt.show()