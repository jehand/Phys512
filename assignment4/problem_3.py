import numpy as np
import matplotlib.pyplot as plt
from problem_1 import shift
from problem_2 import corr

#Defining Gaussians
x = np.linspace(-5,5,100)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

#Defining a routine to take the correlation with an arbitrary shift
def corshift(arr,x):
    arrshift = shift(arr,x)
    return corr(arr,arrshift)

plt.figure(figsize=(8,6))
plt.plot(x,np.real(corshift(y,len(y)/10)),label=r"$\frac{1}{10}$")
plt.plot(x,np.real(corshift(y,len(y)/8)),label=r"$\frac{1}{8}$")
plt.plot(x,np.real(corshift(y,len(y)/6)),label=r"$\frac{1}{6}$")
plt.plot(x,np.real(corshift(y,len(y)/4)),label=r"$\frac{1}{4}$")
plt.plot(x,np.real(corshift(y,len(y)/2)),label=r"$\frac{1}{2}$")
plt.ylabel("Correlation Function",fontsize=14,labelpad=10)
plt.xlabel("x",fontsize=14)
plt.legend(fontsize=14,loc=2)
plt.savefig("problem_3.png",bbox_inches="tight",dpi=500)
plt.show()