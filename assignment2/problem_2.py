import numpy as np
import matplotlib.pyplot as plt
from chebyshev import cheby

#First we define our function
x = np.linspace(0.5,1,1000)
y = np.log2(x)

"""
Now we need to figure out how many terms we need for the truncated fit. We do this by first arbitrarily picking a Chebyshev polynomial of order 20, and then using while loop to see when the max error no longer exceeds 10^-6. 
"""

ncheby = 20 #Arbitrarily chosen
n = 1 #Truncated number
c = cheby(x,y,ncheby,n) #Coefficients for the Cheby fit
while max(np.abs(y-c))>=1e-6 and n<ncheby:
    c = cheby(x,y,ncheby,n)
    n+=1t
n-=1
print("Truncated Chebyshev Polynomial fit: truncated order =", n, "max error =", max(np.abs(y-c)), "RMS error =", np.std(y-c,ddof=1)) #ddof=1 for sample std.

#Plotting the truncated fit
plt.plot(x,y-c,"r-", label="Order " +str(ncheby)+ " Cheby truncated to order " + str(n))

#Plotting the untruncated chebyshev polynomial of order 7
cuy = cheby(x,y,n,n)
plt.plot(x,y-cuy, "k-", label="Cheby polynomial of order 7")
print("Untruncated Chebyshev Polynomial of rder =", n, "max error =", max(np.abs(y-cuy)), "RMS error =", np.std(y-cuy, ddof=1)) #ddof=1 for sample std.

#Polynomial fit
coeffs = np.polynomial.legendre.legfit(x,y,n) #Using the same degree
ly = np.polynomial.legendre.legval(x,coeffs)
print("Polynomial fit: max error =", max(np.abs(y-ly)), "RMS error =", np.std(y-ly,ddof=1)) #ddof=1 for sample std.
plt.plot(x,y-ly,"--",label="Polynomial")
plt.xlabel("x", fontsize=13)
plt.ylabel("Residual", fontsize=13)
plt.legend()
plt.savefig("problem_2_plot.png", bbox_inches="tight", dpi=500)
plt.show()