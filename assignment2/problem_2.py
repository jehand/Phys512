import numpy as np
import matplotlib.pyplot as plt
from chebyshev import cheby

#First we define our function
x = np.linspace(0.5,1,1000)
y = np.log2(x)

"""
Now we need to figure out how many terms we need for the truncated fit. We do this by first arbitrarily picking a Chebyshev polynomial of order 20, and then using a while loop to see when the max error no longer exceeds 10^-6 (using our cheby function from chebyshev.py).
"""

ncheby = 20 #Arbitrarily chosen
n = 1 #Truncated number
c = cheby(x,y,ncheby,n) #Coefficients for the Cheby fit
while max(np.abs(y-c))>=1e-6 and n<ncheby: #Calculating the maximum error each time.
    c = cheby(x,y,ncheby,n)
    n+=1 #Increasing the order each time
n-=1 #As n originally gives the number of terms, yet n-1 is the order of the polynomial. c was still calculated using n terms, but n-1 is for the labelling.
print("Truncated Chebyshev Polynomial fit: truncated order =", n, ", max error =", max(np.abs(y-c)), ", RMS error =", np.std(y-c,ddof=1)) #ddof=1 for sample std.

#Plotting the truncated fit
plt.plot(x,y-c,"r-", label="Order " +str(ncheby)+ " Cheby truncated to order " + str(n))

#Plotting the untruncated chebyshev polynomial of order 7
cuy = cheby(x,y,n,n)
plt.plot(x,y-cuy, "k-", label="Cheby polynomial of order 7")
print("Untruncated Chebyshev Polynomial of order =", n, ", max error =", max(np.abs(y-cuy)), ", RMS error =", np.std(y-cuy, ddof=1)) #ddof=1 for sample std.

#Polynomial fit
coeffs = np.polynomial.legendre.legfit(x,y,n) #Using the same degree to find coeffs
ly = np.polynomial.legendre.legval(x,coeffs) #Finding the legendre fit
print("Polynomial fit: max error =", max(np.abs(y-ly)), ", RMS error =", np.std(y-ly,ddof=1)) #ddof=1 for sample std.
plt.plot(x,y-ly,"--",label="Polynomial")
plt.xlabel("x", fontsize=13)
plt.ylabel("Residual", fontsize=13)
plt.legend()
plt.savefig("problem_2_plot.png", bbox_inches="tight", dpi=500)
plt.show()