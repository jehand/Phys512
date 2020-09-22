import numpy as np
import matplotlib.pyplot as plt
from chebyshev import cheby

#First we define our function
x = np.linspace(0.5,1,1000)
y = np.log2(x)

"""
Now we need to figure out how many terms we need. We do this by first arbitrarily picking a Chebyshev polynomial of order 20, and then using while loop to see when the max error no longer exceeds 10^-6. 
"""
n = 1
ncheby = 20 #Arbitrarily chosen
yerr = [1] #Just for the while loop to work properly and to update the value of yerr
c = cheby(x,y,ncheby)[1] #Coefficients for the Cheby fit

while max(np.abs(yerr))>=1e-6 and n<ncheby:
    yerr = y-np.polynomial.chebyshev.chebval(x,c[:n+1]) #Calculating the error each time using n coefficients
    n += 1
print("truncated order=",n)

#Plotting the truncated fit
plt.plot(x,yerr,"r-", label="Order " +str(ncheby)+ " Cheby truncated to order " + str(n))

#Polynomial fit
coeffs = np.polynomial.legendre.legfit(x,y,7) #Using the same degree
ly = np.polynomial.legendre.legval(x,coeffs)
plt.plot(x,y-ly,"--",label="Polynomial")
plt.legend()
#plt.savefig("problem_2_plot.png", bbox_inches=True, dpi=500)
plt.show()