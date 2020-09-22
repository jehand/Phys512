import numpy as np
import matplotlib.pyplot as plt
from chebyshev import cheby

#First we define our function
x = np.linspace(0.5,1,1000)
y = np.log2(x)

ord=7
#Chebyshev polynomial fit untruncated
cy = cheby(x,y,ord)
plt.plot(x,y-cy,label="Cheby")

#Now we need to figure out how many terms we need: Do this by calculating accuracy for a 

#Polynomial fit
coeffs = np.polynomial.legendre.legfit(x,y,ord) #Using the same degree
ly = np.polynomial.legendre.legval(x,coeffs)
plt.plot(x,y-ly,label="Polynomial")
plt.legend()
#plt.savefig("problem_2_plot.png", bbox_inches=True, dpi=500)
plt.show()