import numpy as np
import matplotlib.pyplot as plt

#First we define our function
x = np.linspace(0.5,1,50)
y = np.log2(x)
plt.plot(x,y,label="$log_2{x}$")

#Chebyshev polynomial fit


#Polynomial fit
coeffs = np.polynomial.legendre.legfit(x,y,5) #Using the same degree
ly = np.polynomial.legendre.legval(coeffs,x)
plt.plot(x,ly,"label=Polynomial")
#plt.savefig("problem_2_plot.png", bbox_inches=True, dpi=500)
plt.show()