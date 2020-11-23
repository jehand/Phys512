import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

n = 10000000 #number of points

u,v = np.random.rand(n), 4*(np.random.rand(n)-0.5)*np.exp(-1) #where u = [0,1] and v is defined as in the LaTex document to be v = [-2√e,2√e]. 
rat = v/u
accept = u < np.sqrt(np.exp(-v/u))
acceptratio = np.sum(accept)/(2*n)
print("Acceptance rate =", acceptratio)

#Plotting the results
x = np.linspace(0,5,1000)
y = np.exp(-x)
plt.hist(rat[accept], density=True, bins=100, range=(0,5), histtype="bar", color="teal", ec="black",label="Histogram")
plt.plot(x,y,"r-",label="Predicted")
plt.xlabel("Number", fontsize=14)
plt.ylabel("Relative Frequency", fontsize=14, labelpad=5)
plt.legend(fontsize=12)
plt.savefig("plots/problem_3.png", bbox_inches="tight", dpi=500)
plt.show()