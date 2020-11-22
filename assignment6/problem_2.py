import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

n = 10000000 #number of points
alpha = -1.5 #for our power law
saveplot = False #to save plots when we want
rands, acceptrands = np.random.rand(n), np.random.rand(n)

#Note that as we are only concerned with x>0 we take the absolute so that all our numbers are >0. 
def lorentz(rands):
    return np.abs(np.tan(np.pi * (rands-0.5)))

def gauss(rands,n=n): 
    #Need to generate another random number
    rands2 = np.random.rand(n)
    R = np.sqrt(-2*np.log(rands))
    theta = 2*np.pi*rands2
    return np.abs(R * np.sin(theta)) #Gives a Gaussian centered at 0 with an std of 1

def power(rands,alpha=alpha):
    return np.abs((1-rands) ** (1/(1+alpha)))

#Probability of accepting is ratio of exponential to the distribution
lorentznumbs = lorentz(rands)
gaussnumbs = gauss(rands)
powernumbs = power(rands)

problorentz = np.exp(-np.abs(lorentznumbs)) / (1/(1+lorentznumbs**2))
probgauss = np.exp(-gaussnumbs) / (3*np.exp(-0.5 * (gaussnumbs**2))/np.sqrt(2*np.pi))
probpower = np.exp(-(powernumbs-1)) / (powernumbs**(1/(1+alpha))) #this is wrong, remember to change

acceptlorentz = acceptrands < problorentz
acceptgauss = acceptrands < probgauss
acceptpower = acceptrands < probpower

#assert(np.max(problorentz) <= 1) #Just to check that the probability is always less than 1, i.e. that our distribution is always greater than exp.
#assert(np.max(probgauss) <= 1) #This is just a test to prove that it can never work for a Gaussian
#assert(np.max(probpower) <= 1)

#We can calculate our acceptance rate for each distribution
proportionlorentz = np.sum(acceptlorentz)/n
proportiongauss = np.sum(acceptgauss)/(n*2) #multiplied by 2 cause the Gaussian generator requires two numbers per time
proportionpower = np.sum(acceptpower)/n

print("Lorentz Acceptance Rate =", proportionlorentz)
print("Gauss Acceptance Rate =", proportiongauss)
print("Power Acceptance Rate =", proportionpower)

#Plotting lorentz rejection data
x = np.linspace(0,5,1000)
y = np.exp(-x)
plt.hist(lorentznumbs[acceptlorentz], density=True, bins=100, range=(0,5), histtype="bar", ec="black",label="Histogram")
plt.plot(x,y,"r-",label="Predicted")
plt.xlabel("Number", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
if saveplot: plt.savefig("plots/problem_2_lorentz.png", bbox_inches="tight", dpi=500)
plt.show()

#Plotting gauss rejection data
plt.hist(gaussnumbs[acceptgauss], density=True, bins=100, range=(0,5), histtype="bar", ec="black",label="Histogram")
plt.plot(x,y,"r-",label="Predicted")
plt.xlabel("Number", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
if saveplot: plt.savefig("plots/problem_2_gauss.png", bbox_inches="tight", dpi=500)
plt.show()

#Plotting lorentz rejection data
plt.hist(powernumbs[acceptpower], density=True, bins=100, range=(1,5), histtype="bar", ec="black",label="Histogram")
plt.plot(x,np.exp(-(x-1)),"r-",label="Predicted")
plt.xlabel("Number", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
plt.xlim(1,5.2)
plt.ylim(0,1.05)
if saveplot: plt.savefig("plots/problem_2_power.png", bbox_inches="tight", dpi=500)
plt.show()