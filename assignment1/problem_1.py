import numpy as np

"""
We know that 𝛿=0.00102,0.102 for e^x and e^0.01x, respectively. We can check the accuracy by comparing the value of the actual
derivatives, to the values that we get using the formula for f'(x) derived. d/dx[e^x] = e^x and d/dx[e^0.01x] = 0.01e^0.01x.
"""

delta = np.linspace(-6, 0, 2000)
x0 = 1 #random value

def exp(x, a): #defining the exponential function
    return np.exp(a*x)

def deriv(func, a, d): #derivative formula derived
    first = (2/3)*(func(x0+d,a) - func(x0-d,a))
    second = (1/12)*(-func(x0+2*d,a) + func(x0-2*d,a))
    return (first+second)/d

diffs1, diffs2 = [], [] #arrays to store the difference between true derivative and the delta derivative (1 being e^x and 2 e^0.01x)
for d in 10**delta:
    diffs1.append(np.abs(deriv(exp,1,d)-exp(x0,1)))
    diffs2.append(np.abs(deriv(exp,0.01,d)-0.01*exp(x0,0.01)))

deltaopt1 = 10**delta[diffs1.index(min(diffs1))] #finding the value of delta
deltaopt2 = 10**delta[diffs2.index(min(diffs2))] #finding the value of delta
print(deltaopt1, deltaopt2)