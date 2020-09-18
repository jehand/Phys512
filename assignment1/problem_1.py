import numpy as np

"""
We know that 𝛿=0.00102,0.102 for e^x and e^0.01x, respectively. We can check the accuracy by comparing the value of the actual
derivatives, to the values that we get using f'(x) = f(x+delta) - f(x) / delta. d/dx[e^x] = e^x and d/dx[e^0.01x] = 0.01e^0.01x.
"""

delta = np.linspace(10**(-10), 0, 1000)
x0 = 1 #random value
for i in delta: #This is a modification of the original test_deriv_accuracy.py
