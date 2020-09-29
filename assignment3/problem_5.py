import numpy as np

x,y,z = np.loadtxt("dish_zenith.txt", usecols=0), np.loadtxt("dish_zenith.txt", usecols=1), np.loadtxt("dish_zenith.txt", usecols=2) #Importing data

"""
We now carry out our fit. First, we define our predicted function with our new parameters. 
"""

def para(x,y,x0,y0,z0,a,r):
    return a(x**2+y**2-x0*x-y0*x+r)+z0 #returning z-values

