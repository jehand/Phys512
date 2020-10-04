import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl

cmap = mpl.cm.Greys(np.linspace(0,1,50))
cmap = mpl.colors.ListedColormap(cmap[30:,:-1])

data = np.loadtxt("dish_zenith.txt") #Importing data

"""
We now carry out our fit. First, we define our predicted parabaloid function that is linear in our new parameters. 
"""

def para(x,y,x0,y0,c0,a):
    return a(x**2+y**2-x0*x-y0*x)+c0 #returning z-values

"""def linear(d,pars):
    vec = d[:,:2] #We want to just use the x,y columns.
    A = np.empty(shape=(len(data),pars))
    A[:,0] = 1.0
    for i in range(1,pars):
        A[:,i] = vec @ A[:,i-1]
    cov = np.linalg.inv(A.transpose() @ A)
    m = cov @ (A.transpose() @ vec)
    return m

pars = linear(data,4)"""

#Plotting the original data in 3D
x,y,z = data[:,0], data[:,1], data[:,2]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(x,y,z,c=-z,s=10,cmap=cmap,label="Data")

#Now we create our surface using the our linear interpolation for z.
"""
X,Y = np.meshgrid(x,y)
Z = para(X,Y,x0,y0,c0,a)
ax.plot_trisurf(X,Y,Z,cmap="viridis",label="Linear Fit")
"""

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()
