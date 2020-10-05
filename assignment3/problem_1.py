import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl

cmap = mpl.cm.Reds(np.linspace(0,1,50)) #Next 2 lines are just to make the 3D plot easier to see
cmap = mpl.colors.ListedColormap(cmap[30:,:-1])

data = np.loadtxt("dish_zenith.txt") #Importing data

"""
We now carry out our fit. First, we define our predicted parabaloid function that is linear in our new parameters. 
"""

def para(x,y,x0,y0,c0,a1,a2):
    return a1*x**2+a2*y**2+x0*x+y0*y+c0 #returning z-values

def para1(u,v,a,z0):
    return a*(u**2+v**2)+z0

def linear(d,vars): #vars is the number of variables we are fitting i.e. x^2, x, y^2, y
    #vec = d[:,:2].reshape(len(d)*2) #We want to just use the x,y columns.
    A = np.empty(shape=(len(d),vars+1)) #A = [1,x^2,x,y^2,y]
    A[:,0], A[:,1], A[:,2],A[:,3], A[:,4] = 1.0, d[:,0]**2, d[:,0], d[:,1]**2, d[:,1]
    cov = np.linalg.inv(A.transpose() @ A)
    m = cov @ (A.transpose() @ d[:,2])
    return m

#Plotting the original data in 3D
x,y,z = data[:,0], data[:,1], data[:,2]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(x,y,z,c=-z,s=30,cmap=cmap,label="Data")

#Now we create our surface using the our linear interpolation for z. We want to take x,y samples in a circle for best comparison
c0,a1,x0,a2,y0 = linear(data,4)

r = ((y.max()-y.min())+(x.max()-x.min()))/4 #Radius is just distance between xmin and xmax and ymin and ymax. Take average for best results.

R = np.linspace(0,r,50)
the = np.linspace(0,2*np.pi,50)

xl = []
yl = []
for i in R:
    for j in the:
        xl.append(i*np.cos(j))
        yl.append(i*np.sin(j))
X,Y = np.meshgrid(xl,yl,sparse=True)
Z = para(X,Y,x0,y0,c0,a1,a2)
ax.plot_wireframe(X,Y,Z,color="grey",rcount=10,ccount=10,label="Linear Fit")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()