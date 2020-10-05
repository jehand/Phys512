import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl

cmap = mpl.cm.Greys(np.linspace(0,1,50)) #Next 2 lines are just to make the 3D plot easier to see
cmap = mpl.colors.ListedColormap(cmap[30:45,:-1])

data = np.loadtxt("dish_zenith.txt") #Importing data

"""
We now carry out our fit. First, we define our predicted parabaloid function that is linear in our new parameters. 
"""

def para(x,y,x0,y0,c0,a1,a2):
    return a1*x**2+a2*y**2+x0*x+y0*y+c0 #returning z-values

def para1(u,v,a,z0):
    return a*(u**2+v**2)+z0

def linear(d):
    A = np.empty(shape=(len(d),5)) #A = [1,x^2,x,y^2,y]
    A[:,0], A[:,1], A[:,2],A[:,3], A[:,4] = 1.0, d[:,0]**2, d[:,0], d[:,1]**2, d[:,1]
    cov = np.linalg.inv(A.transpose() @ A)
    m = cov @ (A.transpose() @ d[:,2])
    return m,cov

x,y,z = data[:,0], data[:,1], data[:,2]
c0,a1,x0,a2,y0 = linear(data)[0] #Getting our parameters
cov = linear(data)[1]
errs = np.sqrt(np.diag(cov))
a = (a1+a2)/2 #Calculating a as the average of both a values
aerr = (errs[1]+errs[3])/2 #Calculating error in a as the average of both uncerts for a1 and a2
print("c0=",c0,"a1=",a1,"x0=",x0,"a2=",a2,"y0=",y0,"a_mean=",a)
print("errs=",errs,"a_mean err=",aerr)

"""#Plotting the original data in 3D
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(x,y,z,c=-z,s=15,cmap=cmap,label="Data")
"""
"""
Now we create our surface using the our linear interpolation for z. We plot with cylindrical coordinates so that the meshgrid looks better, 
radius=(max_x-min_x)/2 ~ (max_y-min_y)/2. For accuracy we take the average of both and use this.
"""
"""rmax = ((x.max()-x.min())+(y.max()-y.min()))/4
r = np.linspace(0,rmax,1000)
thetha = np.linspace(0,2*np.pi,1000)
r, thetha = np.meshgrid(r,thetha)
X, Y = r*np.cos(thetha), r*np.sin(thetha) #calculating X and Y values
Z = para(X,Y,x0,y0,c0,a1,a2)
ax.plot_wireframe(X,Y,Z,color="teal",rstride=20,cstride=20,linewidth=0.7,alpha=0.7,label="Linear Fit")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_zlim(z.min(),z.max())
ax.tick_params(axis='both', which='major', labelsize=8)
ax.view_init(24,-45)
ax.legend(fontsize=8)
plt.savefig("problem_1_3D.png", dpi=500)
plt.show()"""

#Estimating the noise in the data
res = z-para(x,y,x0,y0,c0,a1,a2)
N = np.std(res,ddof=1) #This is the noise in our data
print("Noise=",N)
focal = 1/(4*a)
ufocal = aerr/(4*a**2)
print("Focal=",focal,"±",ufocal)

"""
#Plotting residuals for the fit
plt.clf()
plt.plot(x,res,"kx")
plt.ylabel("Residual",fontsize=12)
plt.xlabel("x", fontsize=12)
plt.savefig("problem_1_residual.png",bbox_inches="tight",dpi=500)
plt.show()"""