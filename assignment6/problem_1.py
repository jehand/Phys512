import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy.random import randint

x,y,z = np.loadtxt("rand_points.txt",delimiter=" ",unpack=True) #Importing the data

#plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection="3d") 
ax.scatter3D(x,y,z,c=z,s=2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.view_init(1, -120)
#plt.savefig("plots/problem_1_planes.png",bbox_inches="tight",dpi=500)
plt.show()

#Checking pythons random number generator
xpy,ypy,zpy = randint(0,10**8,size=(3,len(x)))
fig = plt.figure()
ax = fig.add_subplot(projection="3d") 
ax.scatter3D(xpy,ypy,zpy,c=zpy,s=2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.view_init(1, -120)
#plt.savefig("plots/problem_1_pythonrandom.png",bbox_inches="tight",dpi=500)
plt.show()

#Checking local machine random numbers
xlocal,ylocal,zlocal = np.loadtxt("rand_points_local_machine.txt",delimiter=" ",unpack=True) #Importing the data
fig = plt.figure()
ax = fig.add_subplot(projection="3d") 
ax.scatter3D(xlocal,ylocal,zlocal,c=zlocal,s=2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.view_init(1, -120)
#plt.savefig("plots/problem_1_localmachine.png",bbox_inches="tight",dpi=500)
plt.show()