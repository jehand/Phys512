import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x,y,z = np.loadtxt("rand_points.txt",delimiter=" ",unpack=True) #Importing the data

fig = plt.figure()
ax = plt.axes(projection="3d") 
ax.scatter3D(x,y,z,c=z,s=15)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()