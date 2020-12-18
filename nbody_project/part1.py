import numpy as np
from animation_class import Animation
import scipy.constants as cons

#Set the parameters. Do not define r and v so that they are generated randomly inside the class.
npart = 1
size = 50
r = np.asarray([[size//2, size//2, size//2]]).T
v = []
m = []
time = 300
dt = 1
G = 1 #cons.G #Importing the value of G

ani = Animation(r=r,m=m,npart=npart,size=size,dt=dt,G=G,softening=0.01,bc_type="periodic")
ani.animate(time=time,save_plt=True)