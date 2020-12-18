import numpy as np
from animation_class import Animation
import scipy.constants as cons

#Set the parameters. Do not define r and v so that they are generated randomly inside the class.
npart = 125000
r = []
v = []
m = []
time = 300
dt = 0.1
size = 50
G = 1 #cons.G #Importing the value of G

ani = Animation(m=m,npart=npart,size=size,dt=dt,G=G,softening=0.1,bc_type="periodic")
ani.animate(time=time,save_plt=False)