import numpy as np
from animation_class import Animation
import scipy.constants as cons

#Set the parameters. Do not define r and v so that they are generated randomly inside the class.
npart = 150000
r = []
v = []
m = []
time = 3000000
dt = 10000
size = 50
G = 1 #cons.G #Importing the value of G

ani = Animation(m=m,npart=npart,size=size,dt=dt,G=G,softening=100,bc_type="periodic",early_universe=True)
ani.animate(time=time,save_plt=False)