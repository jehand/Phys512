import numpy as np
from animation_class import Animation
import scipy.constants as cons

#Set the parameters. Do not define r and v so that they are generated randomly inside the class.
npart = 150000
size = 128 #give powers of 2 for this one 
r = np.random.randint(0,size,size=(3,npart)) + 0.5 #starting particles at the center of grid cells
v = []
m = []
time = 10
dt = 0.1
G = 1 #cons.G #Importing the value of G

ani = Animation(r=r,m=m,npart=npart,size=size,dt=dt,G=G,softening=0.1,bc_type="periodic",early_universe=True)
ani.animate(time=time,save_plt=False)