import numpy as np
from animation_class import Animation
import scipy.constants as cons

#Set the parameters. We set it up so that there is no motion in the z-axis for simplicity.
npart = 2
size = 6
G = 1 #cons.G #Importing the value of G
m = np.ones(npart) #Defining our masses
first = [size//2, 2.5, size//2] #position of first particle
second = [size//2, 3.5,size//2] #position of second particle
r = np.asarray([first,second]).T
#vv = np.sqrt(G/(abs(first[1]-second[1]) * (m[0]+m[1])))
vv = np.sqrt(G*(m[0]+m[1])/(-first[1]+second[1])**3) * (size/2 - first[1])
first_v = [vv*m[1],0,0] #velocity of first particleµ
second_v = [-vv*m[0],-0,0] #velocity of second particle whereby we are starting with 0 x-velocity and y-velocity.
v = np.asarray([first_v,second_v]).T / (np.pi**(3/2))
time = 100
dt = 0.05

ani = Animation(r=r,v=v,m=m,npart=npart,size=size,dt=dt,G=G,softening=0.1,bc_type="normal")
ani.animate(time=time,save_plt=False)