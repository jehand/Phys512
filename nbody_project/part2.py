import numpy as np
from animation_class import Animation
import scipy.constants as cons

#Set the parameters. We set it up so that there is no motion in the z-axis for simplicity.
npart = 2
size = 8
m = np.asarray([10,1]) #Defining our masses
first = [size//2, size//2, size//2] #position of first particle
second = [size//2,3*size//4,size//2] #position of second particle
r = np.asarray([first,second]).T
first_v = [0,0,0] #velocity of first particle
second_v = [-2*np.sqrt(m[0]/size),0,0] #velocity of second particle whereby we are starting with 0 x-velocity and y-velocity.
v = np.asarray([first_v,second_v]).T
time = 300
dt = 0.1
G = 1 #cons.G #Importing the value of G

#print("r=",r)
#print("v=",v)
ani = Animation(r=r,v=v,m=m,npart=npart,size=size,dt=dt,G=G,bc_type="normal")
ani.animate(time=time,save_plt=False)