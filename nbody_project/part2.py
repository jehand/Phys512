import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nbody_class import Nbody
from mpl_toolkits.mplot3d import Axes3D
from animation_class import Animation
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.constants as cons

#Set the parameters. We set it up so that there is no motion in the z-axis for simplicity.
npart = 2
size = 8
first = [size//2, size//2, size//2] #position of first particle
second = [size//2,3*size//4,size//2] #position of second particle
r = np.asarray([first,second]).T
first_v = [0,0,0] #velocity of first particle
second_v = [-2*np.sqrt(10/size),0,0] #velocity of second particle whereby we are starting with 0 x-velocity and y-velocity.
v = np.asarray([first_v,second_v]).T
m = np.asarray([10,1]) #Defining our masses
time = 300
dt = 0.1
G = 1 #cons.G #Importing the value of G

#print("r=",r)
#print("v=",v)
ani = Animation(r=r,v=v,m=m,npart=npart,size=size,dt=dt,G=G,bc_type="periodic")
ani.animate(time=time,save_plt=False)