import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nbody_class import Nbody
from mpl_toolkits.mplot3d import Axes3D
from animation_class import Animation
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#Set the parameters. Do not define r and v so that they are generated randomly.
r = None
v = None
npart = 20
m = np.ones(npart)
time = 20
dt = 1
size = 50
G = 1

ani = Animation(m=m,npart=npart,size=size,dt=dt,G=G)
ani.animate(time=time,save_plt=False)