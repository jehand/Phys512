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

#Set the parameters. Do not define r and v so that they are generated randomly inside the class.
npart = 200
r = []
v = []
m = None
time = 300
dt = 0.01
size = 10
G = 1 #cons.G #Importing the value of G

ani = Animation(m=m,npart=npart,size=size,dt=dt,G=G,bc_type="periodic")
ani.animate(time=time,save_plt=False)