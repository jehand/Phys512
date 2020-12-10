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

#Set the parameters. Do not define r and v so that they are generated randomly.
npart = 100
r = []
v = []
m = None
time = 50
dt = 0.001
size = 50
G = 1 #cons.G #Importing the value of G

ani = Animation(m=m,npart=npart,size=size,dt=dt,G=G,bc_type="normal")
ani.animate(time=time,save_plt=False)