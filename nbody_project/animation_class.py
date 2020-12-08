import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nbody_class import Nbody
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#------------------------------------------------------------------------------
# Making a class to produce a constant evolving plot of the nBody simulation from class Nbody. This will be the only classs that needs to be called 
# to run the simulation.
#------------------------------------------------------------------------------

class Animation:
    """
    INPUT PARAMETERS:
    -----------
    r : array-like
        The position of the particles as r = [x,y,z]. Can also be set to 0 if you want uniform random integer positions generated between 0 and the size of the grid.

    v : array-like
        The velocities of the particles as v = [vx,vy,vz]. Can also be set to 0 if you want uniform random integer velocities generated between -1 and 1.

    m : array-like
        The mass of the particles (set to 1 for all particles for ease, but can be changed).

    G : int
        Value of Newton's Gravitational Constant that you would like to use

    npart : int
        The number of particles in the simulation.

    softening : float
        The value of epsilon added into our force law such that our simulation does not act weird at large potentials.

    size : int
        The size of the grid within which you want to generate particles.

    dt : float
        Size of the time steps to take.

    bc_type : string
        Sets whether we are going to use periodic or non-perioud boundary conditions. "periodic" meaning periodic and "normal" meaning non-periodic.

    time : int
        The amount of time over which you want to run the simulation for, i.e. time/dt is the number of iterations.

    save_plt : Boolean
        Decides whether to save the plot produced as a .gif. False means it is not saved and vice versa.
    """

    def __init__(self,r=None,v=None,m=1,G=1,npart=10,softening=1e-3,size=50,dt=0.1,bc_type="normal"):
        self.m = m
        self.G = G
        self.npart = npart
        self.softening = softening
        self.size = size
        self.dt = dt
        self.bc_type = bc_type
        if r:
            if isinstance(r,(np.ndarray)): #Checking if is an ndarray or the code will not work
                self.r = r.copy()
            else:
                try:
                    self.r = np.ndarray(r) #Converting to ndarray if it is not
                except:
                    print("An exception occurred: r is not of the form np.ndarray")
                    quit()
        else:
            self.r = np.random.randint(0,self.size,size=(3,self.npart))
        self.x, self.y, self.z = self.r[0], self.r[1], self.r[2]
        
        if v:
            if isinstance(r,(np.ndarray)): #Checking if is an ndarray or the code will not work
                self.v = v.copy()
            else:
                try:
                    self.v = np.ndarray(v) #Converting to ndarray if it is not
                except:
                    print("An exception occurred: v is not of the form np.ndarray")
                    quit()
        else:
            self.v = np.random.randint(-1,1,size=(3,self.npart))
        self.vx, self.vy, self.vz = self.v[0], self.v[1], self.v[2]
        
        #Call the class Nbody with our current settings
        self.particles = Nbody(self.r, self.v, self.m, self.G, self.npart, self.softening, self.size, self.dt, self.bc_type)

    def animate(self,time=50,save_plt=False):
        tf.reset_default_graph()
        for i in np.arange(0,time,self.dt):
            density = self.particles.get_dens_field()
            self.particles.evolve_system(density)
            tfdensity = tf.constant(abs(density))
            with tf.Session() as sess:
                istate = sess.run(tfdensity)
            plt.clf()
            fig=plt.figure(figsize=(10,10))#Create 3D axes
            try: ax=fig.add_subplot(111,projection="3d")
            except : ax=Axes3D(fig)
            ax.scatter(istate[0, 0,:,0],istate[0, 0,:,1], istate[0, 0,:,2],color="royalblue",marker=".",s=.5)
            ax.set_xlabel("x-coordinate",fontsize=14)
            ax.set_ylabel("y-coordinate",fontsize=14)
            ax.set_zlabel("z-coordinate",fontsize=14)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            #plt.imshow(abs(density))
            plt.pause(0.001)
            plt.show()