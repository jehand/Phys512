import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.dpi'] = 150
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nbody_class import Nbody
from matplotlib import gridspec
import decimal

#------------------------------------------------------------------------------
# Making a class to produce a constant evolving plot of the nBody simulation from class Nbody. This will be the only classs that needs to be called 
# to run the simulation. The animation will be produced using transparent voxels from Google's TensorFlow.
#------------------------------------------------------------------------------

class Animation:
    """
    INPUT PARAMETERS:
    -----------
    r : array-like
        The position of the particles as r = [x,y,z]. Can also be set to 0 if you want uniform random integer positions generated between 0 and the size of the grid-1.

    v : array-like
        The velocities of the particles as v = [vx,vy,vz]. Can also be set to 0 if you want your initial velocities to just be an array of 0's.

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

    early_universe : Boolean
        Decides whether we calculate mass fluctuations according to k^-3 or not. False means normal masses and True is according to k^-3.

    save_plt : Boolean
        Decides whether to save the plot produced as a .gif. False means it is not saved and vice versa.
    """

    def __init__(self,r=[],v=[],m=1,G=1,npart=10,softening=0.1,size=50,dt=0.1,bc_type="normal",early_universe=False):
        self.m = m
        self.G = G
        self.npart = npart
        self.softening = softening
        self.size = size
        self.dt = dt
        self.bc_type = bc_type
        self.early_universe = early_universe
        if len(r) != 0:
            if isinstance(r,(np.ndarray)): #Checking if is an ndarray or the code will not work
                self.r = r.copy()
            else:
                try:
                    self.r = np.asarray(r) #Converting to ndarray if it is not
                except:
                    print("An exception occurred: r is not of the form np.ndarray")
                    quit()
        else:
            self.r = np.random.randint(0,self.size-1,size=(3,self.npart))
        
        if len(v) != 0:
            if isinstance(r,(np.ndarray)): #Checking if is an ndarray or the code will not work
                self.v = v.copy()
            else:
                try:
                    self.v = np.asarray(v) #Converting to ndarray if it is not
                except:
                    print("An exception occurred: v is not of the form np.ndarray")
                    quit()
        else:
            self.v = np.zeros([3,self.npart])        

        #Call the class Nbody with our current settings
        self.particles = Nbody(self.r, self.v, self.m, self.G, self.npart, self.softening, self.size, self.dt, self.bc_type, self.early_universe)

    def animate(self,time=50,save_plt=False): #Possibility of upgrades with matplotlib.animation.FuncAnimation, scale opacity by density
        #for i in np.arange(0,time,self.dt):
        #    self.particles.evolve_system()
        plt.ion()
        fig=plt.figure(figsize=(6,6))#Create 3D axes
        gs = gridspec.GridSpec(ncols=1,nrows=2,figure=fig,height_ratios=[2,1])
        ax=fig.add_subplot(gs[0],projection="3d",autoscale_on=False)
        ax2=fig.add_subplot(gs[1])
        times = list(np.arange(0,time,self.dt))
        dt = decimal.Decimal(str(self.dt))
        dps = abs(dt.as_tuple().exponent)
        f = "{:."+str(dps)+"f}"
        for i in times:
            self.particles.evolve_system()
            ax.clear()
            ax.scatter(self.particles.x,self.particles.y,self.particles.z,color="royalblue",marker=".",s=20,alpha=1) #change the size depending on number of particles you have
            ax.axes.set_xlim3d(0,self.size)
            ax.axes.set_ylim3d(0,self.size)
            ax.axes.set_zlim3d(0,self.size)
            ax.set_xlabel("x",fontsize=11)
            ax.set_ylabel("y",fontsize=11)
            ax.set_zlabel("z",fontsize=11)
            ax.set_title("Time "+f.format(i),fontsize=12)

            #Plotting energy
            self.particles.energy()
            inds = times.index(i)+1
            ax2.clear()
            ax2.axhline(color="black")
            ax2.axes.set_xlim(0,time)
            ax2.plot(times[0:inds],self.particles.karray[0:inds],"r-",label="Kinetic Energy")
            ax2.plot(times[0:inds],self.particles.parray[0:inds],"b-",label="Potential Energy")
            ax2.plot(times[0:inds],self.particles.tarray[0:inds],"k-",label="Total Energy")
            ax2.set_xlabel("Time",fontsize=11)
            ax2.set_ylabel("Energy",fontsize=11)
            ax2.legend(loc="upper right",fontsize=8)
            plt.draw()
            plt.pause(0.01)

            if save_plt:
                plt.savefig("figures/part1/fig"+str(inds)+".png",bbox_inches="tight")