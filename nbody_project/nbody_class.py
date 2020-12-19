import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mass_class import mass

#------------------------------------------------------------------------------
# The first step is to design my own n-body class. I want to go about this using a Leapfrog scheme with a softened potential whereby under a certain 
# value a0, the potential goes to some constant value. I will attempt to do this by arbitrarily deciding this point and seeing what the best value 
# for this and a0 is. Every time a simulation is run, a plot is automatically displayed showing the evolution of the particles in the system as well as a side by side plot showing the kinetic, potential and total energy in the system.
#------------------------------------------------------------------------------

class Nbody:
    """
    INPUT PARAMETERS:
    -----------
    r : array-like
        The position of the particles as r = [x,y,z]. 

    v : array-like
        The velocities of the particles as v = [vx,vy,vz].

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

    early_universe : Boolean
        Decides whether we calculate mass fluctuations according to k^-3 or not. False means normal masses and True is according to k^-3.
    """

    def __init__(self,r=[],v=[],m=[],G=1,npart=10,softening=0.1,size=50,dt=0.1,bc_type="normal",early_universe=False):
        self.G = G
        self.npart = npart
        self.softening = softening
        self.size = size
        self.dt = dt
        self.bc_type = bc_type
        self.r = r.copy()
        self.x, self.y, self.z = self.r[0], self.r[1], self.r[2]
        self.v = v.copy()
        self.vx, self.vy, self.vz = self.v[0], self.v[1], self.v[2]
        self.acc = np.zeros([3,self.npart])
        self.early_universe = early_universe
        if len(m) != 0:
            self.m = m.copy()
        else:
            if self.early_universe: #Determine the masses according to this k^-3 power law
                masses = mass(self.r,self.npart,self.size)
                self.m = masses.find_m()
            else: #Defining values for m if not provided to be 1 for each particle
                self.m = np.ones(self.npart) 

        self.greens = self.Greens_function()
        self.greens_fft = np.fft.rfftn(self.greens)
        
        self.karray = [] #Defining array for KE
        self.parray = [] #Defining array for PE
        self.tarray = [] #Defining array for Total Energy

    def Greens_function(self):
        ticks = np.arange(self.size)
        kx,ky,kz = np.array(np.meshgrid(ticks,ticks,ticks)) + 0.5 - self.size/2 #to center the function
        norm = 4*np.pi*self.G*np.sqrt(kx**2+ky**2+kz**2+self.softening**2)
        greens = -1/(norm)
        greens += np.flip(greens,0) #applying our green function to each corner of our grid equally.
        greens += np.flip(greens,1)
        greens += np.flip(greens,2) 
        return greens

    def get_dens_field(self):
        grid, edges = np.histogramdd([self.x,self.y,self.z], bins=self.size, range=[[0,self.size],[0,self.size],[0,self.size]], weights=self.m)
        self.grid = grid.copy()
        self.edges = edges.copy()

    def get_potential(self): 
        dens_field_fft = np.fft.rfftn(self.grid)
        potential = np.fft.irfftn(self.greens_fft*dens_field_fft)
        potential = 0.5*(np.roll(potential,1,axis=0)+potential) #to center our potential in the middle of the grid/cells
        potential = 0.5*(np.roll(potential,1,axis=1)+potential)
        potential = 0.5*(np.roll(potential,1,axis=2)+potential)
        self.potential = potential.copy()
        return potential

    def get_forces(self,potential):
        #We can calculate the gradient as f'(x) = f(x+dx) - f(x-dx) / 2dx which gives us the following:
        if self.bc_type == "periodic":
            self.Fx = -0.5 * (np.roll(potential, 1, axis = 0) - np.roll(potential, -1, axis=0)) * self.grid
            self.Fy = -0.5 * (np.roll(potential, 1, axis = 1) - np.roll(potential, -1, axis=1)) * self.grid
            self.Fz = -0.5 * (np.roll(potential, 1, axis = 2) - np.roll(potential, -1, axis=2)) * self.grid
        else: #we want to pad our potential if it's a normal boundary so that there is no crossover when rolling, hence only need 1 pad each side
            n = 1
            potentialp = np.pad(potential, [[n,n],[n,n],[n,n]])
            self.Fx = -0.5 * (np.roll(potentialp, 1, axis = 0) - np.roll(potentialp, -1, axis=0))[n:-n,n:-n,n:-n] * self.grid
            self.Fy = -0.5 * (np.roll(potentialp, 1, axis = 1) - np.roll(potentialp, -1, axis=1))[n:-n,n:-n,n:-n] * self.grid
            self.Fz = -0.5 * (np.roll(potentialp, 1, axis = 2) - np.roll(potentialp, -1, axis=2))[n:-n,n:-n,n:-n] * self.grid

    def energy(self):
        #Calculate the energy of the system at each stage
        kinetic = 0.5 * np.sum(self.m * (self.vx**2 + self.vy**2 + self.vz**2))
        potentialen = -0.5 * np.sum(self.potential * self.grid) 
        total = kinetic + potentialen
        self.karray.append(kinetic)
        self.parray.append(potentialen)
        self.tarray.append(total)

    def leap_frog(self,r,v,a,a_new,dt):
        #Evolve the system by ways of the leapfrog method
        r_new = r + v*dt + 0.5*a*(dt**2)
        v_new = v + 0.5*(a+a_new)*dt
        return r_new,v_new

    def evolve_system(self):
        #This is the function that will evolve the system
        self.get_dens_field()
        pot = self.get_potential()
        self.get_forces(pot)
        acc_new = np.zeros([3,self.npart])

        #Let's try np.digitize
        part_indx = np.digitize(np.floor(self.x),bins=self.edges[0],right=True)
        part_indy = np.digitize(np.floor(self.y),bins=self.edges[1],right=True)
        part_indz = np.digitize(np.floor(self.z),bins=self.edges[2],right=True)
        acc_new[0][:] = self.Fx[part_indx, part_indy, part_indz]/self.m
        acc_new[1][:] = self.Fy[part_indx, part_indy, part_indz]/self.m
        acc_new[2][:] = self.Fz[part_indx, part_indy, part_indz]/self.m
        
        #print(acc_new)
        self.r, self.v = self.leap_frog(self.r, self.v, self.acc, acc_new, self.dt)
        if self.bc_type == "periodic": #For periodic B.C's, we take the modulo so that it appears on the other side of the grid.
            self.r = self.r % (self.size-1)
        if self.bc_type == "normal": #For non-periodic boundary conditions we want to remove the particles that leave the grid
            ind_top = np.argwhere(self.r >= self.size)
            ind_bot = np.argwhere(self.r < 0)
            ind = [i for i in np.append(ind_top,ind_bot)]
            self.v = np.delete(self.v,ind,axis=1)
            acc_new = np.delete(acc_new,ind,axis=1)
            self.m = np.delete(self.m,ind)
            self.r = np.delete(self.r,ind,axis=1)
            self.acc = np.delete(self.acc,ind,axis=1)
            self.npart = len(self.m)
        self.acc = acc_new.copy() #Change the value of a
        self.x, self.y, self.z = self.r[0].copy(), self.r[1].copy(), self.r[2].copy()
        self.vx, self.vy, self.vz = self.v[0].copy(), self.v[1].copy(), self.v[2].copy()