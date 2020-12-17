import numpy as np
import matplotlib.pyplot as plt
from numba import *

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
    """

    def __init__(self,r=[],v=[],m=[],G=1,npart=10,softening=0.1,size=50,dt=0.1,bc_type="normal"):
        self.G = G
        self.npart = npart
        #Defining values for m if not provided to be 1 for each particle
        if len(m) != 0:
            self.m = m.copy()
        else:
            self.m = np.ones(self.npart)
        self.softening = softening
        self.size = size
        self.dt = dt
        self.bc_type = bc_type
        self.r = r.copy()
        self.x, self.y, self.z = self.r[0], self.r[1], self.r[2]
        self.v = v.copy()
        self.vx, self.vy, self.vz = self.v[0], self.v[1], self.v[2]
        self.acc = np.zeros([3,self.npart])
        self.greens = self.Greens_function()
        if self.bc_type == "periodic":
            self.greens_fft = np.fft.rfftn(self.greens)
        else:
            self.greens_fft = np.fft.rfftn(self.greens,[2*self.size,2*self.size,2*self.size])
        self.karray = [] #Defining array for KE
        self.parray = [] #Defining array for PE
        self.tarray = [] #Defining array for Total Energy

    def Greens_function(self):
        #We can get our grid using Green's function and set the singularity ourselves.
        ticks = np.arange(self.size)
        kx, ky, kz = np.meshgrid(ticks,ticks,ticks)
        norm = 4*np.pi*self.G*np.sqrt(kx**2 + ky**2 + kz**2 + self.softening**2)
        #norm[norm < self.softening] = 4*np.pi*self.G*self.softening
        greens = 1/(norm)
        greens += np.flip(greens,0)
        greens += np.flip(greens,1)
        greens += np.flip(greens,2)
        return greens

    def get_dens_field(self): #Possibility of using Numba? Removed the %, because it updates anyways.
        #np.round(self.x).astype(int),np.round(self.y).astype(int),np.round(self.z).astype(int)
        grid, edges = np.histogramdd([self.x,self.y,self.z], bins=self.size, range=[[0,self.size],[0,self.size],[0,self.size]], weights=self.m)
        self.grid = grid.copy()
        self.edges = edges.copy()
        return self.grid

    def get_potential(self,dens_field): #probably need to zero pad if our conditions are different
        if self.bc_type == "periodic":
            dens_field_fft = np.fft.rfftn(dens_field)
            potential = np.fft.fftshift(np.fft.irfftn(dens_field_fft * self.greens_fft)) #Convolving the density field with the potential for a particle
        else: #so that we pad
            dens_field_fft = np.fft.rfftn(dens_field,[2*self.size,2*self.size,2*self.size])
            potential = np.fft.fftshift(np.fft.irfftn(dens_field_fft * self.greens_fft)[:self.size,:self.size,:self.size]) #Convolving the density field with the potential for a particle
        potential = 0.5*(np.roll(potential,1,axis=0)+potential)
        potential = 0.5*(np.roll(potential,1,axis=1)+potential)
        potential = 0.5*(np.roll(potential,1,axis=2)+potential)
        """if self.bc_type == "normal": #Setting the value to 0 on the edge of the boundaries; possible improvements by convolving with window?
            potential[0:,0,0] = 0
            potential[0:,-1,0] = 0
            potential[0:,-1,-1] = 0
            potential[0:,0,-1] = 0
            potential[0,0:,0] = 0
            potential[-1,0:,0] = 0
            potential[0,0:,-1] = 0
            potential[-1,0:,-1] = 0
            potential[0,0,0:] = 0
            potential[-1,0,0:] = 0
            potential[0,-1,0:] = 0
            potential[-1,-1,0:] = 0
            potential[-1,-1,-1] = 0"""
        self.potential = potential.copy()
        return potential

    def get_forces(self,potential):
        #We can calculate the gradient as f'(x) = f(x+dx) - f(x-dx) / 2dx which gives us the following
        self.Fx = -0.5 * (np.roll(potential, 1, axis = 0) - np.roll(potential, -1, axis=0)) * self.grid
        self.Fy = -0.5 * (np.roll(potential, 1, axis = 1) - np.roll(potential, -1, axis=1)) * self.grid
        self.Fz = -0.5 * (np.roll(potential, 1, axis = 2) - np.roll(potential, -1, axis=2)) * self.grid
        print(max(abs(np.ravel(self.Fx))),max(abs(np.ravel(self.potential))))

    def energy(self):
        #Calculate the energy of the system at each stage
        kinetic = 0.5 * np.sum(self.m * (self.vx**2 + self.vy**2 + self.vz**2))
        potential = -0.5 * np.sum(self.potential * self.grid) 
        total = kinetic + potential
        self.karray.append(kinetic)
        self.parray.append(potential)
        self.tarray.append(total)

    def leap_frog(self,r,v,a,a_new,dt):
        #Evolve the system by ways of the leapfrog method
        r_new = r + v*dt + 0.5*a*(dt**2)
        v_new = v + 0.5*(a+a_new)*dt
        return r_new,v_new

    def evolve_system(self):
        #This is the function that will evolve the system
        dens = self.get_dens_field()
        pot = self.get_potential(dens)
        self.get_forces(pot)
        self.acc_new = np.zeros([3,self.npart]) #does not need to be a self.
        """for i in range(self.npart): #update:removed %self.size for each self.r
            self.acc_new[i][0] += self.Fx[(np.floor(self.x[i])).astype("int64"),(np.floor(self.y[i])).astype("int64"),(np.floor(self.z[i])).astype("int64")] / self.m[i]
            self.acc_new[i][1] += self.Fy[(np.floor(self.x[i])).astype("int64"),(np.floor(self.y[i])).astype("int64"),(np.floor(self.z[i])).astype("int64")] / self.m[i]
            self.acc_new[i][2] += self.Fz[(np.floor(self.x[i])).astype("int64"),(np.floor(self.y[i])).astype("int64"),(np.floor(self.z[i])).astype("int64")] / self.m[i]"""
        #Let's try np.digitize
        part_indx = np.digitize(self.x,bins=self.edges[0],right=True)
        part_indy = np.digitize(self.y,bins=self.edges[1],right=True)
        part_indz = np.digitize(self.z,bins=self.edges[2],right=True)
        self.acc_new[:][0] = self.Fx[part_indx, part_indy, part_indz]/self.m
        self.acc_new[:][1] = self.Fy[part_indx, part_indy, part_indz]/self.m
        self.acc_new[:][2] = self.Fz[part_indx, part_indy, part_indz]/self.m

        self.r, self.v = self.leap_frog(self.r, self.v, self.acc, self.acc_new, self.dt)
        #Note that for non-periodic boundary conditions we want to remove the particles that leave the grid
        if self.bc_type == "periodic":
            self.r = self.r % (self.size-1)
        if self.bc_type == "normal":
            ind_top = np.argwhere((self.r > self.size -1))
            ind_bot = np.argwhere((self.r < 0))
            ind = [i for i in np.append(ind_top,ind_bot)]
            self.v = np.delete(self.v,ind,axis=1)
            self.acc_new = np.delete(self.acc_new,ind,axis=1)
            self.m = np.delete(self.m,ind,axis=0)
            self.r = np.delete(self.r,ind,axis=1)
            self.acc = np.delete(self.acc,ind,axis=1)
            self.npart = len(self.m)
        self.acc = self.acc_new.copy() #Change the value of a
        self.x, self.y, self.z = self.r[0].copy(), self.r[1].copy(), self.r[2].copy()
        self.vx, self.vy, self.vz = self.v[0].copy(), self.v[1].copy(), self.v[2].copy()

#Create some stuff with numba so that it runs faster here