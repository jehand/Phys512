import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self,r=0,v=0,m=1,G=1,npart=10,softening=1e-3,size=50,dt=0.1,bc_type="normal"):
        self.m = m.copy()
        self.G = G.copy()
        self.npart = npart.copy()
        self.softening = softening.copy()
        self.size = size.copy()
        self.dt = dt.copy()
        self.bc_type = bc_type.copy()
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
            self.r = np.random.randint(0,self.size,self.npart,size=(3,self.npart))
        self.x, self.y, self.z = self.r[:,0], self.r[:,1], self.r[:,2]
        
        if v:
            if isinstance(r,(np.ndarray)): #Checking if is an ndarray or the code will not work
                self.v = v.copy()
            else:
                try:
                    self.v = np.ndarray(v) #Converting to ndarray if it is not
                else:
                    print("An exception occurred: v is not of the form np.ndarray")
                    quit()
        else:
            self.v = np.random.randint(-1,1,self.npart,size=(3,self.npart))
        self.vx, self.vy, self.vz = self.v[:,0], self.v[:,1], self.v[:,2]
        self.acc = np.zeros([npart,3])

    def Greens_function(self):
        ticks = np.linspace(0,self.size-1,self.size)
        kx, ky, kz = np.meshgrid(ticks,ticks,ticks)
        greens = 1/(4*np.pi*np.sqrt(kx**2 + ky**2 + kz**2 + self.softening**2))
        greens[0,0,0] = 0
        greens += np.flip(greens,0)
        greens += np.flip(greens,1)
        self.greens = greens.copy()

    def get_dens_field(self):
        #We can get our grid using Green's function and set the singularity ourselves to 0.
        if self.bc_type == "normal":
            grid = np.histogramdd(np.round(self.x).astype(int),np.round(self.y).astype(int),np.round(self.z).astype(int), bins=self.size, range=[[0,self.size],[0,self.size],[0,self.size]], weights=self.m)[0]
        if self.bc_type == "periodic":
            grid = np.histogramdd((np.round(self.x)%self.size).astype(int),(np.round(self.y)%self.size).astype(int),(np.round(self.z)%self.size).astype(int), bins=self.size, range=[[0,self.size],[0,self.size],[0,self.size]], weights=self.m)[0]
        rho = grid.copy()
        return rho

    def get_potential(self,dens_field,greens):
        dens_field_fft = np.fft.rfftn(dens_field)
        potential_fft = np.fft.rfftn(greens)
        potential = np.irfftn(dens_field_fft * potential_fft) #Convolving the density field with the potential for a particle
        return potential

    def get_forces(self,potential,grid):
        #We apply the leapfrog method later in the evolve_system stage
        self.F = -np.gradient(potential) * self.G
        self.Fx, self.Fy, self.Fz = self.F[:,0], self.F[:,0], self.F[:,0]
        """
        self.vx += self.Fx[(np.round(self.x)%self.size).astype(int),(np.round(self.y)%self.size).astype(int),(np.round(self.z)%self.size).astype(int)]*self.dt
        self.vy += self.Fy[(np.round(self.x)%self.size).astype(int),(np.round(self.y)%self.size).astype(int),(np.round(self.z)%self.size).astype(int)]*self.dt
        self.vz += self.Fz[(np.round(self.x)%self.size).astype(int),(np.round(self.y)%self.size).astype(int),(np.round(self.z)%self.size).astype(int)]*self.dt

        self.x_new += self.vx * self.dt
        self.y_new += self.vy * self.dt
        self.z_new += self.vz * self.dt"""
        #return self.x_new, self.y_new, self.z_new

    def energy(self,potential):
        #Calculate the energy of the system at each stage
        kinetic = 0.5 * np.sum(self.m * np.sqrt(self.vx**2 + self.vy**2 + self.vz**2))
        potential = -0.5*np.sum(potential.copy())
        total = kinetic + potential
        return kinetic, potential, total

    def leap_frog(self,r,v,a,a_new,dt):
        #Evolve the system by ways of the leapfrog method
        r_new = r + v*dt + 0.5*a*(dt**2)
        v_new = v + 0.5*(a+a_new)*dt
        return r_new,v_new

    def evolve_system(self):
        #This is the function that will evolve the system
        dens = self.get_dens_field()
        pot = self.get_potential(dens,self.greens)
        force = self.get_forces(pot,dens)
        self.a_new = self.F / self.m[:,None]
        self.r, self.v = self.leap_frog(self.r, self.v, self.acc, self.acc_new,self.dt)
        
        #Change the value of a now
        #Note that for non-periodic boundary conditions we want to remove the particles that leave the grid
        if self.bc_type == "normal":
            ind = np.argwhere()