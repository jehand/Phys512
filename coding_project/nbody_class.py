import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# The first step is to design my own n-body class. I want to go about this using a Leapfrog scheme with a softened potential whereby under a certain 
# value a0, the potential goes to some constant value. I will attempt to do this by arbitrarily deciding this point and seeing what the best value 
# for this and a0 is. 
#------------------------------------------------------------------------------

class Nbody:
    """
    PARAMETERS:
    -----------
    x : array-like
        The position of the particles as x = [x,y,z]

    v : array-like
        The velocities of the particles as v = [vx,vy,vz]

    m : array-like
        The mass of the particles (set to 1 for all for ease, but can be changed)

    dens_field:
    """

    def __init__(self,x,v,m=1):
        self.x = x.copy()
        self.v = v.copy()
        self.m = m.copy()
        self.n = self.x.shape[0]


    def get_a(self):
        self.f = np.zeros(x.shape())

    def get_dens_field(self):


    def get_potential(self,dens_field,potential_one):
        self.dens_field = dens_field.copy()
        self.potential_one = potential_one.copy()
        dens_field_fft = np.fft.rfftn(self.dens_field)
        potential_one_fft = np.fft.rfftn(self.potential_one)
        self.potential = np.irfftn(self.dens_field_fft * self.potential_one_fft) #Convolving the density field with the potential for a particle

        return self.potential
