import numpy as np

#------------------------------------------------------------------------------
# Making a class to calculate the values of mass according to a 1/k^3 distribution as shown here: https://garrettgoon.com/gaussian-fields/. 
#------------------------------------------------------------------------------

class mass:
    """
    INPUT PARAMETERS:
    -----------
    r : array-like
        The position of the particles as r = [x,y,z].

    npart : int
        The number of particles in the simulation.

    size : int
        The size of the grid within which you want to generate particles.
    """

    def __init__(self,r=[],npart=10,size=50):
        self.r = r.copy()
        self.npart = npart
        self.size = size


    def white_noise(self): #Calculating our white noise on the grid using a normal distribution
        white = np.random.randn(self.size,self.size,self.size)
        whitefft = np.fft.fft(white)
        self.white = white.copy()
        self.whitefft = whitefft.copy()

    def prob(self):
        N = self.size
        first = np.arange(0,N/2+1) #We add one so that we get the values at the boundary as well.
        second = np.arange(-N/2 + 1,0)
        kvector = (2*np.pi / N) * np.stack((first,second),axis=0) #stack might be an error
        kvector[kvector==0] = np.inf
        power = 1/np.abs(kvector**3)


        

    