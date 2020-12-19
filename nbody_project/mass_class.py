import numpy as np
import matplotlib.pyplot as plt

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

    plot : Boolean
        True plots the heatmap of the inverse Fourier transformed power spectrum and saves it.
    """

    def __init__(self,r=[],npart=10,size=50):
        self.r = r.copy()
        self.npart = npart
        self.size = size
        self.white_noise()
        self.prob()

    def white_noise(self): 
        #Calculating our white noise on the grid using a normal distribution
        white = np.random.randn(self.size,self.size,self.size)
        white_fft = np.fft.fftn(white)
        self.white = white.copy()
        self.white_fft = white_fft.copy()

    def prob(self):
        #Calculates the probability distribution according to k^-3
        N = self.size
        first = np.arange(0,N/2+1) #We add one so that we get the values at the boundary as well.
        second = np.arange(-N/2 + 1,0)
        kvector = (2*np.pi / N) * np.concatenate((first,second))
        ka, kb, kc = np.meshgrid(kvector,kvector,kvector)
        k = np.sqrt(ka**2 + kb**2 + kc**2)
        power = np.sqrt(1/(k**3))
        power[0,0,0] = 0
        massgrid = np.fft.irfftn(power * self.white_fft)

        #So far it has just been a rewording of the provided document; now we determine the probability by dividing by the range
        self.massprob = (massgrid-massgrid.min())/(massgrid.max()-massgrid.min())

    def find_cell(self):
        #Used to find the grid cell corresponding to each particle for the potential 
        edges = np.arange(self.size+1)
        part_indx = np.digitize(np.floor(self.r[0]),bins=edges,right=True)
        part_indy = np.digitize(np.floor(self.r[1]),bins=edges,right=True)
        part_indz = np.digitize(np.floor(self.r[2]),bins=edges,right=True)
        self.prob4part = self.massprob[part_indx,part_indy,part_indz]

    def find_m(self,plot=False):
        #Now we calculate our new masses; check which cell particle lies in and then assign it a mass
        #Generate uniformly with a probability, can do it from 0.1 so there is no divide by 0.
        #The maximum mass can just be set to 1 for ease.
        self.find_cell()
        
        m = np.ravel([np.random.uniform(0.1,self.prob4part[i],1) for i in range(self.npart)])
        
        if plot:
            #for plotting
            edges_x = np.arange(self.size)
            edges_y = np.arange(self.size)
            edges_z = np.arange(self.size)
            fig,ax = plt.subplots(figsize=(10,10), dpi=100) 
            im = ax.imshow(
                self.massprob.sum(axis=2),
                origin="lower",
                extent=(edges_y.min(), edges_y.max(), edges_x.min(), edges_x.max()), 
                aspect="auto"
            )
            fig.colorbar(im, orientation='horizontal')
            # A bit backwards, but need to be careful of coordinate definitions when collapsing...
            # When in doubt, just plot the histogram and double check the position of your points!
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            minor_ticks_x = edges_x
            minor_ticks_y = edges_y
            ax.set(xlim=(minor_ticks_y[0], minor_ticks_y[-1]), ylim=(minor_ticks_x[0], minor_ticks_x[-1]))
            ax.set_xticks(minor_ticks_y, minor=True)
            ax.set_yticks(minor_ticks_x, minor=True)
            # Add a grid
            ax.grid(which='both', alpha=0.75, color='w')
            plt.savefig("part4_invpowerspectrum.png")
            plt.show()
        return np.asarray(m)
