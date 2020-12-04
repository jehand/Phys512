import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class particles:
    def __init__(self,m=1.0,npart=10,soft=1,G=1.0,dt=0.1,grid_size=50):
        
        #initializing partcle settings
        self.opts={}
        self.opts['soft']=soft
        self.opts['n']=npart
        self.opts['G']=G
        self.opts['dt']=dt
        self.opts['grid_size']=grid_size

        self.opts['m']=m
        self.x,self.y=np.random.randint(0, self.opts['grid_size'], size=(2, self.opts['n']))
        #self.m=m
        self.vx=np.double(self.x.tolist())*0
        self.vy=np.double(self.y.tolist())*0
        
        #creating greens function
        xx= np.linspace(0, grid_size-1, grid_size)
        kx,ky=np.meshgrid(xx,xx)
        
        Gr=1/(1e-13+4*np.pi*((kx)**2+(ky)**2)**(1/2))
        Gr[0,0]=1/(4*np.pi*soft)
        
        #making the function apply to every corner
        Gr+=np.flip(Gr,0)
        Gr+=np.flip(Gr,1)
        #fourier transform to convolve
        Gr_ft=np.fft.fft2(Gr)
        self.Gr_ft=Gr_ft
        self.kin_energy=0
        
    def get_grid(self,x,y):
        
        grid_size=self.opts['grid_size']
    
        #density grid
        #the boundary conditions are represented by the % operator
        #using the closest distance method by rounding the particle location
        A=np.histogram2d((np.round(x)%grid_size).astype(int),(np.round(y)%grid_size).astype(int),bins=grid_size,range=[[0, grid_size], [0, grid_size]])[0]*self.opts['m']

#       Convolving the greens function with the density grid
        Gr_ft=self.Gr_ft
        rho_ft=np.fft.fft2(A)
    
        conv=Gr_ft*rho_ft
        pot=np.fft.ifft2(conv)
        #making sure the potential is centered with the particle positions
        pot=0.5*(np.roll(pot,1,axis=1)+pot)
        pot=0.5*(np.roll(pot,1,axis=0)+pot)
        
        return A,pot
        
    def get_force(self,x,y,pot,A):
        
        #taking the gradient of the potential to get the forces
        #multiplying by density matrix
        forcex=-1/2*(np.roll(pot,1,axis=0)-np.roll(pot,-1,axis=0))*A
        forcey=-1/2*(np.roll(pot,1,axis=1)-np.roll(pot,-1,axis=1))*A
        
        x_new=np.double(x.tolist())*0
        y_new=np.double(y.tolist())*0
        
        #changing the velocities of each particle by taking the force of each particle and multiplying it by time
        #changing the positions of each particle with the new velocity
        self.vx+=np.real(forcex[(np.round(x)%self.opts['grid_size']).astype(int),(np.round(y)%self.opts['grid_size']).astype(int)])*self.opts['dt']
        x_new=x+self.vx*self.opts['dt']
        self.vy+=np.real(forcey[(np.round(x)%self.opts['grid_size']).astype(int),(np.round(y)%self.opts['grid_size']).astype(int)])*self.opts['dt']
        y_new=y+self.vy*self.opts['dt']
        
        #getting the kinetic energy
        self.kin_energy=1/2*np.sum(self.vx**2+self.vy**2)*m

        return x_new,y_new
        
if __name__=='__main__':
     
    #setting the parameters
    n=50
    grid_size=100
    m=20
    time=300
    dt=0.01
    #initializing the particle
    part=particles(m=m,npart=n,dt=dt,grid_size=grid_size)
    
    #getting the density and the potential
    A,pot=part.get_grid(part.x,part.y)
    x_new,y_new=part.get_force(part.x,part.y,pot,A)
    
    
    count=0
    kin_energy=np.zeros(int(time//dt))

    for i in np.arange(0,time,part.opts['dt']):
        
        A_new,pot=part.get_grid(x_new,y_new)

        x_new,y_new=part.get_force(x_new,y_new,pot,A_new)
        
        kin_energy[count]=np.real(part.kin_energy)
        plt.clf()
        plt.imshow(abs(A_new))
        plt.pause(0.0001)
        
        count+=1
        
        
    print(x_new-part.x)
    print(y_new-part.y)