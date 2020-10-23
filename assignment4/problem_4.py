import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt

#To get past the wrapping around nature of a Fourier transform, we use zero padding. Assume both arrays are the same length.
def conv(arr1,arr2):
    f1 = fft(arr1,len(arr1)*2) #using the in-built padding provided by np.fft.fft
    f2 = fft(arr2,len(arr1)*2)
    transform = ifft(f1*f2)
    return transform[:len(arr1)]

#Let's take an example to check if this works. We use the same example from class.
x=np.arange(1000)
tau=50
nhit=200
x_hit=np.asarray(np.floor(len(x)*np.random.rand(nhit)),dtype='int')
y_hit=np.random.rand(nhit)**2
T=0.0*x
for i in range(nhit):
    mylen=len(x)-x_hit[i]
    T[x_hit[i]:]=T[x_hit[i]:]+y_hit[i]*np.exp(-np.arange(mylen)/tau)

#Normal convolution
f=0.0*x
for i in range(nhit):
    f[x_hit[i]]=f[x_hit[i]]+y_hit[i]
g=np.exp(-1.0*x/tau)
T2=np.fft.irfft(np.fft.rfft(g)*np.fft.rfft(f))
T2=T2[:len(T)]

#Now we pad with our function
T3 = conv(f,g)

#Plotting
plt.plot(x,T,"r-",linewidth=2,label="Original Data")
plt.plot(x,T2,color="darkblue",linewidth=2,label="Convolution without Padding")
plt.plot(x,T3,"g--",linewidth=2,label="Convolution with Padding")
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.legend(fontsize=12)
plt.savefig("problem_4.png",bbox_inches="tight",dpi=500)
plt.show()

#Make a residual plot so it's easier to see the difference
plt.plot(x,T2-T,color="darkblue",linewidth=2,label="Convolution without Padding")
plt.plot(x,T3-T,"g-",linewidth=2,label="Convolution with Padding")
plt.xlabel("x",fontsize=12)
plt.ylabel("Residual",fontsize=12)
plt.legend(fontsize=12)
plt.savefig("problem_4_res.png",bbox_inches="tight",dpi=500)
plt.show()