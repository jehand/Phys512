import numpy as np
from numpy.fft import fft,fftfreq
import matplotlib.pyplot as plt

#We arbitrarly choose a non-integer sine wave of f=7.7
f = 7.7
N = 1024 #arbitrarily chosen number of points
x = np.linspace(-5,10,N) #non-integer number of periods
y = np.sin(2*np.pi*f*x)

#Determining the fft from numpy (for all the following fft's we cut out the negative portion of the transform as it is symmetric anyways)
fftsine = fft(y)[:int(len(y)/2)]
freq = fftfreq(x.shape[-1])[:int(len(y)/2)]/((x.max()-x.min())/len(x))
plt.plot(freq,np.abs(fftsine),"r-",label="FFT",linewidth=2)

#Analytic estimate of the DFT
M = np.empty(shape=(N,N),dtype=complex)
for i in range(0,N):
    for j in range(0,N):
        M[j,i] = np.exp(-2*np.pi*complex(0,1)*i*j/N)
fftanalytic = M@y
plt.plot(freq,np.abs(fftanalytic)[:int(len(y)/2)],"b--",label="Analytic",linewidth=2)
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Amplitude",fontsize=12)
plt.legend(fontsize=12)
#plt.savefig("problem_5_analytic.png",bbox_inches="tight",dpi=500)
plt.show()

#Plot Residuals of the fit
res = np.abs(fftanalytic)[:int(len(y)/2)]-np.abs(fftsine)
plt.plot(freq,res*(1e11),color="darkblue")
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel(r"Residauls ($10^{-11}$)",fontsize=12)
#plt.savefig("problem_5_analyticres.png",bbox_inches="tight",dpi=500)
plt.show()

#print("Mean Residual=",np.mean(res))
#print("Residual STD=",np.std(res,ddof=1))

#Windowing: choose the window function to be 0.5-0.5cos(2πx/N) (i.e. the Hann window)
xx = np.linspace(0,N-1,N)/N
ywind = 0.5*(1-np.cos(2*np.pi*xx))
fftnew = fft(y*ywind)[:int(len(y)/2)]
plt.plot(freq,np.abs(fftnew),color="darkblue",linewidth=2,label="With Hann Window")
plt.plot(freq,np.abs(fftsine),"r-",linewidth=2,label="Without Hann Window")
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Amplitude",fontsize=12)
plt.legend(fontsize=12)
#plt.savefig("problem_5_Hann.png",bbox_inches="tight",dpi=500)
plt.show()

#Plotting a log plot to compare the sine wave without window to the function with a window
plt.semilogy(freq,np.abs(fftnew),color="darkblue",label="With Hann Window")
plt.semilogy(freq,np.abs(fftsine),"r-",label="Without Hann Window")
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Amplitude",fontsize=12)
plt.legend(fontsize=12)
#plt.savefig("problem_5_Hannlog.png",bbox_inches="tight",dpi=500)
plt.show()

#Fourier transform of the window
ywindfft = fft(ywind)
print("Showing the Fourier Transform of the Window =", np.real(ywindfft)/N)

"""
#The analytic solution we produced if you want to check, however the results are not great
def window(k):
    first = (1-np.exp(-2*np.pi*complex(0,1)*k))/(1-np.exp(-2*np.pi*complex(0,1)*k/N))
    second = (1-np.exp(-2*np.pi*complex(0,1)*(k+1)))/(1-np.exp(-2*np.pi*complex(0,1)*(k+1)/N))
    third = (1-np.exp(-2*np.pi*complex(0,1)*(k-1)))/(1-np.exp(-2*np.pi*complex(0,1)*(k-1)/N))
    return 0.5*(first-0.5*(second+third))

print("Analytic=", (window(np.linspace(0,N-1,N)))/N)"""

#Recreating the window function with some code ;)
fftsine = fft(y) #so we get the entire transform as we had cut out half of it before
fftwindowm = []
for i in range(0,len(fftsine)):
    if i==0: #remember that it is symmetric, so we can find neighbor terms even for the boundary terms
        fftwindowm.append(0.5*fftsine[i]-0.25*(fftsine[-1]+fftsine[i+1]))
        continue
    if i==len(fftsine)-1: #same as above
        fftwindowm.append(0.5*fftsine[i]-0.25*(fftsine[0]+fftsine[i-1]))
        continue
    newval = 0.5*fftsine[i]-0.25*(fftsine[i-1]+fftsine[i+1])
    fftwindowm.append(newval)
fftwindowm = fftwindowm[:int(len(y)/2)]
plt.plot(freq,np.abs(fftnew),color="darkblue",linewidth=2,label="Normal Method")
plt.plot(freq,np.abs(fftwindowm),"r--",linewidth=2,label="Using Neighbors")
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Amplitude",fontsize=12)
plt.legend(fontsize=12)
plt.show()

print("Discrepancy =", np.mean(np.abs(fftwindowm)-np.abs(fftnew)))