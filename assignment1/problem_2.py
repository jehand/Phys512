import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("lakeshore.txt") 

"""
Now that we have the data imported, we want to interpolate to find a temperature from a voltage.
We are given dV/dT (in mV/K) so we can simply linearly interpolate each time, assuming that the the
function is linear between respective voltages, being careful to convert mV to V. 
We make the assumption that since we are interpolating, there cannot be voltages smaller or larger than what
we have been given. The arbitrary voltage will be written as a Varb. As the space between voltages is relatively small,
the linear interpolation should be fairly accurate.

To estimate the error we now consider what would happen if we only used half our data points to interpolate between. As a result,
our interpolation is worse, and hence by comparing the two we can approximate a maximum on the error of our interpolation as if
we were able to achieve greater accuracy, we would have double the amount of points available to us to increase the strength of
our current interpolation. This is very rough, but will provide an upper bound on the error of our temperature. We will do this
by removing every second point in each array and then interpolating with the same point, and subtracting the difference of that 
interpolation with our current one. This is obviously just an estimation in the error of an interpolation given the fact that
we can not possibly have access to all the data points and does not account for errors in the actual voltages and temperatures
being measured.
"""
Varb = 1.5

T, Terr, V, Verr, dV, dVerr = [], [], [], [], [], [] #defining arrays to store our data in (all subscripts err are for the error analysis)
for i in range(len(data)):
    T.append(data[i][0])
    V.append(data[i][1])
    dV.append(data[i][2])
    if i%2 == 0: 
        Terr.append(data[i][0])
        Verr.append(data[i][1])
        dVerr.append(data[i][2])
V = np.asarray(V)
Verr = np.asarray(Verr)
ind = (np.abs(V-Varb)).argmin()
inderr = (np.abs(Verr-Varb)).argmin()

"""
Now we know the index and hence the voltage closest to it. From (y-b) = m(x-c) where x is T and y is V, 
-> x = (y-b)/m + c --> Tarb = (Varb - V)/dV/dT + T where dV/dT will be divided by 1000 so it is in volts, and Varb is in volts.
"""
Tarb = ((Varb-V[ind])/(dV[ind]/1000)) + T[ind]
Terr = np.abs(Tarb-(((Varb-Verr[inderr])/(dVerr[inderr]/1000)) + Terr[inderr])) #subtracting the prediction of Terr from Tarb to get an error

print("Tarb=", Tarb, "±", Terr)

"""
We can also now see the error in our fit if we plot the results of our interpolation. 
"""

#First we make a function to give us all the Temperatures
def Ts(Varbs):
    ind = (np.abs(V-Varbs)).argmin()
    Tarb = ((Varbs-V[ind])/(dV[ind]/1000)) + T[ind]
    return Tarb

Vs = np.linspace(0.1,1.64,100) #Creating a list of spaced Voltages
Temps = []
for i in Vs:
    Temps.append(Ts(i))

plt.plot(T,V,"kx",label="Lakeshore 670 Diode")
plt.plot(Temps,Vs,"r-",label="Linear Interpolation")
plt.xlabel("Temperature (K)", fontsize=12)
plt.ylabel("Voltage (V)", fontsize=12)
plt.legend()
plt.savefig("problem_2_plot.png",bbox_inches="tight", dpi=500)
plt.show()