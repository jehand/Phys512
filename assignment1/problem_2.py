import numpy as np

data = np.loadtxt("lakeshore.txt") 

"""
Now that we have the data imported, we want to interpolate to find a temperature from a voltage.
We are given dV/dT (in mV/K) so we can simply linearly interpolate each time, assuming that the the
function is linear between respective voltages, being careful to convert mV to V. 
We make the assumption that since we are interpolating, there cannot be voltages smaller or larger than what
we have been given. The arbitrary voltage will be written as a Varb. As the space between voltages is relatively small,
the linear interpolation should be fairly accurate.
"""
Varb = 1.10932

T, V, dV = [], [], []
for i in range(len(data)):
    T.append(data[i][0])
    V.append(data[i][1])
    dV.append(data[i][2])
V = np.asarray(V)
ind = (np.abs(V-Varb)).argmin()

"""
Now we know the index and hence the voltage closest to it. From (y-b) = m(x-c) where x is T and y is V, 
-> x = (y-b)/m + c --> Tarb = (Varb - V)/dV/dT + T where dV/dT will be divided by 1000 so it is in volts, and Varb is in volts.
"""
Tarb = ((Varb-V[ind])/(dV[ind]/1000)) + T[ind]

"""
To estimate the error we now consider
"""

print("Tarb=", Tarb, "±")