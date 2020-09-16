import numpy as np

data = np.loadtxt("lakeshore.txt") 

"""
Now that we have the data imported, we want to interpolate to find a temperature from a voltage.
We are given dV/dT (in mV/K) so we can simply linearly interpolate each time, assuming that the the
function is linear between respective voltages, being careful to convert mV to V. 
We make the assumption that since we are interpolating, there cannot be voltages smaller or larger than what
we have been given.
"""


