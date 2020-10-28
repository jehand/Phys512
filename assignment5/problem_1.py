import numpy as np
import os
import matplotlib.pyplot as plt
import json
from simple_read_ligo import read_template, read_file
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import tukey
from scipy.ndimage import gaussian_filter

os.chdir('/Users/jehandastoor/Phys512/assignment5/ligodata') #changing the directory to the one with the data

#--------------------------------------------
# First we import all the data directly from the json as suggested in LOSC_Event_tutorial.py, taking inspiration from simple_read_ligo.py
#--------------------------------------------

fnjson = "BBH_events_v3.json"
eventname = 'GW150914' 
#eventname = 'GW151226' 
#eventname = 'LVT151012'
#eventname = 'GW170104'

event = json.load(open(fnjson,"r"))[eventname]
fn_H = event['fn_H1']               # File name for H data
fn_L = event['fn_L1']               # File name for L data
fn_template = event['fn_template']  # File name for template waveform
fs = event['fs']                    # Set sampling rate
tevent = event['tevent']            # Set approximate event GPS time
fband = event['fband']              # frequency band for bandpassing signal

strain_H, time_H, utc_H = read_file(fn_H)
strain_L, time_L, utc_L = read_file(fn_L)
template_H, template_L = read_template(fn_template)

#As dt is the same for both Hanford and Livingston, we use time_H for simplicity
time = time_H
fs = 1/time
n = len(strain_H) #number of points are the same for H and L
window = tukey(n,alpha=0.5) #defining our Tukey window with 0.5 the window inside tapered region

#--------------------------------------------
# We now determine our noise model for our data
#--------------------------------------------

#First, calculate the power spectrums and frequencies multiplied by the window
power_H = np.abs(rfft(strain_H*window))**2
power_L = np.abs(rfft(strain_L*window))**2
power_tempH = np.abs(rfft(template_H*window))**2
power_tempL = np.abs(rfft(template_L*window))**2
freq = rfftfreq(n,d=time)

#Next, we arbitrarily only use the region between [70Hz,1690Hz]
minindex = (np.abs(freq-70)).argmin() #lower cutoff = 70Hz
maxindex = (np.abs(freq-1690)).argmin() #upper cutoff = 1690Hz
power_H = power_H[minindex:maxindex]
power_L = power_L[minindex:maxindex]
power_tempH = power_tempH[minindex:maxindex]
power_tempL = power_tempL[minindex:maxindex]
freq = freq[minindex:maxindex]

#We then convolve our answer with a Gaussian filter
power_smoothH = gaussian_filter(power_H,sigma=75)
power_smoothL = gaussian_filter(power_L,sigma=75)

#Plotting the Hanford data first
plt.loglog(freq,power_tempH,label="Template")
plt.loglog(freq,power_H,label="Windowed Data")
plt.loglog(freq,power_smoothH,label="Smoothed Data")
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Power Spectrurm",fontsize=12)
plt.legend(fontsize=12)
plt.show()

#Now plotting the Livingston data
plt.loglog(freq,power_tempL,label="Template")
plt.loglog(freq,power_L,label="Windowed Data")
plt.loglog(freq,power_smoothL,label="Smoothed Data")
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Power Spectrurm",fontsize=12)
plt.legend(fontsize=12)
plt.show()