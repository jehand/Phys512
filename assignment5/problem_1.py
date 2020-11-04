import numpy as np
import os
import matplotlib.pyplot as plt
import json
from simple_read_ligo import read_template, read_file
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import tukey
from scipy.ndimage import gaussian_filter
from correlation import corr
from scipy.constants import c 

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
tevent = event['tevent']            # Set approximate event GPS time
fband = event['fband']              # frequency band for bandpassing signal

strain_H, time_H, utc_H = read_file(fn_H)
strain_L, time_L, utc_L = read_file(fn_L)
template_H, template_L = read_template(fn_template)

#As dt is the same for both Hanford and Livingston, we use time_H for simplicity
time = time_H
n = len(strain_H) #number of points are the same for H and L
window = tukey(n,alpha=0.5) #defining our Tukey window with 0.5 the window inside tapered region

#--------------------------------------------
# We now determine the noise model for our data
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
power_Hcut = power_H[minindex:maxindex]
power_Lcut = power_L[minindex:maxindex]
power_tempHcut = power_tempH[minindex:maxindex]
power_tempLcut = power_tempL[minindex:maxindex]
freqcut = freq[minindex:maxindex]

#We then convolve our answer with a Gaussian filter with sigma arbitrarily chosen as 75 to give best results
sig = 75
power_smoothH = gaussian_filter(power_Hcut,sigma=sig)
power_smoothL = gaussian_filter(power_Lcut,sigma=sig)

#Plotting the Hanford data first
plt.loglog(freqcut,power_tempHcut,"r-",label="Template",zorder=5)
plt.loglog(freqcut,power_Hcut,color="dodgerblue",label="Windowed Data",zorder=0)
plt.loglog(freqcut,power_smoothH,color="darkblue",label="Smoothed Data",zorder=10)
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Power Spectrurm",fontsize=12)
plt.legend(fontsize=10)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/whitened_H.png",bbox_inches="tight",dpi=500)
plt.show()

#Now plotting the Livingston data
plt.loglog(freqcut,power_tempLcut,color="darkslategrey",label="Template",zorder=5)
plt.loglog(freqcut,power_Lcut,color="orchid",label="Windowed Data",zorder=0)
plt.loglog(freqcut,power_smoothL,color="purple",label="Smoothed Data",zorder=10)
plt.xlabel("Frequency (Hz)",fontsize=12)
plt.ylabel("Power Spectrurm",fontsize=12)
plt.legend(fontsize=10)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/whitened_L.png",bbox_inches="tight",dpi=500)
plt.show()


#Finally, our noise matrix (1D array in this case) is just our smoothed power spectrums
#However, since we have excluded a subset of our data, we have to include these in our noise matrix again. As we normally use N^-1, we make these   
#values = infinity so that we are not dividing by zero and they lead to no contribution. 
noise = np.ones(int(n/2)+1)*np.infty
N_H = noise.copy()
N_L = noise.copy()
N_H[minindex:maxindex] = power_smoothH.copy()
N_L[minindex:maxindex] = power_smoothL.copy()

#--------------------------------------------
# Using this noise matrix, we now use the method of matched filters to calculate our whitened data set 
# i.e. calculating N^-1/2 d and N^-1/2 A where d is our data and A is our template.
#--------------------------------------------

#Calculating the whitened data sets
whitened_H = irfft(np.power(N_H,-1/2)*rfft(strain_H*window),n)
whitened_H_temp = irfft(np.power(N_H,-1/2)*rfft(template_H*window),n)
whitened_L = irfft(np.power(N_L,-1/2)*rfft(strain_L*window),n)
whitened_L_temp = irfft(np.power(N_L,-1/2)*rfft(template_L*window),n)

#Matched filtering by calculating the correlation of N^(-1/2)A with N^(-1/2)d. Do this using "correlation.py" (Assignment 4).
matchfilterH = np.abs(corr(whitened_H_temp, whitened_H,n))
matchfilterL = np.abs(corr(whitened_L_temp, whitened_L,n))

#We can now plot our matched filtered results as a check
xtimes = np.arange(0,n)*time
plt.plot(xtimes,matchfilterH,color="darkblue",label="Hanford Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("m",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/matchfilter_H.png",bbox_inches="tight",dpi=500)
plt.show()

plt.plot(xtimes,matchfilterL,color="purple",label="Livingston Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("m",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/matchfilter_L.png",bbox_inches="tight",dpi=500)
plt.show()

#--------------------------------------------
# We now proceed to calculate the SNR's for each detector and the combined system
#--------------------------------------------

#Determining sigma_m as the square root of the mean of our covariance matrix (A^T N^-1 A)^-1 the same as we had for our least squares regression
#We can use our N^-1/2 A basis and just square it to get N^-1 A^2 which is the same as before since we are dealing with 1D arrays. 
sigma_m_H = np.sqrt(np.mean(whitened_H_temp**2))
SNR_H = matchfilterH/sigma_m_H
plt.plot(xtimes,SNR_H,color="darkblue",label="Hanford Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("SNR",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/SNRmodel_H.png",bbox_inches="tight",dpi=500)
plt.show()
print("SNR Hanford Sigma_m_H =", sigma_m_H)

sigma_m_L = np.sqrt(np.mean(whitened_L_temp**2))
SNR_L = matchfilterL/sigma_m_L
plt.plot(xtimes,SNR_L,color="purple",label="Livingston Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("SNR",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/SNRmodel_L.png",bbox_inches="tight",dpi=500)
plt.show()
print("SNR Livingston Sigma_m_L =", sigma_m_L)

sigma_m = np.sqrt(np.mean((whitened_L_temp+whitened_H_temp)**2))
SNR = (matchfilterL+matchfilterH)/sigma_m
plt.plot(xtimes,SNR,color="crimson",label="Combined Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("SNR",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/SNRmodel_combined.png",bbox_inches="tight",dpi=500)
plt.show()
print("SNR Combined Sigma_m =", sigma_m)

#--------------------------------------------
# Comparison of the SNR from the scatter of the matched filter to the analytic SNR from our model
#--------------------------------------------

#We calculate the sample standard deviation of our matched filter
std_H = np.std(matchfilterH,ddof=1) #ddof=1 as it is the sample std not population since we have a discrete data set
SNR_H_scatter = matchfilterH/std_H
plt.plot(xtimes,SNR_H_scatter,color="darkblue",label="Hanford Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("SNR",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/SNRscatter_H.png",bbox_inches="tight",dpi=500)
plt.show()
print("Scatter SNR Hanford std_H =", std_H, "Discrepancy_H =", std_H-sigma_m_H, "% Discrep =", (std_H-sigma_m_H)/sigma_m_H * 100)

std_L = np.std(matchfilterL,ddof=1)
SNR_L_scatter = matchfilterL/std_L
plt.plot(xtimes,SNR_L_scatter,color="purple",label="Livingston Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("SNR",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/SNRscatter_L.png",bbox_inches="tight",dpi=500)
plt.show()
print("Scatter SNR Livingston std_L =", std_L, "Discrepancy_L =", std_L-sigma_m_L, "% Discrep =", (std_L-sigma_m_L)/sigma_m_L * 100)

std = np.std(matchfilterL+matchfilterH,ddof=1)
SNR_scatter = (matchfilterL+matchfilterH)/std_L
plt.plot(xtimes,SNR_scatter,color="crimson",label="Combined Matched Filter")
plt.xlabel("Time (s)",fontsize=12)
plt.ylabel("SNR",fontsize=12)
plt.savefig("/Users/jehandastoor/Phys512/assignment5/plots/SNRscatter_combined.png",bbox_inches="tight",dpi=500)
plt.show()
print("Scatter SNR Combined std_L =", std, "Discrepancy_L =", std-sigma_m, "% Discrep =", (std-sigma_m)/sigma_m * 100)

#--------------------------------------------
# Finding the mid-point frequency
#--------------------------------------------

#Calculate a cumulative sum of the PS of whitened template ("template and noise model") and find where it is equal to half the power
whitened_H_PS = np.abs(rfft(whitened_H_temp))**2
whitened_L_PS = np.abs(rfft(whitened_L_temp))**2

cumsum_H = np.cumsum(whitened_H_PS)
cumsum_L = np.cumsum(whitened_L_PS)

midfreq_H = freq[np.argmin(abs(cumsum_H-np.sum(whitened_H_PS)/2))]
print("Mid frequency Hanford =", midfreq_H)
midfreq_L = freq[np.argmin(abs(cumsum_L-np.sum(whitened_L_PS)/2))]
print("Mid frequency Livingston =", midfreq_L)

#--------------------------------------------
# Finding the time of arrival of the GW
#--------------------------------------------

#We find the max SNR for each event per detector and compare the times at which these occur
arrival_H = xtimes[np.argmax(np.abs(SNR_H))]
arrival_L = xtimes[np.argmax(np.abs(SNR_L))]
print("Arrival Time H =", arrival_H, "Arrival Time L =", arrival_L)

#Positional uncertainty can now be theoretically calculated by assuming the GW travels at the speed of light 
#and using the measured distance between both detectors.
 
dist = 3002*1000 #distance in metres
posituncert = dist/c
print("Theoretical Uncert =", posituncert, "Calculated Uncert =", np.abs(arrival_H-arrival_L))