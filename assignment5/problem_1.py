import numpy as np
import os
import matplotlib.pyplot as plt
import json
from simple_read_ligo import read_template, read_file
from numpy.fft import fft, ifft, rfft, irfft

os.chdir('/Users/jehandastoor/Phys512/assignment5/ligodata') #changing the directory to the one with the data

#--------------------------------------------
#First we import all the data directly from the json as suggested in LOSC_Event_tutorial.py, taking inspiration from simple_read_ligo.py
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

#As dt is the same for both Livingston and Hanford, we use time_H for simplicity
time = time_H
fs = 1/time
