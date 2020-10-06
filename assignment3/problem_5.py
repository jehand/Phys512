import numpy as np
import matplotlib.pyplot as plt
from wmap_camb_example import get_spectrum
from markov import mcmc

"""
We begin by recognising that we are adding a new value of tau as a prior in our matrix. Hence, we must adapt our 'markov.py' file (which we already
have). 
"""

tau = 0.0544
utau = 0.0073