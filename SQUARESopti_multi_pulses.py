import numpy as np
import matplotlib.pyplot as plt
import sys, os
import cmath as cm
import pathlib
import time
import datetime
newModPath=pathlib.Path(os.path.dirname(os.path.abspath(__file__)),'NewModule')
sys.path.insert(0, str(newModPath))
from arc import *  # Import ARC (Alkali Rydberg Calculator)
from PulseFunction import U, apply_pulse, pulse_evolution, pulse_evolution_final, total_population,seconds_to_au , ghz_to_hartree, conv_ce, hbar
from matplotlib.colors import LinearSegmentedColormap

debut = time.time()

# Initialization of the system
atom = Calcium40()
calc = StarkMap(atom)

n = 35
l = 3
j = 3
mj = 0
s = 0
nmin = n - 1
nmax = n + 2
lmax = nmax - 1
calc.defineBasis(n, l, j, mj, nmin, nmax, lmax, progressOutput=True, s=s)