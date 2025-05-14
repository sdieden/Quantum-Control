import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pathlib
import time
import datetime

newModPath = pathlib.Path(os.path.dirname(os.path.abspath(__file__)), 'NewModule')
sys.path.insert(0, str(newModPath))
from arc import *  # Import ARC (Alkali Rydberg Calculator)
from PulseFunction import U, apply_pulse, pulse_evolution, pulse_evolution_final, total_population, seconds_to_au, ghz_to_hartree, conv_ce, hbar, optimization_pulse
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
initial_wf = np.zeros(len(calc.basisStates), dtype=complex)
initial_wf[calc.indexOfCoupledState] = 1

# Définition des paramètres pour les champs électriques
Emin = 0.0#8e2  # 800 V/m
Emax = 15e2#15e2  # 1500 V/m

N = 2
min_t_interval = 2e-10
min_v_interval = 0.0029296875 * 100  # [V/m]
step = round((Emax - Emin) / N)
F_pos = np.linspace(Emin, Emax, num=2)
F_neg = np.linspace(-Emax, -Emin, num=2)

a = np.concatenate((F_pos, F_neg))
F_values = np.sort(a)  # Trier les valeurs pour assurer l'ordre croissant

# dt_values = np.logspace(-7.221, -7.15, num = N)# Distribution logarithmique entre 10^-9 et 10^-6, avec plus de points
dt_values = np.logspace(-10,-6, num = N)
#dt_values = np.linspace(10e-9, 10e-8, num=N)
# Dictionnaires pour stocker les résultats pour chaque valeur de dt

optimization_pulse(dt_values,F_values,initial_wf=initial_wf)
print("Starting detailed l-level population analysis...")


fin_calcul = time.time()
# Sauvegarde des résultats
date = datetime.datetime.now()
name = f'Plot/resultats_multi_dt_lin{date}.npz'
save_dict = {
    'dt_values': dt_values,
    'amplitudes': a
}
for l_level in range(0, 35):
    l_key = f'l{l_level}_pop'
    l_data = {}
    for dt in dt_values:
        l_data[dt] = all_l_populations[dt][l_level]
    save_dict[l_key] = l_data
np.savez(name, **save_dict)
print(f"Résultats sauvegardés dans{name}")



