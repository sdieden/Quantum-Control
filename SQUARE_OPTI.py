import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pathlib
import time
import datetime

newModPath = pathlib.Path(os.path.dirname(os.path.abspath(__file__)), 'NewModule')
sys.path.insert(0, str(newModPath))
from arc import *  # Import ARC (Alkali Rydberg Calculator)
from PulseFunction import U, apply_pulse, pulse_evolution, pulse_evolution_final, total_population, seconds_to_au, ghz_to_hartree, conv_ce, hbar
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
a = np.sort(a)  # Trier les valeurs pour assurer l'ordre croissant

# dt_values = np.logspace(-7.221, -7.15, num = N)# Distribution logarithmique entre 10^-9 et 10^-6, avec plus de points
dt_values = np.logspace(-10,-6, num = N)
#dt_values = np.linspace(10e-9, 10e-8, num=N)
# Dictionnaires pour stocker les résultats pour chaque valeur de dt
all_l_populations = {}
all_l_sup_10_populations = {}
all_l_coefficients = {}
# initialize dictionnaries for every l = 10 - 35
for dt in dt_values:
    all_l_populations[dt] = {}
    all_l_coefficients[dt] = {}
    for l_level in range(0, 35):  # l de 10 à 34 inclus
        all_l_populations[dt][l_level] = []
    for amplitudes in a:
        all_l_coefficients[dt][amplitudes] = np.zeros(len(calc.basisStates), dtype=complex)

print("Starting detailed l-level population analysis...")
optimized_wavefuction_population = 0

for dt in dt_values:
    print(f"\n=== Calculs avec dt = {dt} secondes ===")

    # Construction des pulses pour cette valeur de dt
    pulse_square = []
    for amplitude in a:
        pulse_square.append([(amplitude, dt)])

    output_coef = []
    output_pop = []

    for i, pulse_list in enumerate(pulse_square):

        # Chaque pulse est appliqué sur l'état initial initial_wf, pas sur l'état résultant du pulse précédent
        x = pulse_evolution(pulse_list, initial_coupled=initial_wf, calc=calc)
        all_l_coefficients[dt][pulse_list[0][0]] = x

        # Ne garde que l'etat final (après le pulse)
        output_coef.append(x[-1])

        # Calcule la population
        y = np.abs(x[-1]) ** 2


        output_pop.append(y)

        # Calculer la population pour chaque l de 10 à 34
        for l_level in range(0, 35):
            # Trouver tous les états avec l = l_level
            l_pop = 0.0
            for idx, state in enumerate(calc.basisStates):
                if state[1] == l_level:  # l est à l'index 1 dans basisStates
                    l_pop += y[idx]

            # Stocker cette population
            all_l_populations[dt][l_level].append(l_pop)

        # Afficher la progression
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pulse_square)} pulses")
    for l in range(10,lmax-1):
        searching_optimized_wf = np.max(all_l_populations[dt][l], axis=0)
        if searching_optimized_wf > optimized_wavefuction_population:
            optimized_wavefuction_population = searching_optimized_wf
            idx = l
            idx_dt = dt
            idx_amplitudes = pulse_list[0][0] # TODO : cannot work, how to find the amplitude related to ?


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
