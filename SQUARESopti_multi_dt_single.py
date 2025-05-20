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

n = 79
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

N = 500
min_t_interval = 2e-10
min_v_interval = 0.0029296875 * 100  # [V/m]
step = round((Emax - Emin) / N)
F_pos = np.linspace(Emin, Emax, num=200)
F_neg = np.linspace(-Emax, -Emin, num=10)

a = np.concatenate((F_pos, F_neg))
a = np.sort(a)  # Trier les valeurs pour assurer l'ordre croissant

# dt_values = np.logspace(-7.221, -7.15, num = N)# Distribution logarithmique entre 10^-9 et 10^-6, avec plus de points
dt_values = np.linspace(1e-10,1e-9, num = N)
#dt_values = np.linspace(10e-9, 10e-8, num=N)
# Dictionnaires pour stocker les résultats pour chaque valeur de dt
all_l_populations = {}
all_l_sup_10_populations = {}
# initialize dictionnaries for every l = 10 - 35
for dt in dt_values:
    all_l_populations[dt] = {}
    for l_level in range(0, lmax+1):  # l de 10 à 34 inclus
        all_l_populations[dt][l_level] = []

print("Starting detailed l-level population analysis...")

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

        # Ne garde que l'etat final (après le pulse)
        output_coef.append(x[-1])

        # Calcule la population pour l'etat final seulement
        y = np.abs(x[-1]) ** 2
        output_pop.append(y)

        # Calculer la population pour chaque l de 10 à 34
        for l_level in range(0, lmax+1):
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
fin_calcul = time.time()
# Sauvegarde des résultats dans un fichier NumPy pour une utilisation ultérieure
date = datetime.datetime.now()
name = f'Plot/resultats_multi_dt_lin{date}.npz'
save_dict = {
    'dt_values': dt_values,
    'amplitudes': a
}
for l_level in range(0, lmax+1):
    l_key = f'l{l_level}_pop'
    l_data = {}
    for dt in dt_values:
        l_data[dt] = all_l_populations[dt][l_level]
    save_dict[l_key] = l_data
np.savez(name, **save_dict)
print(f"Résultats sauvegardés dans{name}")

print("plotting...")

#selected_l_levels = [3, 10, 15, 20,25, 34]  # Niveaux l sélectionnés pour visualisation
selected_l_levels = list(range(0, lmax+1))
# Créer une figure avec plusieurs sous-graphiques (un par niveau l sélectionné)
fig, axes = plt.subplots(len(selected_l_levels), 2, figsize=(16, 4 * len(selected_l_levels)))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
fig.subplots_adjust(wspace=0.4)
# Création d'une palette de couleurs personnalisée
colors = [(1, 1, 1),  # blanc pour très faibles valeurs
          (0.8, 0.9, 1),  # bleu très clair
          (0.3, 0.5, 0.9),  # bleu moyen
          (0, 0.2, 0.8),  # bleu foncé pour valeurs intermédiaires (~0.6)
          (0, 0.7, 0.2),  # vert pour valeurs élevées
          (1, 0.5, 0)]  # orange pour valeurs maximales

positions = [0, 0.1, 0.3, 0.6, 0.8, 1.0]
custom_cmap = LinearSegmentedColormap.from_list('custom_blues_greens_orange', list(zip(positions, colors)), N=512)

# Séparer les données en deux parties: champs négatifs et positifs
neg_indices = np.where(a < 0)[0]
pos_indices = np.where(a >= 0)[0]

# Créer la grille pour les graphiques
X, Y = np.meshgrid(a, dt_values)
X_neg = X[:, neg_indices]
Y_neg = Y[:, neg_indices]
X_pos = X[:, pos_indices]
Y_pos = Y[:, pos_indices]

print("Creating visualizations for selected l levels...")

for i, l_level in enumerate(selected_l_levels):
    # Créer les matrices Z pour ce niveau l
    Z_l = np.zeros((len(dt_values), len(a)))
    for j, dt in enumerate(dt_values):
        for k, amp in enumerate(a):
            Z_l[j, k] = all_l_populations[dt][l_level][k]

    Z_neg = Z_l[:, neg_indices]
    Z_pos = Z_l[:, pos_indices]

    # Créer les graphiques
    ax_neg = axes[i, 0]
    ax_pos = axes[i, 1]

    pcm_neg = ax_neg.pcolormesh(X_neg, Y_neg, Z_neg, cmap=custom_cmap, shading='auto')
    ax_neg.set_title(f'Population l={l_level} - Champs négatifs')
    ax_neg.set_xlabel('Amplitude du champ électrique (V/m)')
    if i == 0:
        ax_neg.set_ylabel('Durée du pulse (s)')

    pcm_pos = ax_pos.pcolormesh(X_pos, Y_pos, Z_pos, cmap=custom_cmap, shading='auto')
    ax_pos.set_title(f'Population l={l_level} - Champs positifs')
    ax_pos.set_xlabel('Amplitude du champ électrique (V/m)')

    # Ajouter une barre de couleur pour chaque niveau l
    plt.colorbar(pcm_pos, ax=[ax_neg, ax_pos], label=f'Population l={l_level}')

# plt.tight_layout()
plot_filename = f'Plot/heatmap_selected_l_levels_{date.strftime("%Y%m%d_%H%M%S")}.png'
plt.savefig(plot_filename)
print(f"Visualisation sauvegardée dans {plot_filename}")

fin_plot = time.time()
plt.show()
temps_calcul = fin_calcul - debut
temps_plot = fin_plot - fin_calcul
temps_total = fin_plot - debut
print(f"Temps de calcul: {temps_calcul:.2f} secondes")
print(f"Temps de génération des graphiques: {temps_plot:.2f} secondes")
print(f"Temps total: {temps_total:.2f} secondes")