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
initial_wf = np.zeros(len(calc.basisStates),dtype=complex)
initial_wf[calc.indexOfCoupledState]= 1

# Définition des paramètres pour les champs électriques
Emin = 8e2    # 800 V/m
Emax = 15e2   # 1500 V/m

N = 200
min_t_interval = 2e-10
min_v_interval = 0.0029296875 * 100 #[V/m]
step = round((Emax - Emin) / N)
F_pos = np.linspace(Emin, Emax, num = 200)
F_neg = np.linspace(-Emax, -Emin, num = 1)

a = np.concatenate((F_pos, F_neg))
a = np.sort(a)  # Trier les valeurs pour assurer l'ordre croissant

# Liste des différentes durées de pulse à tester
#dt_values = np.logspace(-7.221, -7.15, num = N)# Distribution logarithmique entre 10^-9 et 10^-6, avec plus de points
dt_values = np.linspace(1e-10,9e-9, num = N)
# Dictionnaires pour stocker les résultats pour chaque valeur de dt
all_l10_populations = {}
all_l_sup_10_populations = {}

for dt in dt_values:
    print(f"\n=== Calculs avec dt = {dt} secondes ===")
    
    # Construction des pulses pour cette valeur de dt
    pulse_square = []
    for amplitude in a:
        pulse_square.append([(amplitude, dt)])
    
    output_coef = []
    output_pop = []
    # Listes pour stocker les populations des états d'intérêt pour chaque pulse
    l10_populations = []
    l_sup_10_populations = []
    
    for i, pulse_list in enumerate(pulse_square):
        #print("enumerage number",i)
        # Chaque pulse est appliqué sur l'état initial initial_wf, pas sur l'état résultant du pulse précédent
        x = pulse_evolution(pulse_list, initial_coupled=initial_wf, calc=calc)
        # Ne garde que l'état final (après le pulse)
        output_coef.append(x[-1])
        # Calcule la population pour l'état final seulement
        y = np.abs(x[-1])**2
        output_pop.append(y)
        """
        # Calcule la population pour l=10
        l10_index = calc.indexOfCoupledState+(10-l)
        if l10_index < len(y):
            pop_l_10 = y[l10_index]
        else:
            pop_l_10 = 0
            print(f"Avertissement: l'index {l10_index} pour l=10 est hors limites (taille: {len(y)})")
        l10_populations.append(pop_l_10)
        """
        # Calcule la population pour l>10
        pop_l_sup_10 = 0
        start_idx = calc.indexOfCoupledState+(10-l)
        end_idx = min(len(y), calc.indexOfCoupledState+(n-l))

        if start_idx < len(y):
            #print('in the boucle')
            weighted_pop = 0
            max_l = max(calc.basisStates[idx][1] for idx in range(start_idx, end_idx))
            for idx in range(start_idx, end_idx):
                l_idx = calc.basisStates[idx][1]
                weighted_pop += (l_idx/max_l) * y[idx]  # Population pondérée par l/l_max
            
            pop_l_sup_10 = weighted_pop

        l_sup_10_populations.append(pop_l_sup_10)
    # Stocker les résultats pour cette valeur de dt
    all_l10_populations[dt] = l10_populations
    all_l_sup_10_populations[dt] = l_sup_10_populations
fin_calcul = time.time()
# Sauvegarde des résultats dans un fichier NumPy pour une utilisation ultérieure
date = datetime.datetime.now()
name = f'Plot/resultats_multi_dt_lin{date}.npz'
np.savez(name,
         dt_values=dt_values, 
         amplitudes=a, 
         l10_pop=all_l10_populations, 
         l_sup_10_pop=all_l_sup_10_populations)
print(f"Résultats sauvegardés dans{name}")


print("plotting...")

# Création des matrices pour pcolormesh
X, Y = np.meshgrid(a, dt_values)
Z_l_sup_10 = np.zeros((len(dt_values), len(a)))

# Remplir la matrice avec les populations l>10
for i, dt in enumerate(dt_values):
    for j, amp in enumerate(a):
        Z_l_sup_10[i, j] = all_l_sup_10_populations[dt][j]

# Séparer les données en deux parties: champs négatifs et positifs
neg_indices = np.where(a < 0)[0]
pos_indices = np.where(a >= 0)[0]

X_neg = X[:, neg_indices]
Y_neg = Y[:, neg_indices]
Z_neg = Z_l_sup_10[:, neg_indices]

X_pos = X[:, pos_indices]
Y_pos = Y[:, pos_indices]
Z_pos = Z_l_sup_10[:, pos_indices]

# Création d'une palette de couleurs personnalisée
# Définir les points de couleur et leur position
# blanc (0) -> bleu clair (0.2) -> bleu foncé (0.6) -> vert (0.8) -> orange (1.0)
colors = [(1, 1, 1),       # blanc pour très faibles valeurs
          (0.8, 0.9, 1),   # bleu très clair
          (0.3, 0.5, 0.9), # bleu moyen
          (0, 0.2, 0.8),   # bleu foncé pour valeurs intermédiaires (~0.6)
          (0, 0.7, 0.2),   # vert pour valeurs élevées
          (1, 0.5, 0)]     # orange pour valeurs maximales

# Positions des couleurs sur l'échelle de 0 à 1
positions = [0, 0.1, 0.3, 0.6, 0.8, 1.0]
print("plotting...")
# Créer la carte de couleurs
custom_cmap = LinearSegmentedColormap.from_list('custom_blues_greens_orange', list(zip(positions, colors)),N=512)

# Création d'une figure avec deux sous-graphiques côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# 1er sous-graphique : Champs négatifs avec pcolormesh
pcm_neg = ax1.pcolormesh(X_neg, Y_neg, Z_neg, cmap=custom_cmap, shading='auto')
ax1.set_xlabel('Amplitude du champ électrique (V/m)')
ax1.set_ylabel('Durée du pulse (s)')
ax1.set_yscale('log')
ax1.set_title('Population l>10 - Champs électriques négatifs')
#ax1.grid(True, alpha=0.3, linestyle='--')

# 2e sous-graphique : Champs positifs avec pcolormesh
pcm_pos = ax2.pcolormesh(X_pos, Y_pos, Z_pos, cmap=custom_cmap, shading='auto')
ax2.set_xlabel('Amplitude du champ électrique (V/m)')
#ax2.set_xlabel('Durée du pulse (s)')
ax2.set_title('Population l>10 - Champs électriques positifs')
ax2.set_yscale('log')
#ax2.grid(True, alpha=0.3, linestyle='--')

# Barre de couleur commune (à droite)
cbar = fig.colorbar(pcm_pos, ax=[ax1, ax2], label='Population des états l>10')

#plt.tight_layout()

plt.savefig(f'Plot/heatmap{date}.png')
plt.show()

fin_plot = time.time()

temps_calcul = fin_calcul - debut
temps_plot = fin_plot - debut
print(f"temps de calcul:{temps_calcul:.2f}secondes, temps plot:{temps_plot:.2f}secondes")
#print(debut,temps_calcul,temps_plot) #probleme avec debut ??

