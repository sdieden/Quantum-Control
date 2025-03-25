import numpy as np
import matplotlib.pyplot as plt
import sys, os
import cmath as cm
import pathlib
newModPath=pathlib.Path(os.path.dirname(os.path.abspath(__file__)),'NewModule')
sys.path.insert(0, str(newModPath))
from arc_sam import *  # Import ARC (Alkali Rydberg Calculator)
from PulseFunction import U, apply_pulse, pulse_evolution, pulse_evolution_final, total_population, conv_en, conv_t, conv_ce, hbar
from matplotlib.colors import LinearSegmentedColormap


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
N = 6
F_pos = np.linspace(Emin, Emax, N)
F_neg = np.linspace(-Emax, -Emin, N)
a = np.concatenate((F_pos, F_neg))
a = np.sort(a)  # Trier les valeurs pour assurer l'ordre croissant

# Liste des différentes durées de pulse à tester
dt_values = np.logspace(-9, -6, 2*N) # Distribution logarithmique entre 10^-9 et 10^-6, avec plus de points

# Dictionnaires pour stocker les résultats pour chaque valeur de dt
all_l10_populations = {}
all_l_sup_10_populations = {}

for dt in dt_values:
    #print(f"\n=== Calculs avec dt = {dt} secondes ===")
    
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
            for idx in range(start_idx, end_idx):
                #print(f"idx: {idx}, basis element: {calc.basisStates[idx]}")
                pop_l_sup_10 += y[idx]
        
        l_sup_10_populations.append(pop_l_sup_10)
        """
        # Affiche les résultats pour quelques pulses (pas tous pour ne pas surcharger l'affichage)
        if i % 5 == 0:
            print(f"Pulse {i+1} (amplitude {pulse_list[0][0]} V/m): Population l=10: {pop_l_10:.6e}, Population l>10: {pop_l_sup_10:.6e}")
        """
    # Stocker les résultats pour cette valeur de dt
    all_l10_populations[dt] = l10_populations
    all_l_sup_10_populations[dt] = l_sup_10_populations

# Sauvegarde des résultats dans un fichier NumPy pour une utilisation ultérieure
np.savez('resultats_multi_dt.npz', 
         dt_values=dt_values, 
         amplitudes=a, 
         l10_pop=all_l10_populations, 
         l_sup_10_pop=all_l_sup_10_populations)
print("Résultats sauvegardés dans 'resultats_multi_dt.npz'")
"""
# Création des graphiques pour comparer les résultats avec différents dt
# Figure pour l=10 (échelle linéaire)
plt.figure(figsize=(12, 6))
for dt in dt_values:
    plt.plot(a, all_l10_populations[dt], 'o-', label=f'dt = {dt}s')
plt.xlabel('Amplitude du champ électrique (V/m)')
plt.ylabel('Population')
plt.title('Population l=10 en fonction de l\'amplitude du champ électrique pour différents dt')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('l10_lineaire.png')
plt.show()

# Figure pour l>10 (échelle linéaire)
plt.figure(figsize=(12, 6))
for dt in dt_values:
    plt.plot(a, all_l_sup_10_populations[dt], 's-', label=f'dt = {dt}s')
plt.xlabel('Amplitude du champ électrique (V/m)')
plt.ylabel('Population')
plt.title('Population l>10 en fonction de l\'amplitude du champ électrique pour différents dt')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('l_sup_10_lineaire.png')
plt.show()
"""
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
ax1.grid(True, alpha=0.3, linestyle='--')

# 2e sous-graphique : Champs positifs avec pcolormesh
pcm_pos = ax2.pcolormesh(X_pos, Y_pos, Z_pos, cmap=custom_cmap, shading='auto')
ax2.set_xlabel('Amplitude du champ électrique (V/m)')
ax2.set_title('Population l>10 - Champs électriques positifs')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, linestyle='--')

# Barre de couleur commune (à droite)
cbar = fig.colorbar(pcm_pos, ax=[ax1, ax2], label='Population des états l>10')

#plt.tight_layout()
plt.savefig('heatmap_l_sup_10_split_custom.png')
plt.show()


"""
# Barre de couleur commune (à droite)
cbar = fig.colorbar(pcm_pos, ax=[ax1, ax2], label='Population des états l>10')

plt.tight_layout()
plt.savefig('heatmap_l_sup_10_split_custom.png')
plt.show()

# Création du heatmap pour l>10 avec une meilleure présentation
fig, ax = plt.subplots(figsize=(12, 8))
pcm = ax.pcolormesh(X, Y, Z_l_sup_10, cmap=custom_cmap, shading='auto')
c = plt.colorbar(pcm, ax=ax, extend='both', label='Population des états l>10')

plt.xlabel('Amplitude du champ électrique (V/m)')
plt.ylabel('Durée du pulse (s)')
plt.yscale('log')  # Échelle logarithmique pour mieux visualiser les différentes durées
plt.title('Population des états l>10 en fonction du champ électrique et de la durée')
plt.grid(True, alpha=0.3, linestyle='--')  # Grille discrète
plt.tight_layout()
plt.savefig('heatmap_l_sup_10_custom.png')
plt.show()

# Version avec contours pour une meilleure visualisation
plt.figure(figsize=(12, 8))
levels = 20  # Nombre de niveaux de contour
contour = plt.contourf(X, Y, Z_l_sup_10, levels, cmap='PuBu_r')
plt.colorbar(contour, label='Population des états l>10')
plt.xlabel('Amplitude du champ électrique (V/m)')
plt.ylabel('Durée du pulse (s)')
plt.yscale('log')
plt.title('Population des états l>10 (visualisation avec contours)')
plt.grid(True, alpha=0.3, linestyle='--')

# Ajouter une annotation pour indiquer que la zone centrale n'est pas calculée
plt.axvspan(-Emin, Emin, alpha=0.1, color='gray')
plt.text(0, np.min(dt_values)*3, 'Zone non calculée', 
         ha='center', va='bottom', 
         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

plt.tight_layout()
plt.savefig('contour_l_sup_10.png')
plt.show()
"""



"""
# Graphique comparatif supplémentaire - Ratio des populations pour différents dt
plt.figure(figsize=(12, 6))
for dt in dt_values:
    ratio = []
    for i in range(len(a)):
        if all_l10_populations[dt][i] > 0:  # Éviter division par zéro
            ratio.append(all_l_sup_10_populations[dt][i] / all_l10_populations[dt][i])
        else:
            ratio.append(0)
    plt.plot(a, ratio, 'o-', label=f'dt = {dt}s')
plt.xlabel('Amplitude du champ électrique (V/m)')
plt.ylabel('Ratio (l>10 / l=10)')
plt.title('Ratio des populations l>10 / l=10 pour différents dt')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('ratio_populations.png')
plt.show() 
"""