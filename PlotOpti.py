import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Charger les données
data = np.load("Plot/resultats_multi_dt_lin2025-05-12 20:50:56.516904.npz", allow_pickle=True)
print(data)
dt_values = data['dt_values']
amplitudes = data['amplitudes']
#l3_pop = data['l3_pop'].item()
l_sup_10_pop = data['l_sup_10_pop'].item()

# Création des matrices pour la visualisation
X, Y = np.meshgrid(amplitudes, dt_values)
Z_l_sup_10 = np.zeros((len(dt_values), len(amplitudes)))
#Z_l_3 = np.zeros((len(dt_values), len(amplitudes)))
# Remplir les matrices
for i, dt in enumerate(dt_values):
    for j, amp in enumerate(amplitudes):
        Z_l_sup_10[i, j] = l_sup_10_pop[dt][j]
        #Z_l_3[i, j] = l3_pop[dt][j]

# Séparer les données négatives et positives
neg_indices = np.where(amplitudes < 0)[0]
pos_indices = np.where(amplitudes >= 0)[0]

X_neg, X_pos = X[:, neg_indices], X[:, pos_indices]
Y_neg, Y_pos = Y[:, neg_indices], Y[:, pos_indices]
Z_neg, Z_pos = Z_l_sup_10[:, neg_indices], Z_l_sup_10[:, pos_indices]
#Z_neg_single, Z_pos_single = Z_l_3[:, neg_indices], Z_l_3[:, pos_indices]

colors = [(1, 1, 1),       # blanc pour très faibles valeurs
          (0.8, 0.9, 1),   # bleu très clair
          (0.3, 0.5, 0.9), # bleu moyen
          (0, 0.2, 0.8),   # bleu foncé pour valeurs intermédiaires (~0.6)
          (0, 0.7, 0.2),   # vert pour valeurs élevées
          (1, 0.5, 0)]     # orange pour valeurs maximales

# Positions des couleurs sur l'échelle de 0 à 1
positions = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
#epsilon = 1e-10
#log_positions = np.log(np.array(positions) + epsilon)
#log_positions = (log_positions - log_positions.min()) / (log_positions.max() - log_positions.min())
#custom_cmap = LinearSegmentedColormap.from_list('custom_blues_greens_orange',
                                              #list(zip(log_positions, colors)),
                                               #N=512)
custom_cmap = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)), N=512)

# Création d'une figure avec deux sous-graphiques côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Graphiques pour les champs négatifs et positifs
pcm_neg = ax1.pcolormesh(X_neg, Y_neg, Z_neg,
                          cmap=custom_cmap,
                          shading='auto',
                          #vmin=0.0,
                          #vmax=1.0)
                         )
pcm_pos = ax2.pcolormesh(X_pos, Y_pos, Z_pos,
                          cmap=custom_cmap,
                          shading='auto',
                          #vmin=0.0,
                          #vmax=1.0)
                         )
# Configuration des axes et labels
for ax, title in [(ax1, 'Champs électriques négatifs'), (ax2, 'Champs électriques positifs')]:
    ax.set_xlabel('Amplitude du champ électrique (V/m)')
    ax.set_title(f'Population l>10 - {title}')
    #ax.grid(False, alpha=0.3, linestyle='--')
    ax.set_yscale('log')  # Ajouter l'échelle logarithmique

ax1.set_ylabel('Durée du pulse (s)')

# Ajout de la barre de couleur entre les deux graphiques
#plt.subplots_adjust(wspace=0.2)  # Augmenter l'espace entre les graphiques
cbar = fig.colorbar(pcm_pos, ax=[ax1, ax2], label='Population des états l>10')


# Ajustement de la mise en page
#plt.savefig('analyse_populations.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyse statistique
print("\nAnalyse statistique des populations:")
print(f"Population l>10 moyenne: {np.mean(Z_l_sup_10):.3f}")
print(f"Population l>10 maximale: {np.max(Z_l_sup_10):.3f}")

# Trouver les parametres optimaux
max_pop_idx = np.unravel_index(np.argmax(Z_l_sup_10), Z_l_sup_10.shape)

print("\nParamètres optimaux pour la population l>10 maximale:")
print(f"  - Durée: {dt_values[max_pop_idx[0]]:.2e} s")
print(f"  - Champ électrique: {amplitudes[max_pop_idx[1]]:.2f} V/m")
print(f"  - Population: {Z_l_sup_10[max_pop_idx]:.3f}")

