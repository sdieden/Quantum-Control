import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_optimization_results(data_path, data_key, title_prefix="Population", save_path=None):
    """
    Fonction pour visualiser les résultats d'optimisation.
    
    Args:
        data_path (str): Chemin vers le fichier .npz contenant les données
        data_key (str): Clé des données à visualiser dans le fichier .npz
        title_prefix (str): Préfixe pour le titre du graphique
        save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, affiche le graphique.
    """
    # Charger les données
    data = np.load(data_path, allow_pickle=True)
    dt_values = data['dt_values']
    amplitudes = data['amplitudes']
    population_data = data[data_key].item()

    # Création des matrices pour la visualisation
    X, Y = np.meshgrid(amplitudes, dt_values)
    Z = np.zeros((len(dt_values), len(amplitudes)))
    
    # Remplir les matrices
    for i, dt in enumerate(dt_values):
        for j, amp in enumerate(amplitudes):
            Z[i, j] = population_data[dt][j]

    # Séparer les données négatives et positives
    neg_indices = np.where(amplitudes < 0)[0]
    pos_indices = np.where(amplitudes >= 0)[0]

    X_neg, X_pos = X[:, neg_indices], X[:, pos_indices]
    Y_neg, Y_pos = Y[:, neg_indices], Y[:, pos_indices]
    Z_neg, Z_pos = Z[:, neg_indices], Z[:, pos_indices]

    colors = [(1, 1, 1),       # blanc pour très faibles valeurs
              (0.8, 0.9, 1),   # bleu très clair
              (0.3, 0.5, 0.9), # bleu moyen
              (0, 0.2, 0.8),   # bleu foncé pour valeurs intermédiaires (~0.6)
              (0, 0.7, 0.2),   # vert pour valeurs élevées
              (1, 0.5, 0)]     # orange pour valeurs maximales

    positions = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    custom_cmap = LinearSegmentedColormap.from_list('custom', list(zip(positions, colors)), N=512)

    # Création d'une figure avec deux sous-graphiques côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Graphiques pour les champs négatifs et positifs
    pcm_neg = ax1.pcolormesh(X_neg, Y_neg, Z_neg,
                            cmap=custom_cmap,
                            shading='auto')
    pcm_pos = ax2.pcolormesh(X_pos, Y_pos, Z_pos,
                            cmap=custom_cmap,
                            shading='auto')

    # Configuration des axes et labels
    for ax, title in [(ax1, 'Champs électriques négatifs'), (ax2, 'Champs électriques positifs')]:
        ax.set_xlabel('Amplitude du champ électrique (V/m)')
        ax.set_title(f'{title_prefix} - {title}')
        #ax.set_yscale('log')

    ax1.set_ylabel('Durée du pulse (s)')

    # Ajout de la barre de couleur entre les deux graphiques
    cbar = fig.colorbar(pcm_pos, ax=[ax1, ax2], label=title_prefix)

    # Sauvegarde ou affichage du graphique


    plt.show()

    # Analyse statistique
    print("\nAnalyse statistique des populations:")
    print(f"{title_prefix} moyenne: {np.mean(Z):.3f}")
    print(f"{title_prefix} maximale: {np.max(Z):.3f}")

    # Trouver les paramètres optimaux
    max_pop_idx = np.unravel_index(np.argmax(Z), Z.shape)

    print(f"\nParamètres optimaux pour la {title_prefix} maximale:")
    print(f"  - Durée: {dt_values[max_pop_idx[0]]:.2e} s")
    print(f"  - Champ électrique: {amplitudes[max_pop_idx[1]]:.2f} V/m")
    print(f"  - {title_prefix}: {Z[max_pop_idx]:.3f}")

    return fig, (ax1, ax2)

# Exemple d'utilisation
if __name__ == "__main__":
    data_path = "Plot/resultats_multi_dt_lin2025-05-12 14:57:22.524554.npz"
    plot_optimization_results(data_path,'l0_pop',title_prefix='L0 population')
    plot_optimization_results(data_path,'l1_pop',title_prefix='L1 population')
    plot_optimization_results(data_path,'l2_pop',title_prefix='L2 population')
    plot_optimization_results(data_path,'l3_pop',title_prefix='L3 population')
    plot_optimization_results(data_path, 'l4_pop', title_prefix="Population l=4")
    plot_optimization_results(data_path, 'l5_pop', title_prefix="Population l=5")
    plot_optimization_results(data_path, 'l6_pop', title_prefix="Population l=6")
    plot_optimization_results(data_path, 'l7_pop', title_prefix="Population l=7")
    plot_optimization_results(data_path, 'l8_pop', title_prefix="Population l=8")
    plot_optimization_results(data_path, 'l9_pop', title_prefix="Population l=9")
    plot_optimization_results(data_path,'l10_pop', title_prefix="Population l=10")
    plot_optimization_results(data_path,'l11_pop', title_prefix="Population l=11")
    plot_optimization_results(data_path,'l12_pop', title_prefix="Population l=12")
    plot_optimization_results(data_path,'l13_pop', title_prefix="Population l=13")
    plot_optimization_results(data_path,'l14_pop', title_prefix="Population l=14")
    plot_optimization_results(data_path,'l15_pop', title_prefix="Population l=15")
    plot_optimization_results(data_path,'l16_pop', title_prefix="Population l=16")
    plot_optimization_results(data_path,'l17_pop', title_prefix="Population l=17")
    plot_optimization_results(data_path,'l18_pop', title_prefix="Population l=18")
    plot_optimization_results(data_path,'l19_pop', title_prefix="Population l=19")
    plot_optimization_results(data_path,'l20_pop', title_prefix="Population l=20")
    plot_optimization_results(data_path,'l21_pop', title_prefix="Population l=21")
    plot_optimization_results(data_path,'l22_pop', title_prefix="Population l=22")
    plot_optimization_results(data_path,'l23_pop', title_prefix="Population l=23")
    plot_optimization_results(data_path,'l24_pop', title_prefix="Population l=24")
    plot_optimization_results(data_path,'l25_pop', title_prefix="Population l=25")
    plot_optimization_results(data_path,'l26_pop', title_prefix="Population l=26")
    plot_optimization_results(data_path,'l27_pop', title_prefix="Population l=27")
    plot_optimization_results(data_path,'l28_pop', title_prefix="Population l=28")
    plot_optimization_results(data_path,'l29_pop', title_prefix="Population l=29")
    plot_optimization_results(data_path,'l30_pop', title_prefix="Population l=30")
    plot_optimization_results(data_path,'l31_pop', title_prefix="Population l=31")
    plot_optimization_results(data_path,'l32_pop', title_prefix="Population l=32")
    plot_optimization_results(data_path,'l33_pop', title_prefix="Population l=33")
    plot_optimization_results(data_path,'l34_pop', title_prefix="Population l=34")
    """
    plot_optimization_results(data_path,'l35_pop', title_prefix="Population l=35")
    plot_optimization_results(data_path,'l36_pop', title_prefix="Population l=36")
    plot_optimization_results(data_path,'l37_pop', title_prefix="Population l=37")
    plot_optimization_results(data_path,'l38_pop', title_prefix="Population l=38")
    plot_optimization_results(data_path,'l39_pop', title_prefix="Population l=39")
    plot_optimization_results(data_path,'l40_pop', title_prefix="Population l=40")
    plot_optimization_results(data_path,'l41_pop', title_prefix="Population l=41")
    plot_optimization_results(data_path,'l42_pop', title_prefix="Population l=42")
    plot_optimization_results(data_path,'l43_pop', title_prefix="Population l=43")
    plot_optimization_results(data_path,'l44_pop', title_prefix="Population l=44")
    plot_optimization_results(data_path,'l45_pop', title_prefix="Population l=45")
    plot_optimization_results(data_path,'l46_pop', title_prefix="Population l=46")
    plot_optimization_results(data_path,'l47_pop', title_prefix="Population l=47")
    plot_optimization_results(data_path,'l48_pop', title_prefix="Population l=48")
    plot_optimization_results(data_path,'l49_pop', title_prefix="Population l=49")
    plot_optimization_results(data_path,'l50_pop', title_prefix="Population l=50")
    plot_optimization_results(data_path,'l51_pop', title_prefix="Population l=51")
    plot_optimization_results(data_path,'l52_pop', title_prefix="Population l=52")
    plot_optimization_results(data_path,'l53_pop', title_prefix="Population l=53")
    plot_optimization_results(data_path,'l54_pop', title_prefix="Population l=54")
    plot_optimization_results(data_path,'l55_pop', title_prefix="Population l=55")
    plot_optimization_results(data_path,'l56_pop', title_prefix="Population l=56")
    plot_optimization_results(data_path,'l57_pop', title_prefix="Population l=57")
    plot_optimization_results(data_path,'l58_pop', title_prefix="Population l=58")
    plot_optimization_results(data_path,'l59_pop', title_prefix="Population l=59")
    plot_optimization_results(data_path,'l60_pop', title_prefix="Population l=60")
    plot_optimization_results(data_path,'l61_pop', title_prefix="Population l=61")
    plot_optimization_results(data_path,'l62_pop', title_prefix="Population l=62")
    plot_optimization_results(data_path,'l63_pop', title_prefix="Population l=63")
    plot_optimization_results(data_path,'l64_pop', title_prefix="Population l=64")
    plot_optimization_results(data_path,'l65_pop', title_prefix="Population l=65")
    plot_optimization_results(data_path,'l66_pop', title_prefix="Population l=66")
    plot_optimization_results(data_path,'l67_pop', title_prefix="Population l=67")
    plot_optimization_results(data_path,'l68_pop', title_prefix="Population l=68")
    plot_optimization_results(data_path,'l69_pop', title_prefix="Population l=69")
    plot_optimization_results(data_path,'l70_pop', title_prefix="Population l=70")
    plot_optimization_results(data_path,'l71_pop', title_prefix="Population l=71")
    plot_optimization_results(data_path,'l72_pop', title_prefix="Population l=72")
    plot_optimization_results(data_path,'l73_pop', title_prefix="Population l=73")
    plot_optimization_results(data_path,'l74_pop', title_prefix="Population l=74")
    plot_optimization_results(data_path,'l75_pop', title_prefix="Population l=75")
    plot_optimization_results(data_path,'l76_pop', title_prefix="Population l=76")
    plot_optimization_results(data_path,'l77_pop', title_prefix="Population l=77")
    plot_optimization_results(data_path,'l78_pop', title_prefix="Population l=78")
    """
