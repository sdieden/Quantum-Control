import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fft import rfft,fft, ifft, fftshift, fftfreq

# Charger les données
data = np.load("Plot/resultats_multi_dt_lin2025-05-12 12:43:22.130483.npz", allow_pickle=True)
#print(data)
#print(data['dt_values'])
dt_values = data['dt_values']
#print('dt',type(dt_values))
amplitudes = data['amplitudes']
#l_sup_10_pop = data['l_sup_10_pop'].item()
l10_pop = data['l10_pop'].item()

#print('2',l_sup_10_pop[dt_values[0]])
#print(amplitudes)
## on veut aller chercher toutes les valeurs pour une valeur de champ et en faire la transformee de fourier
# pour aller chercher une colonne, on doit 1. trouver la position de l'amplitude du champ electrique
EF_idx = 10

# 2. aller chercher toutes les valeurs à chaque temps pour ce champ electrique
sup = np.zeros(len(dt_values))
pop = np.zeros(len(dt_values))
for i in range(len(dt_values)):
    #sup[i] = l_sup_10_pop[dt_values[i]][EF_idx]
    pop[i] = l10_pop[dt_values[i]][EF_idx]
print(data)
plt.plot(dt_values, pop)
print('len=',len(pop))
# TDF dessus
L_SUP = fft(sup)
POP = fft(pop)
plt.figure()
plt.subplot(211)
plt.plot(np.real(POP))
plt.ylabel("partie reelle")
plt.subplot(212)
plt.plot(np.imag(POP))
plt.ylabel("partie imaginaire")
plt.show()

