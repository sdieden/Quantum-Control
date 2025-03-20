import numpy as np
import matplotlib.pyplot as plt

import sys, os
import cmath as cm
import pathlib
from numpy.ma.core import shape

# A CORRIGER

newModPath=pathlib.Path(os.path.dirname(os.path.abspath(__file__)),'NewModule')
sys.path.insert(0, str(newModPath))
from arc_sam import *  # Import ARC (Alkali Rydberg Calculator)

atom = Calcium40()
calc = StarkMap(atom)

n = 35
l = 3
j = 3
mj = 0
s = 0
nmin = n - 3
nmax = n + 3
lmax = nmax - 1

"""
Rappel des unités atomiques  : 
action : hbar = 1 , hbar = 1.1*10^-34 J.s
energie : hartree, 1 Hartree = 4.359*10^-18 J
temps : 1 t.ua = 2.418*10^-17 s
champ électrique : 1 ce.ua = 5.14*10^11V/m

"""
conv_en = 4.359e18
conv_t = 2.418e17
conv_ce = 5.14e-11
hbar = 1
calc.defineBasis(n,l,j,mj,nmin,nmax,lmax,s=s, progressOutput=True)


def U(en, dt):
    """
    arg : en : is the energy of the level considered.
    arg : dt : is the time step.
    result : time evolution operator of the level considered, it's a phase on the WaveFunction.

    in this code, the U operator will be used in two cases.
    First evolution when there is no field applied -> U must takes the energy of the atomic level
    When the field is applied -> U must take the energy of the considered Stark level

    """
    u = cm.exp(0 - 1j * en * dt / hbar)
    return u


def apply_pulse(psi_in, pulse):
    psi_out = np.zeros(len(psi_in), dtype=np.complex128)
    """
    Probably better to do the conv_to_ua in the U fonction

    :arg psi_in: wavefunction before the pulse
    :arg pulse: list of 2 elts, pulse amplitude in V/m and pulse duration in s
    :return psi_out: wavefunction after the pulse

    """
    if pulse.stark:
        # Transformation vers la base Stark
        psi_stark = np.zeros(len(psi_in), dtype=np.complex128)
        for stark_state_idx in range(len(psi_in)):
            # Calcul de la composante Stark
            for atomic_comp in calc.composition[pulse.index_of_amplitude()][stark_state_idx]:
                atomic_idx = atomic_comp[1]
                coeff = atomic_comp[0]
                psi_stark[stark_state_idx] += coeff * psi_in[atomic_idx]

        # Évolution temporelle dans la base Stark
        for stark_state_idx in range(len(psi_stark)):
            en = calc.y[pulse.index_of_amplitude()][stark_state_idx] * conv_en
            psi_stark[stark_state_idx] *= U(en, pulse.duration * conv_t)

        # Retour à la base atomique
        for atomic_idx in range(len(psi_in)):
            for stark_state_idx in range(len(psi_stark)):
                for comp in calc.composition[pulse.index_of_amplitude()][stark_state_idx]:
                    if comp[1] == atomic_idx:
                        psi_out[atomic_idx] += comp[0] * psi_stark[stark_state_idx]

    else:
        # Évolution sans effet Stark
        for state in range(len(psi_in)):
            en = calc.y[0][state] * conv_en
            psi_out[state] = psi_in[state] * U(en, pulse.duration * conv_t)
    return psi_out


initial_coupled = np.zeros(len(calc.basisStates), dtype=np.complex128)
initial_coupled[calc.indexOfCoupledState] = 1


def pulse_evolution(pulseList, initial_coupled=initial_coupled):
    """
    :arg pulseList: list of 2 long lists. The first argument is the Electric field
    and the second argument is the duration of the field
    ex : pulseList = [(30,5)(0,10)(20,1)]
    This would be a 30V/m pulse during 5 seconds followed by a null field for 10 seconds and finally a 20V/m pulse during 10 seconds.

    """
    # Réinitialiser les pulses à chaque exécution
    Pulse.liste_pulse = []  # <-- Ajout crucial
    Pulse.amplitudes_list = set()  # <-- Réinitialisation des amplitudes

    # Vérification des pulses
    for pulse in pulseList:
        if len(pulse) != 2:
            raise ValueError(f'pulse {pulse} doit avoir 2 éléments (amplitude, durée)')
        if pulse[1] <= 0:
            raise ValueError(f'duration is negative', {pulse[1]}, 'must be positive')

    # Création des objets Pulse
    for amplitude, duration in pulseList:
        Pulse(amplitude, duration)  # Ajoute à Pulse.liste_pulse

    # initial state is given by initial_coupled, can be changed to any initial state.
    # initial_psi = np.zeros(len(calc.basisStates),dtype=np.complex128)

    initial_psi = initial_coupled

    psi = [initial_psi]

    # Diagonaliser avec toutes les amplitudes
    calc.diagonalise(sorted(tuple(Pulse.amplitudes_list)), upTo=-1, progressOutput=True)

    # Appliquer chaque pulse dans l'ordre correct
    for i in range(len(Pulse.liste_pulse)):
        current_pulse = Pulse.liste_pulse[i]
        psi.append(apply_pulse(psi[-1], current_pulse))

    return psi


def total_population(psi_array):
    abs_array = np.abs(psi_array)
    population = np.zeros(len(psi_array))
    for k in range(len(abs_array)):
        population[k] = np.sum(abs_array[k] ** 2)

    return population


# Test serie de pulses
pulse_list = [(30, 5e-7), (0, 10e-7), (20, 1e-7), (0, 10e-7), (
50, 10e-7)]  # liste des pulses avec amplitude[V/m] + durée[s] #contient l'ensemble des états après chaque pulse
pulse_list_test = [(30, 5e-7), (20, 1e-7), (40, 1e-7), (60, 1e-7), (80, 1e-7)]
initial_coupled_test = np.zeros(len(calc.basisStates), dtype=np.complex128)
##1st test

# for i in range(len(initial_coupled_test)):
#    initial_coupled_test[i] = 1/np.sqrt(len(calc.basisStates))
##2nd test
# initial_coupled_test[calc.indexOfCoupledState-2] = 1/np.sqrt(2)
# initial_coupled_test[calc.indexOfCoupledState-3] = 1/np.sqrt(2)

# 3d test
initial_coupled_test[calc.indexOfCoupledState - 2] = 1
print(calc.basisStates[calc.indexOfCoupledState - 2])
print(initial_coupled_test)
psi_evolution = pulse_evolution(pulse_list)
psi_evolution_test = pulse_evolution(pulse_list_test)
pop_states = []  # population des états, cad juste abs(coef)**2
pop_states_test = []
# Boucle de vérification et stockage
for i, psi in enumerate(psi_evolution):
    # Calcul de la population totale
    pop_total = np.sum(np.abs(psi) ** 2)

    # Calcul des populations individuelles
    pop_state = np.abs(psi) ** 2  # Conversion en liste Python

    # Stockage des résultats
    pop_states.append(pop_state)

    # Affichage de la population totale
    # print(f"Population après pulse {i}: {pop_total:.15f}")
for i, psi in enumerate(psi_evolution_test):
    # Calcul de la population totale
    pop_total_test = np.sum(np.abs(psi) ** 2)

    # Calcul des populations individuelles
    pop_state_test = np.abs(psi) ** 2  # Conversion en liste Python

    # Stockage des résultats
    pop_states_test.append(pop_state_test)

    # Affichage de la population totale
    # print(f"Population après pulse {i}: {pop_total_test:.15f}")

# Exemple d'accès aux données

# print("\nPopulation de tous les états après le 1er pulse:", pop_states[1])
# print("Population du 5ème état après le 2ème pulse:", pop_states[2][4])

import matplotlib.pyplot as plt

# Création du plot logarithmique
plt.figure(figsize=(10, 6))

# Tracer les populations en échelle logarithmique
plt.semilogy(pop_states[-1], label='psi_evolution (3ème pulse)', marker='o')
plt.semilogy(pop_states_test[-1], label='psi_evolution_test (1er pulse)', marker='x')

# Personnalisation du graphique
plt.xlabel('Index de l\'état')
plt.ylabel('Population (échelle log)')
plt.title('Comparaison des populations d\'états\naprès application d\'un pulse de 20V/m pendant 1e-7s')
plt.grid(True)
plt.legend()

# Ajuster les limites de l'axe y pour mieux voir les petites valeurs
plt.ylim(1e-10, 1.1)

plt.show()
# Définir les couleurs et marqueurs (manquants dans le code précédent)
colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y']
markers = ['o', 's', '^', 'D', 'x', '*', '+']
# Alternative : grille de graphiques
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Tracer chaque pulse dans son propre sous-graphique
for i, pop in enumerate(pop_states):
    if i >= len(axes):
        break

    if i == 0:
        title = 'État initial'
    else:
        title = f'Après pulse {i}'

    axes[i].semilogy(pop,
                     color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     markersize=3,
                     linestyle='-',
                     markevery=20,
                     label=calc.basisStates[calc.indexOfCoupledState])

    axes[i].set_title(title)
    axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[i].set_ylim(1e-10, 1.1)

    # Ajouter un indicateur du pic principal
    max_idx = np.argmax(pop)
    axes[i].annotate(f'Max: état {max_idx}\n({pop[max_idx]:.4f})',
                     xy=(max_idx, pop[max_idx]),
                     xytext=(max_idx + 10, pop[max_idx] * 0.5),
                     arrowprops=dict(arrowstyle='->'))

# Ajouter des labels communs
fig.text(0.5, 0.04, 'Index de l\'état', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Population (échelle log)', ha='center', va='center', rotation='vertical', fontsize=12)
fig.suptitle('Évolution des populations d\'états après chaque pulse', fontsize=14)
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', fontsize=12)
"""
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
"""
fig.legend(handles, labels, loc='upper left', fontsize=12)
plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
plt.show()

# Définir les couleurs et marqueurs (manquants dans le code précédent)
colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y']
markers = ['o', 's', '^', 'D', 'x', '*', '+']
# Alternative : grille de graphiques
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Tracer chaque pulse dans son propre sous-graphique
for i, pop in enumerate(pop_states_test):
    if i >= len(axes):
        break

    if i == 0:
        title = 'État initial'
    else:
        title = f'Après pulse {i}'

    axes[i].semilogy(pop,
                     color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     markersize=3,
                     linestyle='-',
                     markevery=20,
                     label=calc.basisStates[calc.indexOfCoupledState])

    axes[i].set_title(title)
    axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[i].set_ylim(1e-10, 1.1)

    # Ajouter un indicateur du pic principal
    max_idx = np.argmax(pop)
    axes[i].annotate(f'Max: état {max_idx}\n({pop[max_idx]:.4f})',
                     xy=(max_idx, pop[max_idx]),
                     xytext=(max_idx + 10, pop[max_idx] * 0.5),
                     arrowprops=dict(arrowstyle='->'))

# Ajouter des labels communs
fig.text(0.5, 0.04, 'Index de l\'état', ha='center', va='center', fontsize=12)
fig.text(0.06, 0.5, 'Population (échelle log)', ha='center', va='center', rotation='vertical', fontsize=12)
fig.suptitle('Évolution des populations d\'états après chaque pulse', fontsize=14)
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', fontsize=12)
"""
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
"""
fig.legend(handles, labels, loc='upper left', fontsize=12)
plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
plt.show()

print_pop_states = np.array(pop_states)
print("\nPopulation du state 35F3 après chaque pulse:", print_pop_states[:, calc.indexOfCoupledState])
print("Population du state 5 après 2ème pulse:", print_pop_states[2, 5])


