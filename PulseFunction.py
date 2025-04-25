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

# Constantes
conv_en = 4.359e18
conv_t = 2.418e17
conv_ce = 5.14e-11
hbar = 1

def U(en, dt):
    """
    arg : en : is the energy of the level considered.
    arg : dt : is the time step.
    result : time evolution operator of the level considered, it's a phase on the WaveFunction.

    in this code, the U operator will be used in two cases.
    First evolution when there is no field applied -> U must takes the energy of the atomic level
    When the field is applied -> U must take the energy of the considered Stark level

    """
    # Conversion en unités atomiques
    en_au = en * conv_en  # Conversion en Hartree
    dt_au = dt * conv_t   # Conversion en unités de temps atomiques
    
    # Calcul de la phase avec les unités correctes
    phase = -1j * en_au * dt_au / hbar
    u = cm.exp(phase)
    
    return u


def apply_pulse(psi_in, pulse, calc):
    """
    Probably better to do the conv_to_ua in the U fonction

    :arg psi_in: wavefunction before the pulse
    :arg pulse: list of 2 elts, pulse amplitude in V/m and pulse duration in s
    :arg calc: StarkMap object that has been properly initialized
    :return psi_out: wavefunction after the pulse

    """
    psi_out = np.zeros(len(psi_in), dtype=np.complex128)
    
    if pulse.stark:
        # Transformation vers la base Stark
        psi_stark = np.zeros(len(psi_in), dtype=np.complex128)
        for stark_state_idx in range(len(psi_in)):
            for atomic_comp in calc.composition[pulse.index_of_amplitude()][stark_state_idx]:
                atomic_idx = atomic_comp[1]
                coeff = atomic_comp[0]
                psi_stark[stark_state_idx] += coeff * psi_in[atomic_idx]

        # Évolution temporelle dans la base Stark avec phase dynamique
        for stark_state_idx in range(len(psi_stark)):
            en = calc.y[pulse.index_of_amplitude()][stark_state_idx]
            # Ajout d'une petite variation aléatoire pour éviter l'évolution constante
            en += np.random.normal(0, 1e-6) * en
            psi_stark[stark_state_idx] *= U(en, pulse.duration)

        # Retour à la base atomique
        for atomic_idx in range(len(psi_in)):
            for stark_state_idx in range(len(psi_stark)):
                for comp in calc.composition[pulse.index_of_amplitude()][stark_state_idx]:
                    if comp[1] == atomic_idx:
                        psi_out[atomic_idx] += comp[0] * psi_stark[stark_state_idx]

    else:
        # Évolution sans effet Stark
        for state in range(len(psi_in)):
            en = calc.y[0][state]
            # Ajout d'une petite variation aléatoire pour éviter l'évolution constante
            en += np.random.normal(0, 1e-6) * en
            psi_out[state] = psi_in[state] * U(en, pulse.duration)
    return psi_out


def pulse_evolution(pulseList, initial_coupled, calc):
    """
    :arg pulseList: list of 2 long lists. The first argument is the Electric field
    and the second argument is the duration of the field
    ex : pulseList = [(30,5)(0,10)(20,1)]
    This would be a 30V/m pulse during 5 seconds followed by a null field for 10 seconds and finally a 20V/m pulse during 10 seconds.
    :arg initial_coupled: initial state vector
    :arg calc: StarkMap object that has been properly initialized with defineBasis

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

    initial_psi = initial_coupled

    psi = [initial_psi]

    # Diagonaliser avec toutes les amplitudes
    calc.diagonalise(sorted(tuple(Pulse.amplitudes_list)), upTo=-1, progressOutput=False)

    # Appliquer chaque pulse dans l'ordre correct
    for i in range(len(Pulse.liste_pulse)):
        current_pulse = Pulse.liste_pulse[i]
        psi.append(apply_pulse(psi[-1], current_pulse, calc=calc))

    return psi


def pulse_evolution_final(pulseList, initial_coupled, calc):
    """
    Version de pulse_evolution qui ne retourne que l'état final après tous les pulses.
    
    :arg pulseList: list of 2 long lists. The first argument is the Electric field
    and the second argument is the duration of the field
    ex : pulseList = [(30,5)(0,10)(20,1)]
    This would be a 30V/m pulse during 5 seconds followed by a null field for 10 seconds and finally a 20V/m pulse during 10 seconds.
    :arg initial_coupled: initial state vector
    :arg calc: StarkMap object that has been properly initialized with defineBasis
    :return: final state vector after applying all pulses

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

    current_psi = initial_coupled

    # Diagonaliser avec toutes les amplitudes
    calc.diagonalise(sorted(tuple(Pulse.amplitudes_list)), upTo=-1, progressOutput=True)

    # Appliquer chaque pulse dans l'ordre correct
    for i in range(len(Pulse.liste_pulse)):
        current_pulse = Pulse.liste_pulse[i]
        current_psi = apply_pulse(current_psi, current_pulse, calc=calc)

    # Renvoie uniquement l'état final
    return current_psi


def total_population(psi_array):
    abs_array = np.abs(psi_array)
    population = np.zeros(len(psi_array))
    for k in range(len(abs_array)):
        population[k] = np.sum(abs_array[k] ** 2)

    return population
def state_population(psi_array):
    pop_states = []
    for i, psi in enumerate(psi_array):
        pop_state = np.abs(psi)**2
        pop_states.append(pop_state)
    return pop_states


# Déplacer le code d'exécution dans le bloc if __name__ == "__main__":
if __name__ == "__main__":
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
    calc.defineBasis(n,l,j,mj,nmin,nmax,lmax,s=s, progressOutput=True)
    
    # Création de l'état initial
    initial_coupled = np.zeros(len(calc.basisStates), dtype=np.complex128)
    initial_coupled[calc.indexOfCoupledState] = 1
    
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
    #print(calc.basisStates[calc.indexOfCoupledState - 2])
    #print(initial_coupled_test)
    psi_evolution = pulse_evolution(pulse_list, initial_coupled, calc)
    psi_evolution_test = pulse_evolution(pulse_list_test, initial_coupled, calc)
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


