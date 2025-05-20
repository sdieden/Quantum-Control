import numpy as np
import matplotlib.pyplot as plt
import sys, os
import cmath as cm
import pathlib
newModPath=pathlib.Path(os.path.dirname(os.path.abspath(__file__)),'NewModule')
sys.path.insert(0, str(newModPath))
from arc import *  # Import ARC (Alkali Rydberg Calculator)
from NewModule.arc.PulseClass import Pulse
# Constantes
conv_ce = 5.14e-11    # conversion de champ électrique, useless ?
hbar = 1
ghz_to_hartree = 1.51983e-7  # 1 GHz = 1.51983e-7 Hartree
seconds_to_au = 4.13414e16  # 1 s = 4.13414e16 atomic time units


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
    en_au = en * ghz_to_hartree # Conversion en Hartree
    dt_au = dt * seconds_to_au   # Conversion en unités de temps atomiques
    
    # Calcul de la phase avec les unités correctes
    phase = -1j * en_au * dt_au #/ hbar #is supposed to be 1
    u = cm.exp(phase)
    
    return u

def create_basis_change(psi_in,pulse,calc):
    """
    create matrix for basis change, from stark to atomic, stark to stark

    """
    stark_basis = calc.composition[pulse.index_of_amplitude]
    # for atomic to stark first
    length = len(psi_in)
    matrix = [[0 for _ in range(length)] for _ in range(length)]
    for i,state in stark_basis:
        idx = state[1]
        coef = state[0]
        matrix[idx][i] = coef
    return matrix



def apply_pulse(psi_in, pulse, calc): #made by claude
    """
    Apply a pulse to a quantum state.

    Args:
        psi_in: wavefunction before the pulse (in atomic basis)
        pulse: Pulse object with amplitude (V/m) and duration (s)
        calc: StarkMap object that has been properly initialized with diagonalise method

    Returns:
        psi_out: wavefunction after the pulse (in atomic basis)
    """
    psi_out = np.zeros(len(psi_in), dtype=np.complex128)

    if pulse.stark:
        # Transformation to Stark basis
        psi_stark = np.zeros(len(psi_in), dtype=np.complex128)

        # Get the composition of the Stark states in terms of atomic states
        composition = calc.composition[pulse.index_of_amplitude()]

        # Project onto Stark basis
        for stark_idx in range(len(composition)): # for every stark level associated to the amplitude
            stark_comp = composition[stark_idx] # taking one Stark level
            for atomic_comp in stark_comp:  # looking at the decomposition of the stark level in the atomic basis
                coef = atomic_comp[0]  # coefficient in the state decomposition ( the stark level has a composition of 57% of the |i> atomic level for exemple)
                atomic_idx = atomic_comp[1]  # index of the atomic state ( |i> is the 32th term in the basis for exemple)
                psi_stark[stark_idx] += coef * psi_in[atomic_idx]

        # Apply time evolution in Stark basis
        for stark_idx in range(len(psi_stark)):
            # Get energy of this Stark state
            energy = calc.y[pulse.index_of_amplitude()][stark_idx]
            # Apply phase evolution
            psi_stark[stark_idx] *= U(energy, pulse.duration)#U(energy, pulse.duration*conv_t) #ancienne version

        # Transform back to atomic basis
        for atomic_idx in range(len(psi_in)): # pour tous les états atomiques
            for stark_idx in range(len(composition)): # on regarde chaque etat stark
                stark_comp = composition[stark_idx] #
                for comp in stark_comp:
                    if comp[1] == atomic_idx:  # If this component maps to our atomic state
                        psi_out[atomic_idx] += comp[0] * psi_stark[stark_idx]

    else:
        # Evolution without Stark effect - directly in atomic basis
        for state_idx in range(len(psi_in)):
            # Energy in field-free case
            energy = calc.y[0][state_idx]
            # Apply phase evolution
            psi_out[state_idx] = psi_in[state_idx] * U(energy, pulse.duration)

    return psi_out

def pulse_evolution(pulseList, initial_coupled, calc, precomputed_eigenvectors=None):
    """
    :arg pulseList: list of 2 long lists. The first argument is the Electric field
    and the second argument is the duration of the field
    ex : pulseList = [(30,5)(0,10)(20,1)]
    This would be a 30V/m pulse during 5 seconds followed by a null field for 10 seconds and finally a 20V/m pulse during 10 seconds.
    :arg initial_coupled: initial state vector
    :arg calc: StarkMap object that has been properly initialized with defineBasis
    :arg precomputed_eigenvectors: Dictionnaire des eigenvectors précalculés pour chaque amplitude

    """
    # Réinitialiser les pulses à chaque exécution
    Pulse.liste_pulse = []
    Pulse.amplitudes_list = set()

    # Vérification des pulses
    for pulse in pulseList:
        if len(pulse) != 2:
            raise ValueError(f'pulse {pulse} doit avoir 2 éléments (amplitude, durée)')
        if pulse[1] <= 0:
            raise ValueError(f'duration is negative', {pulse[1]}, 'must be positive')

    # Creation des objets Pulse
    for amplitude, duration in pulseList:
        Pulse(amplitude, duration)  # Ajoute à Pulse.liste_pulse

    initial_psi = initial_coupled

    psi = [initial_psi]

    # Diagonaliser avec toutes les amplitudes si non précalculées
    if precomputed_eigenvectors is None:
        calc.diagonalise(sorted(tuple(Pulse.amplitudes_list)), upTo=-1, progressOutput=False)
    else:
        # Utiliser les eigenvectors précalculés
        calc.composition = precomputed_eigenvectors

    # Appliquer chaque pulse dans l'ordre correct
    for i in range(len(Pulse.liste_pulse)):
        current_pulse = Pulse.liste_pulse[i]
        psi.append(apply_pulse(psi[-1], current_pulse, calc=calc))

    return psi


def pulse_evolution_final(pulseList, initial_wf, calc, precomputed_eigenvectors=None):
    """
    Version de pulse_evolution qui ne retourne que l'état final après tous les pulses.
    
    Args:
        pulseList: list of 2 long lists. The first argument is the Electric field
        and the second argument is the duration of the field
        ex : pulseList = [(30,5)(0,10)(20,1)]
        This would be a 30V/m pulse during 5 seconds followed by a null field for 10 seconds and finally a 20V/m pulse during 10 seconds.
        initial_wf: État initial
        calc: Objet StarkMap initialisé
        precomputed_eigenvectors: Dictionnaire des eigenvectors précalculés pour chaque amplitude
    
    Returns:
        final state vector after applying all pulses
    """
    # Réinitialiser les pulses à chaque exécution
    Pulse.liste_pulse = []
    Pulse.amplitudes_list = set()

    # Vérification des pulses
    for pulse in pulseList:
        if len(pulse) != 2:
            raise ValueError(f'pulse {pulse} doit avoir 2 éléments (amplitude, durée)')
        if pulse[1] <= 0:
            raise ValueError(f'duration is negative', {pulse[1]}, 'must be positive')

    # Création des objets Pulse
    for amplitude, duration in pulseList:
        Pulse(amplitude, duration)  # Ajoute à Pulse.liste_pulse

    current_psi = initial_wf

    # Diagonaliser avec toutes les amplitudes si non précalculées
    if precomputed_eigenvectors is None:
        calc.diagonalise(sorted(tuple(Pulse.amplitudes_list)), upTo=-1, progressOutput=True)
    else:
        # Utiliser les eigenvectors précalculés
        calc.composition = precomputed_eigenvectors

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

def searching_best_pulse(dt_array, amplitudes_array, initial_wf, calc, lmin = 10, lmax=35):
    """
    Recherche le meilleur pulse unique pour maximiser la population dans un état cible.
    
    Args:
        dt_array: Liste des durées possibles pour les pulses
        amplitudes_array: Liste des amplitudes possibles pour les pulses
        initial_wf: État initial
        calc: Objet StarkMap initialisé
        lmax: Valeur maximale de l pour les états à considérer (par défaut 35)
    
    Returns:
        optimized_pulse_sequence_time: Liste contenant la durée optimale
        optimized_pulse_sequence_amplitude: Liste contenant l'amplitude optimale
        all_l_populations: Dictionnaire des populations pour chaque l
        all_l_coefficients: Dictionnaire des coefficients pour chaque amplitude
        final_wf: État final après application du pulse optimal
    """
    # Initialisation des variables
    idx_dt = None
    idx_amplitudes = None
    final_wf = None
    all_l_populations = {}
    all_l_coefficients = {}
    
    # Initialisation des dictionnaires
    for dt in dt_array:
        all_l_populations[dt] = {}
        all_l_coefficients[dt] = {}
        # Initialiser pour tous les niveaux l possibles dans la base
        for l_level in range(lmin, lmax + 1):  # +1 pour inclure lmax
            all_l_populations[dt][l_level] = []
        for amplitude in amplitudes_array:
            all_l_coefficients[dt][amplitude] = np.zeros(len(calc.basisStates), dtype=complex)
    
    optimized_wavefunction_population = 0
    optimized_pulse_sequence_time = []
    optimized_pulse_sequence_amplitude = []
    
    # Diagonaliser avec toutes les amplitudes une seule fois
    print("\nDiagonalisation avec toutes les amplitudes...")
    calc.diagonalise(sorted(amplitudes_array), upTo=-1, progressOutput=True)
    # Sauvegarder les eigenvectors
    precomputed_eigenvectors = calc.composition.copy()
    
    # Test de toutes les combinaisons possibles
    for dt in dt_array:
        print(f"\n=== Calculs avec durée = {dt} s ===")
        
        # Création de la liste des pulses à tester pour cette durée
        pulse_square = []
        for amplitude in amplitudes_array:
            pulse_square.append([(amplitude, dt)])
        
        # Test de chaque pulse
        for i, pulse_list in enumerate(pulse_square):
            # Application du pulse
            x = pulse_evolution(pulse_list, initial_coupled=initial_wf, calc=calc, precomputed_eigenvectors=precomputed_eigenvectors)
            all_l_coefficients[pulse_list[0][1]][pulse_list[0][0]] = x[-1]
            
            # Calcul des populations
            y = np.abs(x[-1]) ** 2
            
            # Calcul des populations par niveau l
            for l_level in range(lmin, lmax + 1):  # +1 pour inclure lmax
                l_pop = 0.0
                for idx, state in enumerate(calc.basisStates):
                    if state[1] == l_level:
                        l_pop += y[idx]
                all_l_populations[pulse_list[0][1]][l_level].append(l_pop)
                
                # Mise à jour du meilleur pulse si nécessaire
                if l_pop > optimized_wavefunction_population:
                    optimized_wavefunction_population = l_pop
                    final_wf = x[-1]
                    idx_amplitudes = pulse_list[0][0]
                    idx_dt = pulse_list[0][1]
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(pulse_square)} pulses")
    
    # Stockage du meilleur pulse trouvé
    optimized_pulse_sequence_time.append(idx_dt)
    optimized_pulse_sequence_amplitude.append(idx_amplitudes)
    
    return optimized_pulse_sequence_time, optimized_pulse_sequence_amplitude, all_l_populations, all_l_coefficients, final_wf

def optimize_l_population(target_l, dt_array, amplitudes_array, initial_wf, calc, n=35):
    """
    Optimise la population d'un niveau l spécifique pour n=35.
    
    Args:
        target_l: Le niveau l cible à optimiser
        dt_array: Liste des durées possibles pour les pulses
        amplitudes_array: Liste des amplitudes possibles pour les pulses
        initial_wf: État initial
        calc: Objet StarkMap initialisé
        n: Nombre quantique principal (par défaut 35)
    
    Returns:
        best_pulse: Le meilleur pulse trouvé (amplitude, durée)
        best_population: La population maximale atteinte
        all_populations: Dictionnaire des populations pour chaque combinaison
    """
    # Initialisation des variables
    best_population = 0
    best_pulse = None
    all_populations = {}
    
    # Creation d'une grille de resultats
    for dt in dt_array:
        all_populations[dt] = {}
        for amplitude in amplitudes_array:
            all_populations[dt][amplitude] = 0
    
    print(f"\nOptimisation de la population pour l = {target_l}")
    print("Test des différentes combinaisons...")
    
    # Diagonaliser avec toutes les amplitudes une seule fois
    print("\nDiagonalisation avec toutes les amplitudes...")
    calc.diagonalise(sorted(amplitudes_array), upTo=-1, progressOutput=True)
    # Sauvegarder les eigenvectors
    precomputed_eigenvectors = calc.composition.copy()
    
    # Test de toutes les combinaisons
    for dt in dt_array:
        print(f"\n=== Calculs avec durée = {dt} s ===")
        
        for amplitude in amplitudes_array:
            # Application du pulse
            pulse = [(amplitude, dt)]
            wf_after_pulse = pulse_evolution_final(pulse, initial_wf, calc, precomputed_eigenvectors)
            
            # Calcul de la population pour le niveau l cible
            populations = np.abs(wf_after_pulse) ** 2
            l_population = 0.0
            
            for idx, state in enumerate(calc.basisStates):
                if state[1] == target_l and state[0] == n:  # Verifie n et l
                    l_population += populations[idx]
            
            # Stockage du resultat
            all_populations[dt][amplitude] = l_population
            
            # Mise a jour du meilleur pulse si necessaire
            if l_population > best_population:
                best_population = l_population
                best_pulse = (amplitude, dt)
                print(f"  Nouvelle meilleure population trouvée :")
                print(f"    Amplitude : {amplitude} V/m")
                print(f"    Durée : {dt} s")
                print(f"    Population : {l_population:.6f}")

    plt.figure(figsize=(12, 8))
    dt_values = list(dt_array)
    amp_values = list(amplitudes_array)
    pop_matrix = np.zeros((len(dt_values), len(amp_values)))
    
    for i, dt in enumerate(dt_values):
        for j, amp in enumerate(amp_values):
            pop_matrix[i, j] = all_populations[dt][amp]

    # Graphique en 3D
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(amp_values, np.log10(dt_values))  # Utilisation de log10 pour l'axe des durées
    
    # Tracé de la surface
    surf = ax.plot_surface(X, Y, pop_matrix, cmap='viridis')
    
    # Personnalisation du graphique
    ax.set_xlabel('Amplitude (V/m)')
    ax.set_ylabel('log10(Durée) (s)')  # Modification du label pour refléter l'échelle logarithmique
    ax.set_zlabel('Population')
    ax.set_title(f'Population du niveau l={target_l} (n={n})')
    
    # Ajout d'une barre de couleur
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Ajout d'un point pour le meilleur pulse
    best_amp, best_dt = best_pulse
    best_pop = all_populations[best_dt][best_amp]
    ax.scatter([best_amp], [np.log10(best_dt)], [best_pop], color='red', s=100, label='Meilleur pulse')
    
    plt.show()

    return best_pulse, best_population, all_populations

def optimize_pulse_sequence(target_l, dt_array, amplitudes_array, initial_wf, calc, n=35, max_iterations=100, target_population=0.9):
    """
    Optimise une séquence de pulses pour maximiser la population d'un niveau l spécifique.
    
    Args:
        target_l: Le niveau l cible à optimiser
        dt_array: Liste des durées possibles pour les pulses
        amplitudes_array: Liste des amplitudes possibles pour les pulses
        initial_wf: État initial
        calc: Objet StarkMap initialisé
        n: Nombre quantique principal (par défaut 35)
        max_iterations: Nombre maximum d'itérations (par défaut 100)
        target_population: Population cible pour arrêter l'optimisation (par défaut 0.9)
    
    Returns:
        best_sequence: Liste des meilleurs pulses trouvés [(amplitude1, durée1), ...]
        best_population: La population maximale atteinte
        populations_history: Liste des populations à chaque étape
    """
    print(f"\nOptimisation d'une séquence de pulses pour l = {target_l}")
    print(f"Critères d'arrêt : {max_iterations} itérations max ou population > {target_population}")
    
    # Initialisation
    best_sequence = []         # array du meilleur sequencage de pulse
    populations_history = [0]  # Population initiale
    current_wf = initial_wf    # changement de wf a chaque iteration, la current_wf devient l'output wf de la derniere iteration
    iteration = 0              # bah... iteration
    
    while iteration < max_iterations:
        print(f"\n=== Pulse {iteration + 1} ===")
        
        # Optimisation du pulse actuel
        best_pulse, current_pop, _ = optimize_l_population(target_l, dt_array, amplitudes_array, current_wf, calc, n)
        
        # Verification si le pulse ameliore la population
        """if current_pop <= populations_history[-1]:
            print(f"La population n'augmente plus. Arrêt de l'optimisation.")
            break
        """
        # Verification si le pulse est meilleur que le premier pulse
        """if current_pop < populations_history[-1]:
            print(f"La population après {iteration}e pulse est moins performant que le premier pulse")
            break
        """
        # Mise a jour des resultats
        best_sequence.append(best_pulse)
        populations_history.append(current_pop)
        
        # Application du pulse pour l'itération suivante
        current_wf = pulse_evolution_final([best_pulse], current_wf, calc)
        
        # Vérification du critère de population
        if current_pop >= target_population:
            print(f"Population cible atteinte ({current_pop:.6f} >= {target_population})")
            break
        
        iteration += 1
    
    # Affichage des résultats
    print("\nRésultats de l'optimisation :")
    sequence_duration = 0
    for i, (amp, dt) in enumerate(best_sequence):
        print(f"Pulse {i+1}: amplitude = {amp} V/m, durée = {dt} s")
        print(f"Population après pulse {i+1}: {populations_history[i+1]:.6f}")
        sequence_duration += dt
    sequence_duration *= 1e-6
    print(f"Durée de la séquence : {sequence_duration*1e-6:.6f}")
    # Création du graphique de la séquence
    plt.figure(figsize=(15, 10))
    
    # Graphique des amplitudes
    plt.subplot(2, 1, 1)
    times = [0]
    amplitudes = [0]
    current_time = 0
    
    for amp, dt in best_sequence:
        current_time += dt
        times.extend([current_time - dt, current_time])
        amplitudes.extend([amp, amp])
    
    plt.step(times, amplitudes, where='post', label='Amplitude')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude (V/m)')
    plt.title('Séquence de pulses optimale')
    plt.grid(True)
    
    # Graphique des populations
    plt.subplot(2, 1, 2)
    steps = range(len(populations_history))
    plt.plot(steps, populations_history, 'o-', label='Population')
    plt.axhline(y=target_population, color='r', linestyle='--', label='Population cible')
    plt.xlabel('Étape')
    plt.ylabel('Population')
    plt.title('Évolution de la population')
    plt.xticks(steps, ['Initial'] + [f'Après pulse {i+1}' for i in range(len(best_sequence))])
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_sequence, populations_history[-1], populations_history

def optimize_sequence_grape(initial_state, target_l, calc, n_pulses=5, max_iterations=100, learning_rate=0.01):
    """
    Optimise une séquence de pulses en utilisant la méthode GRAPE.
    
    Args:
        initial_state: État initial
        target_l: Niveau l cible
        calc: Objet StarkMap initialisé
        n_pulses: Nombre de pulses dans la séquence
        max_iterations: Nombre maximum d'itérations
        learning_rate: Taux d'apprentissage pour la mise à jour du gradient
    
    Returns:
        best_sequence: Liste des meilleurs pulses trouvés [(amplitude1, durée1), ...]
        best_population: La population maximale atteinte
        populations_history: Liste des populations à chaque étape
    """
    # Initialisation
    current_state = initial_state
    pulse_sequence = []
    populations_history = []
    
    # Paramètres de contrôle
    min_field = -1500  # V/m
    max_field = 1500   # V/m
    min_duration = 1e-10  # s
    max_duration = 1e-6   # s
    
    # Initialisation aléatoire de la séquence
    for _ in range(n_pulses):
        amplitude = np.random.uniform(min_field, max_field)
        duration = np.random.uniform(min_duration, max_duration)
        pulse_sequence.append((amplitude, duration))
    
    best_population = 0
    best_sequence = pulse_sequence.copy()
    
    for iteration in range(max_iterations):
        # Propagation en avant
        current_state = initial_state
        for amplitude, duration in pulse_sequence:
            current_state = pulse_evolution_final([(amplitude, duration)], current_state, calc)
        
        # Calcul de la population cible
        populations = np.abs(current_state) ** 2
        target_population = 0.0
        for idx, state in enumerate(calc.basisStates):
            if state[1] == target_l:
                target_population += populations[idx]
        
        # Mise à jour du meilleur résultat
        if target_population > best_population:
            best_population = target_population
            best_sequence = pulse_sequence.copy()
        
        populations_history.append(target_population)
        
        # Calcul du gradient pour chaque pulse
        gradients = []
        for i in range(n_pulses):
            # Perturbation de l'amplitude
            perturbed_sequence = pulse_sequence.copy()
            perturbed_sequence[i] = (pulse_sequence[i][0] + 0.1, pulse_sequence[i][1])
            
            # Propagation avec la séquence perturbée
            perturbed_state = initial_state
            for amp, dur in perturbed_sequence:
                perturbed_state = pulse_evolution_final([(amp, dur)], perturbed_state, calc)
            
            # Calcul de la population perturbée
            perturbed_populations = np.abs(perturbed_state) ** 2
            perturbed_target_population = 0.0
            for idx, state in enumerate(calc.basisStates):
                if state[1] == target_l:
                    perturbed_target_population += perturbed_populations[idx]
            
            # Calcul du gradient
            gradient = (perturbed_target_population - target_population) / 0.1
            gradients.append(gradient)
        
        # Mise à jour de la séquence
        for i in range(n_pulses):
            new_amplitude = pulse_sequence[i][0] + learning_rate * gradients[i]
            new_amplitude = np.clip(new_amplitude, min_field, max_field)
            pulse_sequence[i] = (new_amplitude, pulse_sequence[i][1])
        
        if (iteration + 1) % 10 == 0:
            print(f"Itération {iteration + 1}/{max_iterations}")
            print(f"Population actuelle: {target_population:.6f}")
            print(f"Meilleure population: {best_population:.6f}")
    
    return best_sequence, best_population, populations_history

def change_stark_basis(psi_stark_in, old_amplitude, new_amplitude, calc):
    """
    Change la base Stark d'un état quantique.
    
    Args:
        psi_stark_in: État dans l'ancienne base Stark
        old_amplitude: Amplitude de l'ancienne base Stark
        new_amplitude: Amplitude de la nouvelle base Stark
        calc: Objet StarkMap initialisé
    
    Returns:
        psi_stark_out: État dans la nouvelle base Stark
    """
    # D'abord, on transforme vers la base atomique
    psi_atomic = np.zeros(len(psi_stark_in), dtype=np.complex128)
    old_composition = calc.composition[old_amplitude]
    
    for atomic_idx in range(len(psi_atomic)):
        for stark_idx in range(len(old_composition)):
            stark_comp = old_composition[stark_idx]
            for comp in stark_comp:
                if comp[1] == atomic_idx:
                    psi_atomic[atomic_idx] += comp[0] * psi_stark_in[stark_idx]
    
    # Ensuite, on transforme vers la nouvelle base Stark
    psi_stark_out = np.zeros(len(psi_stark_in), dtype=np.complex128)
    new_composition = calc.composition[new_amplitude]
    
    for stark_idx in range(len(new_composition)):
        stark_comp = new_composition[stark_idx]
        for atomic_comp in stark_comp:
            coef = atomic_comp[0]
            atomic_idx = atomic_comp[1]
            psi_stark_out[stark_idx] += coef * psi_atomic[atomic_idx]
    
    return psi_stark_out

def apply_pulse_stark(psi_stark_in, pulse, calc):
    """
    Applique un pulse à un état quantique en restant dans la base Stark.
    
    Args:
        psi_stark_in: État dans la base Stark avant le pulse
        pulse: Pulse object avec amplitude (V/m) et duration (s)
        calc: StarkMap object initialisé
    
    Returns:
        psi_stark_out: État dans la base Stark après le pulse
    """
    psi_stark_out = np.zeros(len(psi_stark_in), dtype=np.complex128)
    
    if pulse.stark:
        # Application de l'évolution temporelle dans la base Stark
        for stark_idx in range(len(psi_stark_in)):
            energy = calc.y[pulse.index_of_amplitude()][stark_idx]
            psi_stark_out[stark_idx] = psi_stark_in[stark_idx] * U(energy, pulse.duration)
    else:
        # Si pas d'effet Stark, on transforme vers la base atomique, on évolue, puis on revient
        psi_atomic = np.zeros(len(psi_stark_in), dtype=np.complex128)
        composition = calc.composition[pulse.index_of_amplitude()]
        
        # Transformation vers la base atomique
        for atomic_idx in range(len(psi_atomic)):
            for stark_idx in range(len(composition)):
                stark_comp = composition[stark_idx]
                for comp in stark_comp:
                    if comp[1] == atomic_idx:
                        psi_atomic[atomic_idx] += comp[0] * psi_stark_in[stark_idx]
        
        # Évolution dans la base atomique
        for state_idx in range(len(psi_atomic)):
            energy = calc.y[0][state_idx]
            psi_atomic[state_idx] *= U(energy, pulse.duration)
        
        # Retour vers la base Stark
        for stark_idx in range(len(composition)):
            stark_comp = composition[stark_idx]
            for atomic_comp in stark_comp:
                coef = atomic_comp[0]
                atomic_idx = atomic_comp[1]
                psi_stark_out[stark_idx] += coef * psi_atomic[atomic_idx]
    
    return psi_stark_out

# Déplacer le code d'exécution dans le bloc if __name__ == "__main__":
if __name__ == "__main__":
    print("\n=== Test de optimize_pulse_sequence ===")
    
    # Initialisation de l'atome et du calcul
    atom = Calcium40()
    calc = StarkMap(atom)
    
    # Paramètres de base
    n = 35
    l = 3
    j = 3
    mj = 0
    s = 0
    nmin = n - 1
    nmax = n + 2
    lmax = nmax - 1
    
    print("\nParamètres de base :")
    print(f"n = {n}, l = {l}, j = {j}, mj = {mj}")
    print(f"nmin = {nmin}, nmax = {nmax}, lmax = {lmax}")
    
    # Initialisation de la base
    calc.defineBasis(n, l, j, mj, nmin, nmax, lmax, s=s, progressOutput=True)
    
    # Création de l'état initial
    initial_wf = np.zeros(len(calc.basisStates), dtype=np.complex128)
    initial_wf[calc.indexOfCoupledState] = 1
    
    print(f"\nÉtat initial :")
    print(f"Index de l'état couplé : {calc.indexOfCoupledState}")
    print(f"État couplé : {calc.basisStates[calc.indexOfCoupledState]}")
    
    # Paramètres de test
    N = 300
    dt_array = np.logspace(-10, -8, N)
    amplitudes_array_first_pulse = np.linspace(800, 1500, 151)
    amplitudes_array = np.linspace(-1500, 1500, 11)
    print("\nParamètres de test :")
    print(f"Durées : {dt_array}")
    print(f"Amplitudes : {amplitudes_array}")
    
    # Optimisation de la séquence pour un niveau l spécifique
    target_l = 20  # Niveau l à optimiser
    best_sequence, best_pop, pop_history = optimize_pulse_sequence(
        target_l,
        dt_array,
        amplitudes_array,
        initial_wf,
        calc,
        max_iterations=500,
        target_population=0.9
    )
    
    print(f"\nRésultats de l'optimisation pour l = {target_l}:")
    print(f"Nombre de pulses dans la séquence : {len(best_sequence)}")
    for i, (amp, dt) in enumerate(best_sequence):
        print(f"Pulse {i+1}: amplitude = {amp} V/m, durée = {dt*1e6:.3f} µs")
    print(f"Population maximale atteinte : {best_pop:.6f}")
    
    # Vérification de la normalisation finale
    final_wf = pulse_evolution_final(best_sequence, initial_wf, calc)
    final_pop = np.sum(np.abs(final_wf) ** 2)
    print(f"\nVérification de la normalisation finale : {final_pop:.10f}")
    
    # Affichage des populations significatives par niveau l
    print("\nPopulations significatives par niveau l :")
    populations = np.abs(final_wf) ** 2
    for l_level in range(10, lmax):
        l_pop = 0.0
        for idx, state in enumerate(calc.basisStates):
            if state[1] == l_level:
                l_pop += populations[idx]
        if l_pop > 1e-10:
            print(f"l = {l_level}: {l_pop:.6f}")


