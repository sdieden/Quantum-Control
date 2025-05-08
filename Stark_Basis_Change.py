import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers.diophantine.diophantine import length


def atomic_to_stark(psi_atomic, calc, amplitude_index):
    """
    Transform a wavefunction from the atomic basis to the Stark basis.

    Args:
        psi_atomic (ndarray): Wavefunction in atomic basis
        calc (StarkMap): StarkMap object with pre-computed composition
        amplitude_index (int): Index for the electric field amplitude
                              in calc.composition

    Returns:
        ndarray: Wavefunction in Stark basis
    """
    psi_stark = np.zeros(len(psi_atomic), dtype=np.complex128)

    # For each Stark state (index in Stark basis)
    for stark_idx in range(len(psi_stark)):
        # Get the composition of this Stark state in terms of atomic states
        state_composition = calc.composition[amplitude_index][stark_idx]

        # Add contribution from each atomic state
        for component in state_composition:
            coeff = component[0]  # Coefficient of the atomic state
            atomic_idx = component[1]  # Index of the atomic state

            # Add this atomic state's contribution
            psi_stark[stark_idx] += coeff * psi_atomic[atomic_idx]

    return psi_stark


def stark_to_atomic(psi_stark, calc, amplitude_index):
    """
    Transform a wavefunction from the Stark basis to the atomic basis.

    Args:
        psi_stark (ndarray): Wavefunction in Stark basis
        calc (StarkMap): StarkMap object with pre-computed composition
        amplitude_index (int): Index for the electric field amplitude
                              in calc.composition

    Returns:
        ndarray: Wavefunction in atomic basis
    """
    psi_atomic = np.zeros(len(psi_stark), dtype=np.complex128)

    # For each Stark state
    for stark_idx, stark_amp in enumerate(psi_stark):
        if abs(stark_amp) < 1e-12:
            continue  # Skip negligible amplitudes for efficiency

        # Get composition of this Stark state
        state_composition = calc.composition[amplitude_index][stark_idx]

        # Add contribution to each atomic state
        for component in state_composition:
            coeff = component[0]  # Coefficient for atomic state
            atomic_idx = component[1]  # Index of atomic state

            # Add contribution to this atomic state
            psi_atomic[atomic_idx] += coeff * stark_amp

    return psi_atomic
def atomic_to_stark_corrected(psi_atomic, calc, amplitude_index):
    """
    Transform a wavefunction from the atomic basis to the Stark basis.

    Args:
        psi_atomic (ndarray): Wavefunction in atomic basis
        calc (StarkMap): StarkMap object with pre-computed composition
        amplitude_index (int): Index for the electric field amplitude
                              in calc.composition

    Returns:
        ndarray: Wavefunction in Stark basis
    """
    length = len(psi_atomic)
    psi_stark = np.zeros(length, dtype=np.complex128)
    map, = get_mapping(calc, amplitude_index)
    # For each Stark state (index in Stark basis)
    for stark_idx in range(length):
        # Get the composition of this Stark state in terms of atomic states
        state_composition = calc.composition[amplitude_index][stark_idx]

        # Add contribution from each atomic state
        for component in state_composition:
            coeff = component[0]  # Coefficient of the atomic state
            atomic_idx = component[1]  # Index of the atomic state

            # Add this atomic state's contribution
            psi_stark[stark_idx] += coeff * psi_atomic[atomic_idx]

    return psi_stark
def stark_to_atomic_corrected(psi_stark, calc, amplitude_index):
    """
    Transform a wavefunction from the Stark basis to the atomic basis.

    Args:
        psi_stark (ndarray): Wavefunction in Stark basis
        calc (StarkMap): StarkMap object with pre-computed composition
        amplitude_index (int): Index for the electric field amplitude
                              in calc.composition

    Returns:
        ndarray: Wavefunction in atomic basis
    """
    length = len(psi_stark)
    psi_atomic = np.zeros(length, dtype=np.complex128)
    dummy,map = get_mapping(calc, amplitude_index)
    # For each Stark state
    for stark_idx, stark_amp in enumerate(psi_stark):
        #if abs(stark_amp) < 1e-12:
        #    continue  # Skip negligible amplitudes for efficiency

        # Get composition of this Stark state
        state_composition = calc.composition[amplitude_index][stark_idx]

        # Add contribution to each atomic state
        for component in state_composition:
            coeff = component[0]  # Coefficient for atomic state
            atomic_idx = component[1]  # Index of atomic state

            # Add contribution to this atomic state
            psi_atomic[atomic_idx] += coeff * stark_amp

    return psi_atomic

def get_mapping(calc, amplitude_index):
    """
    Get the mapping between Stark basis and atomic basis.

    Args:
        calc (StarkMap): StarkMap object with pre-computed composition
        amplitude_index (int): Index for the electric field amplitude

    Returns:
        dict: Dictionary mapping Stark index to atomic index
        dict: Dictionary mapping atomic index to Stark index
    """
    stark_to_atomic_map = {}
    atomic_to_stark_map = {}

    # For zero field, there's a 1:1 mapping
    if abs(calc.eFieldList[amplitude_index]) < 1e-10:
        for stark_idx, comp in enumerate(calc.composition[amplitude_index]):
            if len(comp) == 1 and abs(comp[0][0] - 1.0) < 1e-10:
                atomic_idx = comp[0][1]
                stark_to_atomic_map[stark_idx] = atomic_idx
                atomic_to_stark_map[atomic_idx] = stark_idx

    return stark_to_atomic_map, atomic_to_stark_map


def test_transformation_identity(calc, amplitude_index=0):
    """
    For zero field, verify that the transformation is an identity
    (just reordering) by tracing the path of each basis state.

    Args:
        calc (StarkMap): StarkMap object
        amplitude_index (int): Index for the electric field amplitude
    """
    # Check if this is zero field
    if abs(calc.eFieldList[amplitude_index]) > 1e-10:
        print(f"This is not a zero field test (E = {calc.eFieldList[amplitude_index]} V/m)")
        return

    # Get the mapping
    stark_to_atomic_map, atomic_to_stark_map = get_mapping(calc, amplitude_index)

    print(f"Mapping for zero field (amplitude_index={amplitude_index}):")
    print("Stark index -> Atomic index")
    for stark_idx, atomic_idx in sorted(stark_to_atomic_map.items()):
        print(f"  {stark_idx} -> {atomic_idx}")

    # Test each atomic basis state
    print("\nTesting individual atomic states:")
    basis_dim = len(calc.basisStates)

    for i in range(basis_dim):
        # Create state with only one atomic basis component
        psi_atomic = np.zeros(basis_dim, dtype=np.complex128)
        psi_atomic[i] = 1.0

        # Transform to Stark basis
        psi_stark = atomic_to_stark(psi_atomic, calc, amplitude_index)

        # Find which Stark state has amplitude
        stark_indices = np.where(np.abs(psi_stark) > 0.9)[0]

        if len(stark_indices) == 1:
            stark_idx = stark_indices[0]
            print(f"  Atomic state {i} -> Stark state {stark_idx}")

            # Verify the mapping is consistent
            if i in atomic_to_stark_map and atomic_to_stark_map[i] != stark_idx:
                print(f"    WARNING: Mapping inconsistency for atomic state {i}")
        else:
            print(f"  Atomic state {i} -> Multiple or no Stark states {stark_indices}")


def test_basis_conversion(calc, amplitude_index=0, test_state_idx=None):
    """
    Test the basis conversion functions to verify they work correctly.

    Args:
        calc (StarkMap): A StarkMap object with pre-computed composition
        amplitude_index (int): Index for the electric field amplitude
        test_state_idx (int): Index of test state in atomic basis. If None, uses indexOfCoupledState

    Returns:
        float: The maximum difference between the original and the round-trip conversion
    """
    if test_state_idx is None:
        test_state_idx = calc.indexOfCoupledState

    # Create a test state in the atomic basis
    psi_atomic_original = np.zeros(len(calc.basisStates), dtype=np.complex128)
    psi_atomic_original[test_state_idx] = 1.0

    print(f"\nTesting with state {calc.basisStates[test_state_idx]} at E = {calc.eFieldList[amplitude_index]:.2f} V/m:")

    # Convert to Stark basis
    psi_stark = atomic_to_stark(psi_atomic_original, calc, amplitude_index)

    # Print the most significant components
    stark_probs = np.abs(psi_stark) ** 2
    sorted_indices = np.argsort(stark_probs)[::-1]
    significant_indices = sorted_indices[stark_probs[sorted_indices] > 0.01]

    print("  Significant components in Stark basis:")
    total_prob = 0
    for idx in significant_indices:
        print(f"    Stark state {idx}: Probability = {stark_probs[idx]:.6f}")
        total_prob += stark_probs[idx]
    print(f"  Total probability accounted for: {total_prob:.6f}")

    # Convert back to atomic basis
    psi_atomic_converted = stark_to_atomic(psi_stark, calc, amplitude_index)

    # Calculate the difference
    diff = np.abs(psi_atomic_original - psi_atomic_converted)
    max_diff = np.max(diff)
    max_diff_idx = np.argmax(diff)

    print(f"  Maximum difference after round-trip conversion: {max_diff:.6e} at index {max_diff_idx}")

    # Print the norm of each state to verify conservation
    norm_original = np.sum(np.abs(psi_atomic_original) ** 2)
    norm_stark = np.sum(np.abs(psi_stark) ** 2)
    norm_converted = np.sum(np.abs(psi_atomic_converted) ** 2)

    print(f"  Norm of original atomic state: {norm_original:.8f}")
    print(f"  Norm of Stark state: {norm_stark:.8f}")
    print(f"  Norm of converted atomic state: {norm_converted:.8f}")

    return max_diff


def visualize_transformation(calc, amplitude_index=1, test_state_idx=None):
    """
    Visualize the transformation between bases.

    Args:
        calc (StarkMap): A StarkMap object with pre-computed composition
        amplitude_index (int): Index for the electric field amplitude
        test_state_idx (int): Index of test state in atomic basis. If None, uses indexOfCoupledState
    """
    if test_state_idx is None:
        test_state_idx = calc.indexOfCoupledState

    # Create a test state in the atomic basis
    psi_atomic = np.zeros(len(calc.basisStates), dtype=np.complex128)
    psi_atomic[test_state_idx] = 1.0

    # Transform to Stark basis
    psi_stark = atomic_to_stark(psi_atomic, calc, amplitude_index)

    # Transform back to atomic basis
    psi_atomic_back = stark_to_atomic(psi_stark, calc, amplitude_index)

    # Plot the probability distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Atomic basis - original
    axes[0].bar(range(len(psi_atomic)), np.abs(psi_atomic) ** 2)
    axes[0].set_title("Original State in Atomic Basis")
    axes[0].set_xlabel("Atomic Basis State Index")
    axes[0].set_ylabel("Probability")

    # Stark basis
    axes[1].bar(range(len(psi_stark)), np.abs(psi_stark) ** 2)
    axes[1].set_title(f"State in Stark Basis (Field = {calc.eFieldList[amplitude_index]:.1f} V/m)")
    axes[1].set_xlabel("Stark Basis State Index")
    axes[1].set_ylabel("Probability")

    # Atomic basis - after round trip
    axes[2].bar(range(len(psi_atomic_back)), np.abs(psi_atomic_back) ** 2)
    axes[2].set_title("Recovered State in Atomic Basis")
    axes[2].set_xlabel("Atomic Basis State Index")
    axes[2].set_ylabel("Probability")

    plt.tight_layout()
    plt.show()


def print_basis_state_info(calc):
    """
    Print information about the basis states.

    Args:
        calc (StarkMap): A StarkMap object
    """
    print("Basis states:")
    for i, state in enumerate(calc.basisStates):
        print(f"  {i}: {state}")

    print(f"\nTarget state index: {calc.indexOfCoupledState}")
    print(f"Target state: {calc.basisStates[calc.indexOfCoupledState]}")


def analyze_composition(calc, amplitude_index):
    """
    Analyze the composition structure for a given field amplitude.

    Args:
        calc (StarkMap): A StarkMap object
        amplitude_index (int): Index for the electric field amplitude
    """
    print(f"\nComposition analysis for E = {calc.eFieldList[amplitude_index]:.2f} V/m:")

    composition = calc.composition[amplitude_index]

    # Check if this is a simple remapping (for zero field)
    is_simple_mapping = True
    for stark_idx, comp in enumerate(composition):
        if len(comp) != 1 or abs(comp[0][0] - 1.0) > 1e-6:
            is_simple_mapping = False
            break

    if is_simple_mapping:
        print("  Simple one-to-one mapping between bases (no mixing)")
        mapping = {}
        for stark_idx, comp in enumerate(composition):
            atomic_idx = comp[0][1]
            mapping[stark_idx] = atomic_idx
            print(f"    Stark state {stark_idx} = Atomic state {atomic_idx}")
    else:
        print("  Basis states are mixed - detailed analysis:")
        for stark_idx, comp in enumerate(composition):
            if len(comp) > 0:
                print(f"    Stark state {stark_idx} = ", end="")
                terms = []
                for coef, atomic_idx in comp:
                    if abs(coef) > 0.01:  # Show only significant contributions
                        terms.append(f"{coef:.4f}×|{calc.basisStates[atomic_idx]}⟩")
                print(" + ".join(terms))


def main():
    """
    Main function to demonstrate usage of the conversion functions.
    """
    # Import necessary libraries
    from arc import Calcium40, StarkMap

    # Initialize atom and StarkMap object
    atom = Calcium40()
    calc = StarkMap(atom)

    # Define basis
    n = 35
    l = 1
    j = 1
    mj = 0
    s = 0
    nmin = n - 1
    nmax = n + 1
    lmax = 4

    # Define basis states for the calculation
    calc.defineBasis(n, l, j, mj, nmin, nmax, lmax, s=s)

    # Print information about basis states
    print_basis_state_info(calc)

    # Diagonalize for a range of electric fields
    efields = np.linspace(0.0, 2000.0, 5)  # V/m
    calc.diagonalise(efields,upTo=-1)

    # Analyze the composition structure for zero field
    analyze_composition(calc, 0)

    # For non-zero field, also analyze the composition
    analyze_composition(calc, -1)  # Last field value

    # For zero field, verify the mapping
    test_transformation_identity(calc, 0)

    # Test the basis conversion for both zero and non-zero fields
    for i, efield in enumerate(efields):
        if i == 0 or i == len(efields) - 1:  # Test only first (zero) and last (maximum) field
            test_basis_conversion(calc, amplitude_index=i)

    # Visualize the transformation for a non-zero field
    visualize_transformation(calc, amplitude_index=-1)


if __name__ == '__main__':
    main()