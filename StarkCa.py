import numpy as np
import matplotlib.pyplot as plt

import sys, os

sys.path.insert(0, '/Users/sam/PycharmProjects/Quantum-Control/NewModule')
from arc import *  # Import ARC (Alkali Rydberg Calculator)

## Stark Map

calc = StarkMap(Calcium40())
#x%matplotlib qt

# Target state
n0 = 35
l0 = 3
j0 = 3
mj0 = 0
s0 = 0
 # Define max/min n values in basis
nmin = n0 - 3
nmax = n0 + 3


# Maximum value of l to include (l~20 gives good convergence for states with l<5)
lmax = nmax-1 #nmax-1

# Initialise Basis States for Solver : progressOutput=True gives verbose output
calc.defineBasis(n0, l0, j0, mj0, nmin, nmax, lmax, progressOutput=True, s=s0)

Emin = 0.0  # Min E field (V/m)
Emax = 25.0e2  # Max E field (V/m)
N = 1001  # Number of Points

# Generate Stark Map
calc.diagonalise(np.linspace(Emin, Emax, N), progressOutput=True,debugOutput=True)
binary_matrix_diagonal = 'bi_diagonal.npy'
np.save(binary_matrix_diagonal, calc.mat1)
binary_matrix_offdiag = 'bi_offdiag.npy'
np.save(binary_matrix_offdiag, calc.mat2)
exportdata = 'CA_EXPORTDATA.csv'
calc.exportData(exportdata)


# Show Sark Map
calc.plotLevelDiagram(progressOutput=True, units=0, highlightState=True)
calc.ax.set_ylim(-110, -85)
calc.showPlot(interactive=False)


# return the ensemble of mixing states, for small E, we should get 1 for the initial state and 0 for the others
# we want that it shows the quantum numbers associated to every value calculated, to make it more readable, we can just
# show non 0 values.
a = calc.highlight[N-1]

def Term(n, l, j, s):
    # transform list of quantum numbers into terms
    # in general terms are written such so ^2S+1L_J
    first_quantum_number = n
    Spin_part = 2 * s + 1

    if l == 0:
        orbital = "S"
    elif l == 1:
        orbital = "P"
    elif l == 2:
        orbital = "D"
    elif l == 3:
        orbital = "F"
    elif l == 4:
        orbital = "G"
    elif l == 5:
        orbital = "H"
    elif l == 6:
        orbital = "I"
    elif l == 7:
        orbital = "K"
    else:
        orbital = int(l)

    term = "{N}$^{S}{L}_{J}$"
    return term.format(N=int(first_quantum_number), J=int(j), L=orbital, S=int(Spin_part))
def index_efield(Efield, round = True):

    """
    :param Efield: value of E field in V/m
    :return: (closest) index
    """

    i = Efield*N/(Emax-Emin)
    if round == True:
        i = np.round(i)
        i = i.astype(int)
    return i

#________corrected way

MixingStates_Coef = []  #list

MixingStates_Term = []  #list

for i in range(len(a)):
    if a[calc.index_new_basis[i]] > 10 ** -3:
        MixingStates_Coef.append(a[calc.index_new_basis[i]])
        MixingStates_Term.append(calc.basisStates[i])

    else:
        pass
# histogramm of the mixingstates

n_bins = len(MixingStates_Coef)
MixingStates_Coef = np.array(MixingStates_Coef)
MixingStates_Term = np.array(MixingStates_Term)
MixingStates_Term_new = []
#
for i in range(len(MixingStates_Term)):
    MixingStates_Term_new.append(
        Term(MixingStates_Term[i][0], MixingStates_Term[i][1], MixingStates_Term[i][2], MixingStates_Term[i][3]))

x = MixingStates_Term_new

y = MixingStates_Coef

# plot
fig, ax = plt.subplots()
ax.bar(x, y, width=1, edgecolor="white", linewidth=0.5)
multiplier = 0
for coef in MixingStates_Coef:
    offset = multiplier
    rects = ax.bar(offset, coef, width=1)
    ax.bar_label(rects, labels=[f"{coef:.3f}" for coef in rects.datavalues], padding=5, fontsize=8, rotation='vertical')
    multiplier += 1
#length of axes should be adjusted to the number of relevant states in the mixing state
ax.set(xlim=(-0.5, len(MixingStates_Coef) - 0.5),
       ylim=(0, 1))
ax.set_ylabel('coefficient du mix')
ax.set_title(r'Mixing states of Stark Levels in $^{40}Ca$')
ax.tick_params(axis='x', pad=0)
plt.xticks(rotation=90)
plt.show()
