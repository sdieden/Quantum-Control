import sys,os
sys.path.insert(0,'/Users/sam/PycharmProjects/Quantum-Control/NewModule')
from arc import *
import numpy as np
import matplotlib.pyplot as plt
calc = StarkMap(Calcium40())
# Target state
n0 = 35
l0 = 3
j0 = 3
mj0 = 0
s0 = 0
# state0 = np.array([n0, l0, j0, mj0])
# Define max/min n values in basis
nmin = n0 - 5
nmax = n0 + 5
# Maximum value of l to include (l~20 gives good convergence for states with l<5)
lmax = nmax-1 #34
# Initialise Basis States
calc.defineBasis(n0, l0, j0, mj0, nmin, nmax, lmax, progressOutput=True,debugOutput=False, s=s0)

Emin = 0.0  # Min E field (V/m)
Emax = 25.0e2  # Max E field (V/m)
N = 1001  # Number of Points
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

# Generate Stark Map
calc.diagonalise(np.linspace(Emin, Emax, N), progressOutput=True)

def Term(n,l,j,s):
    first_quantum_number = n
    Spin_part = 2*s+1
    l = int(l)
    orbital_term = ["S", "P", "D", "F", "G", "H", "I", "K", "L", "M", "N", "O", "Q", "R", "T", "U", "V", "W", "X", "Y", "Z"]
    if l <= len(orbital_term)-1:
       orbital = orbital_term[l]
    else :
        orbital = int(l)
    term = "{N}$^{S}{L}_{J}$"
    return term.format(N=int(first_quantum_number),J = int(j),L = orbital,S=int(Spin_part))

a = calc.highlight[N-1] #
b = calc.egvector[N-1]

print(a)
print(b)

MixingStates_Coef = []  #list : MixingStates_Coef[i] gives the % of presence of the i-th state in the Stark level for specified value of electric Field E

MixingStates_Term = []  #list : MixingStates_Term[i] gives the quantum numbers of the i-th state

MixingStates_Egvector = [] #list :

for i in range(len(a)):
    if a[calc.index_new_basis[i]] > 10 ** -3:
        MixingStates_Coef.append(a[calc.index_new_basis[i]])
        MixingStates_Term.append(calc.basisStates[i])
        MixingStates_Egvector.append(b[calc.index_new_basis[i]])

    else:
        pass

# histogramm of the mixingstates

n_bins = len(MixingStates_Coef)
#MixingStates_Coef = np.array(MixingStates_Coef)
MixingStates_Egvector = np.array(MixingStates_Egvector)
MixingStates_Term = np.array(MixingStates_Term)
MixingStates_Term_new = []
#
for i in range(len(MixingStates_Term)):
    MixingStates_Term_new.append(
        Term(MixingStates_Term[i][0], MixingStates_Term[i][1], MixingStates_Term[i][2], MixingStates_Term[i][3]))

x = MixingStates_Term_new

# Plot les % de mixage

y = MixingStates_Coef
# plot
fig, ax = plt.subplots()
ax.bar(x, y, width=1, edgecolor="white", linewidth=0.5)
multiplier = 0
ax.set(xlim=(-0.5, len(MixingStates_Coef) - 0.5),
       ylim=(0 , 1))

for coef in MixingStates_Coef:
    offset = multiplier
    rects = ax.bar(offset, coef, width=1)
    ax.bar_label(rects, labels=[f"{coef:.3f}" for coef in rects.datavalues], padding=5, fontsize=8, rotation='vertical')
    multiplier += 1

ax.set_ylabel('coefficient du mix')
ax.set_title(r'Mixing states of Stark Levels in $^{40}Ca$')
ax.tick_params(axis='x', pad=0)
plt.xticks(rotation=90)
plt.show()

"""
Plots les coefficients
y = MixingStates_Egvector

# plot
fig, ax = plt.subplots()
ax.bar(x, y, width=1, edgecolor="white", linewidth=0.5)
multiplier = 0

for egvector in MixingStates_Egvector:
    offset = multiplier
    rects = ax.bar(offset, egvector, width=1)
    ax.bar_label(rects, labels=[f"{egvector:.3f}" for egvector in rects.datavalues], padding=5, fontsize=8, rotation='vertical')
    multiplier += 1
#length of axes should be adjusted to the number of relevant states in the mixing state

ax.set(xlim=(-0.5, len(MixingStates_Egvector) - 0.5),
       ylim=(-1, 1))
ax.set_ylabel('coefficient du mix')
ax.set_title(r'Mixing states of Stark Levels in $^{40}Ca$')
ax.tick_params(axis='x', pad=0)
plt.xticks(rotation=90)
plt.show()
"""