

from arc import *
import numpy as np
import matplotlib.pyplot as plt
from sympy import S

## Stark Map

calc = StarkMap(Calcium40())

# Target state
n0 = 35
l0 = 1
j0 = 1
mj0 = 0
# Define max/min n values in basis
nmin = n0 - 3
nmax = n0 + 3
# Maximum value of l to include (l~20 gives good convergence for states with l<5)
lmax = 5

# Initialise Basis States for Solver : progressOutput=True gives verbose output
calc.defineBasis(n0, l0, j0, mj0, nmin, nmax, lmax, progressOutput=True, s=1)

Emin = 0.0  # Min E field (V/m)
Emax = 4.0e2  # Max E field (V/m)
N = 1001  # Number of Points

# Generate Stark Map
calc.diagonalise(np.linspace(Emin, Emax, N), progressOutput=True)
# Show Sark Map
#calc.plotLevelDiagram(progressOutput=True, units=0, highlightState=True)
#calc.ax.set_ylim(-101.25,-100 )
#calc.showPlot(interactive=False)
# return the ensemble of mixing states, for small E, we should get 1 for the initial state and 0 for the others


# we want that it shows the quantum numbers associated to every value calculated, to make it more readable, we can just
# show non 0 values.
a=calc.highlight[N-1]
MixingStates_Coef = []#np.array([])
print(type(MixingStates_Coef))
MixingStates_Term = []#np.array([])
for i in range(len(a)):
    if a[i] > 10**-3 :
        #print("{}\t{}".format(calc.basisStates[i],a[i]))
        print(calc.basisStates[i])

        MixingStates_Coef.append(a[i])
        MixingStates_Term.append(calc.basisStates[i])
    else :
        pass
print(MixingStates_Coef)
# still need to check that highlight and basisStates are arranged the same way.

# histogramm of the mixingstates

n_bins = len(MixingStates_Coef)
MixingStates_Coef = np.array(MixingStates_Coef)
MixingStates_Term = np.array(MixingStates_Term)
#plt.hist(MixingStates_Coef)
#plt.show()

# plt.style.use('_mpl-gallery')

# make data:
## separation of information for the term matrix
x = []
for i in range(len(MixingStates_Term)):
    x.append(i+0.5)
# transform list of quantum numbers into terms
# in general terms are written such so ^2S+1L_J


y = MixingStates_Coef

# plot
fig, ax = plt.subplots()

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.5)
# length of axes should be adjusted to the number of relevant states in the mixing state
ax.set(xlim=(0, len(MixingStates_Coef)),
       ylim=(0, 1))

plt.show()
Term(n0,l0,j0,s=1)