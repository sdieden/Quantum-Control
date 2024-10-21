from arc import *
import numpy as np

## Stark Map

calc = StarkMap(Calcium40())

# Target state
n0 = 35
l0 = 1
j0 = 2
mj0 = 0
# Define max/min n values in basis
nmin = n0 - 7
nmax = n0 + 7
# Maximum value of l to include (l~20 gives good convergence for states with l<5)
lmax = 34

# Initialise Basis States for Solver : progressOutput=True gives verbose output
calc.defineBasis(n0, l0, j0, mj0, nmin, nmax, lmax, progressOutput=True, s=1)

Emin = 0.0  # Min E field (V/m)
Emax = 8.0e2  # Max E field (V/m)
N = 1001  # Number of Points

# Generate Stark Map
calc.diagonalise(np.linspace(Emin, Emax, N), progressOutput=True)
# Show Sark Map
calc.plotLevelDiagram(progressOutput=True, units=0, highlightState=True)
calc.ax.set_ylim(-101.25,-100 )
calc.showPlot(interactive=True)
# Return Polarizability of target state
print(
    "%.5f MHz cm^2 / V^2 "
    % calc.getPolarizability(showPlot=True, minStateContribution=0.9)
)