

from arc import *
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from holoviews.plotting.bokeh.styles import font_size
from sympy import S
from twisted.protocols.amp import Decimal

## Stark Map

calc = StarkMap(Calcium40())

# Target state
n0 = 35
l0 = 3
j0 = 3
mj0 = 0
# Define max/min n values in basis
nmin = n0 - 5
nmax = n0 + 5
# Maximum value of l to include (l~20 gives good convergence for states with l<5)
lmax = 34

# Initialise Basis States for Solver : progressOutput=True gives verbose output
calc.defineBasis(n0, l0, j0, mj0, nmin, nmax, lmax, progressOutput=True, s=0)

Emin = 0.0  # Min E field (V/m)
Emax = 8.0e2  # Max E field (V/m)
N = 1001  # Number of Points

# Generate Stark Map
calc.diagonalise(np.linspace(Emin, Emax, N), progressOutput=True)
# Show Sark Map
calc.plotLevelDiagram(progressOutput=True, units=0, highlightState=True)
calc.ax.set_ylim(-105,-70 )
calc.showPlot(interactive=False)

#####
#Other method to calcul mixing states composition using directly made in ARC

# return the ensemble of mixing states, for small E, we should get 1 for the initial state and 0 for the others
# we want that it shows the quantum numbers associated to every value calculated, to make it more readable, we can just
# show non 0 values.
a=calc.highlight[N-1]
print('a=',a)
MixingStates_Coef = []#list

MixingStates_Term = []#list
for i in range(len(a)):
    if calc.basisStates[i] == [35, 3, 3, 0] :
        print('Il est ici!!!!',i)
    else :
        pass
for i in range(len(a)):
    if a[i] > 10**-3 :
        print('i=',i)
        #print("{}\t{}".format(calc.basisStates[i],a[i]))
        print(len(a))
        print(len(calc.basisStates))
        print(calc.basisStates)

        MixingStates_Coef.append(a[i])
        MixingStates_Term.append(calc.basisStates[i])
        #print(calc.basisStates[i],'i')
        #print(calc.basisStates[i-1],'i-1')
        #print(calc.basisStates[i + 1], 'i+1')
    else :
        pass
print('coef=',MixingStates_Coef)
# still need to check that highlight and basisStates are arranged the same way.

# histogramm of the mixingstates

n_bins = len(MixingStates_Coef)
MixingStates_Coef = np.array(MixingStates_Coef)
MixingStates_Term = np.array(MixingStates_Term)
#for i in range(len(MixingStates_Coef)):
#    MixingStates_Coef[i] = '%.2E' % Decimal(MixingStates_Coef[i])
# make data:
## separation of information for the term matrix
#x = []
#for i in range(len(MixingStates_Term)):
#    x.append(i+0.5)
# transform list of quantum numbers into terms
# in general terms are written such so ^2S+1L_J
def Term(n,l,j,s):
    first_quantum_number = n
    Spin_part = 2*s+1

    if l==0 :
        orbital = "S"
    elif l==1:
        orbital = "P"
    elif l==2:
        orbital = "D"
    elif l==3:
        orbital ="F"
    elif l==4:
        orbital = "G"
    elif l==5:
        orbital = "H"
    elif l==6:
        orbital = "I"
    elif l==7:
        orbital = "K"
    else :
        orbital = int(l)

    term = "{N}$^{S}{L}_{J}$"
    return term.format(N=int(first_quantum_number),J = int(j),L = orbital,S=int(Spin_part))
# now we rewrite the list of list as a list of terms
MixingStates_Term_new = []
print(MixingStates_Term)
for i in range(len(MixingStates_Term)):

    MixingStates_Term_new.append(Term(MixingStates_Term[i][0],MixingStates_Term[i][1],MixingStates_Term[i][2],MixingStates_Term[i][3]))

x = MixingStates_Term_new
print(x)
y = MixingStates_Coef

# plot
fig, ax = plt.subplots()
ax.bar(x, y, width=1, edgecolor="white", linewidth=0.5)
multiplier = 0
for coef in MixingStates_Coef:
    offset = multiplier
    rects = ax.bar(offset,coef ,width=1)
    ax.bar_label(rects,labels=[f"{coef:.3f}" for coef in rects.datavalues], padding=5, fontsize=8, rotation='vertical')
    multiplier+=1
# length of axes should be adjusted to the number of relevant states in the mixing state
ax.set(xlim=(-0.5, len(MixingStates_Coef)-0.5),
       ylim=(0, 1))
ax.set_ylabel('coefficient du mix')
ax.set_title(r'Mixing states of Stark Levels in $^{40}Ca$')
ax.tick_params(axis='x',pad=0)
plt.xticks(font_size=8,rotation = 90)
plt.show()
print(Term(n0,l0,j0,s=1))
