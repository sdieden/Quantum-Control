from pygments import highlight

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
# Define max/min n values in basis
nmin = n0 - 5
nmax = n0 + 5
# Maximum value of l to include (l~20 gives good convergence for states with l<5)
lmax = nmax-1 #34

# Initialise Basis States for Solver : progressOutput=True gives verbose output
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

print(calc.highlight[index_efield(250, True)])
#print(calc.highlight[N-1])
#fullStarkMatrix = calc.mat1+calc.mat2*eField
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
basisStates = calc.basisStates
a=calc.highlight[500]#[index_efield(Emax)-1]
print(basisStates[calc.indexOfCoupledState],a[calc.indexOfCoupledState])

"""
MixingStates_Coef = []#list

MixingStates_Term = []#list
for i in range(len(a)):
    if a[i] > 10 ** -3:
        # print('i=',i)
        # print("{}\t{}".format(calc.basisStates[i],a[i]))
        # print(len(a))
        # print(len(calc.basisStates))
        # print(calc.basisStates)

        MixingStates_Coef.append(a[i])
        MixingStates_Term.append(calc.basisStates[i-1])
        # print(calc.basisStates[i],'i')
        # print(calc.basisStates[i-1],'i-1')
        # print(calc.basisStates[i + 1], 'i+1')
    else:
        pass
#begin with lowest n and l, raising by l+1, until lmax then go to n+1 with l = 0
MixingStates_Term_new = []

for i in range(len(MixingStates_Term)):
    print(i)
    MixingStates_Term_new.append(Term(MixingStates_Term[i][0], MixingStates_Term[i][1], MixingStates_Term[i][2], MixingStates_Term[i][3]))
x = MixingStates_Term_new
y = MixingStates_Coef#calc.highlight[index_efield(Emax)-1]
# plot
fig, ax = plt.subplots()
ax.bar(x, y, width=1, edgecolor="white", linewidth=0.5)
multiplier = 0
for coef in y:
    offset = multiplier
    rects = ax.bar(offset,coef ,width=1)
    ax.bar_label(rects,labels=[f"{coef:.3f}" for coef in rects.datavalues], padding=5, fontsize=8, rotation='vertical')
    multiplier+=1
#length of axes should be adjusted to the number of relevant states in the mixing state
ax.set(xlim=(-0.5, len(y)-0.5),
       ylim=(0, 1))
ax.set_ylabel('coefficient du mix')
ax.set_title(r'Mixing states of Stark Levels in $^{40}Ca$')
ax.tick_params(axis='x',pad=0)
plt.xticks(rotation = 90)
plt.show()
#print(calc.highlight[0])

"""