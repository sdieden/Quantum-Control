import numpy as np
import matplotlib.pyplot as plt

import sys, os
import cmath as cm
import pathlib
from numpy.ma.core import shape

from PulseTime import initial_psi

newModPath=pathlib.Path(os.path.dirname(os.path.abspath(__file__)),'NewModule')
sys.path.insert(0, str(newModPath))
from arc_sam import *  # Import ARC (Alkali Rydberg Calculator)

atom = Calcium40()
calc = StarkMap(atom)
hbar = 1
n = 35
l = 3
j = 3
mj = 0
s = 0
nmin = n - 1
nmax = n + 1
lmax = nmax - 1
calc.defineBasis(n,l,j,mj,nmin,nmax,lmax,s=s)
def U(en,dt):
    """
    arg : en : is the energy of the level considered.
    arg : dt : is the time step.
    result : time evolution operator of the level considered, it's a phase on the WaveFunction.

    in this code, the U operator will be used in two cases.
    First evolution when there is no field applied -> U must takes the energy of the atomic level
    When the field is applied -> U must take the energy of the considered Stark level

    """
    u = cm.exp(0 - 1j*en*dt/hbar)
    return u
def apply_pulse(psi_in,pulse):

    psi_out = np.zeros(len(psi_in))
    psi_out = np.complex128(psi_out)
    if pulse.stark == True:
        for state in psi_in:
            for substate in range(len(calc.composition[pulse.index_of_amplitude()][state])):
                coef = psi_in[substate]*calc.composition[pulse.index_of_amplitude()][state][substate]
                en = calc.y[pulse.index_of_amplitude()][substate]
                coef *= U(en,pulse.duration)
                psi_out[calc.composition[pulse.index_of_amplitude()][state][1]] += coef


    elif pulse.stark == False:
        for state in psi_in:
            coef = psi_in[state]
            en = calc.y[0][state]
            coef *= U(en,pulse.duration)
            psi_out[state] += coef

    return psi_out
def pulse_evolution(pulseList):
    """
    :arg pulseList: list of 2 long lists. The first argument is the Electric field
    and the second argument is the duration of the field
    ex : pulseList = [(30,5)(0,10)(20,1)]
    This would be a 30V/m pulse during 5 seconds followed by a null field for 10 seconds and finally a 20V/m pulse during 10 seconds.

    """

    for pulse in pulseList:
        if len(pulse) != 2:
            raise ValueError(f'pulse ',pulse ,' in pulseList must have 2 elements, has',{len(pulse)})
        if pulse[1] <= 0:
            raise ValueError(f'duration is negative',{pulse[1]},'must be positive')

    ## initialize the pulses

    for i in range(len(pulseList)):
        Pulse(pulseList[i][0],pulseList[i][1])
        """index = 0
        while sorted_wo_doublon[index] != pulseList[i][0]:
            index += 1
        elif sorted_wo_doublon[index] == pulseList[i][0]:
            Pulse.index = index
        """

# initial state
    initial_psi = np.zeros(len(calc.basisStates))
    initial_psi = np.complex128(initial_psi)
    initial_psi[calc.index_new_basis[calc.indexOfCoupledState]] = 1

    ##
    psi=[initial_psi]

    for pulse in Pulse.liste_pulse:
        idx=pulse.index_of_amplitude()
        psi.append(apply_pulse(psi[-1], pulse))
    return psi

list = [(30,5),(0,10),(20,1)]

pulse_evolution(list)
