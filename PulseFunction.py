import numpy as np
import matplotlib.pyplot as plt

import sys, os
import cmath as cm
import pathlib
from numpy.ma.core import shape



newModPath=pathlib.Path(os.path.dirname(os.path.abspath(__file__)),'NewModule')
sys.path.insert(0, str(newModPath))
from arc_sam import *  # Import ARC (Alkali Rydberg Calculator)

atom = Calcium40()
calc = StarkMap(atom)

n = 35
l = 3
j = 3
mj = 0
s = 0
nmin = n - 1
nmax = n + 1
lmax = 5

"""
Rappel des unités atomiques  : 
action : hbar = 1 , hbar = 1.1*10^-34 J.s
energie : hartree, 1 Hartree = 4.359*10^-18 J
temps : 1 t.ua = 2.418*10^-17 s
champ électrique : 1 ce.ua = 5.14*10^11V/m

"""
conv_en = 4.359e18
conv_t = 2.418e17
conv_ce = 5.14e-11
hbar = 1
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

    psi_out = np.zeros(len(psi_in),dtype=np.complex128)

    if pulse.stark == True:
        for state in range(len(psi_in)):
            coef_state = psi_in[state]
            """if len(calc.composition[pulse.index_of_amplitude()][state]) == 1 :
                coef_substate = calc.composition[pulse.index_of_amplitude()][state][0][0]
                en = calc.y[pulse.index_of_amplitude()][state]
                value = coef_state*coef_substate
                value *= U(en,pulse.duration)
                psi_out[calc.composition[pulse.index_of_amplitude()][state][0][1]] += value
            else:
                for substate in range(len(calc.composition[pulse.index_of_amplitude()][state])):
                    coef = psi_in[substate]*calc.composition[pulse.index_of_amplitude()][state][substate][0]
                    en = calc.y[pulse.index_of_amplitude()][substate]
                    coef *= U(en,pulse.duration)
                    psi_out[calc.composition[pulse.index_of_amplitude()][state][0][1]] += coef
            """
            for substate in range(len(calc.composition[pulse.index_of_amplitude()][state])):
                coef_substate = calc.composition[pulse.index_of_amplitude()][state][substate][0]
                en = calc.y[pulse.index_of_amplitude()][substate] * conv_en
                value = coef_state*coef_substate
                value *= U(en,pulse.duration*conv_t)
                psi_out[calc.composition[pulse.index_of_amplitude()][state][substate][1]] += value

    elif pulse.stark == False:
        for state in range(len(psi_in)):
            coef = psi_in[state]
            en = calc.y[0][state]*conv_en
            coef *= U(en,pulse.duration*conv_t)
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
    initial_psi = np.zeros(len(calc.basisStates),dtype=np.complex128)

    #initial_psi[calc.index_new_basis[calc.indexOfCoupledState]] = 1
    initial_psi[calc.indexOfCoupledState] = 1
#
    psi=[initial_psi]

    for pulse in Pulse.liste_pulse:
        idx=pulse.index_of_amplitude()
    calc.diagonalise(sorted(tuple(pulse.amplitudes_list)),upTo=-1)
    for i in range(len(Pulse.liste_pulse)):
        psi.append(apply_pulse(psi[-1], pulse))
    
    return psi

def total_population(psi_array):
    abs_array = np.abs(psi_array)
    population = np.zeros(len(psi_array))
    for k in range(len(abs_array)):
        population[k] = np.sum(abs_array[k] ** 2)

    return population

list = [(30,5),(0,10),(20,1)]

a=pulse_evolution(list)
peuple = total_population(a)
print('population=',peuple,'final state=',a[-1])


