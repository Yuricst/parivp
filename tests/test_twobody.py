"""
Test for parivp module
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

from parivp import parsolve_ivp



def twobody(t,state,mu):
    """Two-body equation of motion"""
    dstate = np.zeros(6,)
    dstate[0:3] = state[3:6]
    dstate[3:6] = -mu/np.linalg.norm(state[0:3])**3 * state[0:3]
    return dstate


if __name__=="__main__":
    # number of integrations to perform
    n = 50
    
    # create list of final times
    t0 = 0.0
    tf = 10.0
    t_spans = [(t0,tf) for el in range(n)]

    # create list of initial conditions
    ics = []
    state0 = np.array([1.0, 0.0, 0.2, 0.0, 1.0, -0.02])
    sigR, sigV = 0.02, 0.01
    for idx in range(n):
        ics.append(state0 + np.concatenate((sigR*np.random.rand(3),sigV*np.random.rand(3))))

    # create list of argument to EOM
    ps = [(1.0,) for el in range(n)]

    # integrate in parallel
    n_cpu = 4
    t_eval = np.linspace(t0,tf,1000)
    respar = parsolve_ivp(fun=twobody, t_spans=t_spans, ics=ics, ps=ps, n_cpu=n_cpu, t_eval=t_eval)

    # plot result
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    for sol in respar:
        ax.plot(sol.y[0,:], sol.y[1,:], linewidth=0.5)
    ax.axis('equal')
    ax.set(xlabel="x", ylabel="y")
    plt.show()

