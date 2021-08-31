"""
Parallel ODE integration
"""

import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count


def par_solve_ivp(fun, tfs, ics, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, ps=None, rtol=1.e-12, atol=1.e-12, jac=None, jac_sparsity=None, lband=None, uband=None, min_step=0.0):
    """Parallel solve_ivp wrapper

    Args:
        fun (callable): RHS of ODE system used by solve_ivp
        tfs (list): list of propagation times
        ics (list): list of initial conditions
        method (str): integration method used by solve_ivp
        t_eval (array_like or None): time-stamps where solution is computed

    Returns:
        (list): list of bunch objects
    """
    assert len(ics) == len(tfs) == len(ps)

    if n_cpu is None:
        n_cpu = cpu_count()

    if ps is None:
        ps = [None for el in range(len(ics))]

    # construct array of combinations
    arguments = []
    for idx in range(len(ics)):
        # unpack parameters
        mu = ps[idx][0]
        # append tuple of arguments for i^th integration to list
        arguments.append(
                (fun, tfs[idx], ics[idx], method, t_eval, dense_output, events, vectorized, ps[idx], options, first_step, max_step, rtol, atol, jac, jac_sparsity, lband, uband, min_step)
            )

    # run function in parallel
    with Pool(processes=n_cpu) as pool:
        results = pool.starmap(solve_ivp, tuple(arguments))
    return results