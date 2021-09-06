"""
Parallel ODE integration
"""

import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count


def parsolve_ivp(
        fun, 
        t_spans, 
        ics, 
        method='RK45', 
        t_eval=None, 
        dense_output=False, 
        events=None, 
        vectorized=False, 
        ps=None, 
        first_step=None, 
        max_step=np.inf, 
        rtol=1.e-12, 
        atol=1.e-12, 
        jac=None, 
        jac_sparsity=None, 
        lband=None, 
        uband=None, 
        min_step=0.0, 
        n_cpu=None):
    """Parallel solve_ivp wrapper.
    For common arguments with solve_ivp, see scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    Args:
        fun (callable): RHS of ODE system used by solve_ivp, with calling signature `fun(t,y)` or `fun(t,y,p)`
        t_spans (list): list of propagation times
        ics (list): list of initial conditions
        method (str, optional): integration method used by solve_ivp
        t_eval (array_like or None, optional): time-stamps where solution is computed
        dense_output (bool, optional): whether to compute a continuous solution.
        events (callable, or list of callables, optional): events to track.
        vectorized (bool, optional): whether fun is implemented in a vectorized fashion. Default is False.
        ps (list, optional): list of optional arguments passed to user-defined ODE function.
        first_step (bool): Initial step size. Default is None which means that the algorithm should choose
        max_step (float, optional): maximum allowed step-size, default is `np.inf`
        rtol (float or array_like, optional): relative tolerance
        atol (float or array_like, optional): absolute tolerance
        jac (array_like, sparse_matrix, callable or None, optional): Jacobian matrix
        jac_sparsity (array_like, sparse matrix or None, optional): sparsity of Jacobian matrix
        lband (int or None, optional): parameters defining the lower band of the Jacobian for the ‘LSODA’ method.
        uband (int or None, optional): parameters defining the upper band of the Jacobian for the ‘LSODA’ method.
        min_step (float, optional): minimum allowed step size for ‘LSODA’ method. By default min_step is zero.
        n_cpu (int or None, optional): number of CPU to be used by `multiprocessing.Pool`. If None, using `multiprocessing.cpu_count()`. 

    Returns:
        (list): list of bunch-objects, returned by `solve_ivp`
    """
    assert len(ics) == len(t_spans) == len(ps)

    if n_cpu is None:
        n_cpu = cpu_count()

    if ps is None:
        ps = [None for el in range(len(ics))]

    # construct array of combinations
    arguments = []
    for idx in range(len(ics)):
        # append tuple of arguments for i^th integration to list
        arguments.append(
                (fun, t_spans[idx], ics[idx], method, t_eval, dense_output, events, vectorized, ps[idx], first_step, max_step, rtol, atol, jac, jac_sparsity, lband, uband, min_step))

    # run function in parallel
    with Pool(processes=n_cpu) as pool:
        results = pool.starmap(_solve_ivp, tuple(arguments))
    return results



def _solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, first_step=None, max_step=np.inf, rtol=1.e-12, atol=1.e-12, jac=None, jac_sparsity=None, lband=None, uband=None, min_step=0.0):
    """Wrapper to solve_ivp for optional arguments handling"""
    return solve_ivp(fun, t_span, y0, method='RK45', t_eval=t_eval, dense_output=dense_output, events=events, vectorized=vectorized, args=args, first_step=first_step, max_step=max_step, rtol=rtol, atol=atol, jac=jac, jac_sparsity=jac_sparsity, lband=lband, uband=uband, min_step=min_step)



