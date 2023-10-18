"""
Interpolate solution object
"""

import numpy as np
from scipy.interpolate import interp1d

def interp_sol(sol, teval, full_output=False, kind="cubic"):
	"""Evaluate ODE solution at time-stamps via interpolation of each state.
	Note that all elements of teval must be within the times existing in sol.t (may be equal). 

	Args:
		sol (Bunch object): object returned by solve_ivp
		teval (real or array-like): epochs to evaluate solution
		full_output (bool): whether to also return list of interpolation objects
		kind (str): kind of interpolation for scipy.interpolate.interp1d

	Returns:
		(np.array or tuple): interpolated state(s), and optionally list of interpolation objects for each state
	"""
    # construct interpolation object for each state
    nx, nt = sol.y.shape
    interp_list = []
    for i in range(nx):
        interp_list.append(interp1d(sol.t, sol.y[i,:], kind=kind))
    # evaluate interpolated object at time stamps
    states_interpolated = []
    for i in range(nx):
        states_interpolated.append(interp_list[i](teval))
    states_interpolated = np.array(states_interpolated)
    if full_output:
    	return states_interpolated, interp_list
	else:
    	return states_interpolated