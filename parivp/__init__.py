"""
Init file for parivp library, an extension to scipy for parallel ODE integration
"""


from .parallel_ivp import parsolve_ivp
from ._interp_sol import interp_sol