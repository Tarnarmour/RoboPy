"""
simulation module - contains code for:
- Creating dynamic system representations of robot arms
- Executing simulation using RK4, SciPy odeint, etc
- Generating linearized models of arms
- Input respones, etc.

John Morrell, May 2 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from .dynamics import SerialArmDyn

