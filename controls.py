"""
Controls Module - Contains code for:
- generic controller class, which allows for different controller types
- joint by joint PID controller
- Cartesian space controller
- gravity and torque controllers


John Morrell, Jan 15 2023
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
from .dynamics import SerialArmDyn



