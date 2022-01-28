"""
Dynamics Module - Contains code for:
- Dynamic SerialArm class
- RNE Algorithm
- Euler - Lagrange formulation

John Morrell, Jan 28 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
import Kinematics as kin


class SerialArmDyn(kin.SerialArm):
    """
    class representing a serial linkage arm with dynamic
    !!! finish me !!!
    """

    def __init__(self, dh, jt=None, base=eye, tip=eye, joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None):
        super(SerialArmDyn).__init__(dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia

    def RNE(self, q, qd, qdd, Wext, omega_base, alpha_base, acc_base):
