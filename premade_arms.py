from .kinematics import *
from .utility import *
from .transforms import *

"""
Human arm: from M. Z. A. Mieee, “Human Arm Inverse Kinematic Solution Based Geometric Relations and Optimization Algorithm.” 
"""

def HumanArm(L1=0.3, L2=0.285):
    dh = [[0, 0, 0, pi/2],
                      [0, pi/2, 0, pi/2],
                      [0, 0, 0, -pi/2],
                      [0, 0, L1, pi/2],
                      [0, 0, L2, -pi/2],
                      [0, pi/2, 0, -pi/2],
                      [0, 0, 0, -pi/2]]
    return SerialArm(dh, base=se3(roty(pi/2)), tip=se3(rotz(pi/2)))