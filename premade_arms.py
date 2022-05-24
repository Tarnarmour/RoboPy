from .kinematics import *
from .dynamics import *
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

def PlanarDyn(n=3, L=1, joint_damping=None):

    if isinstance(L, (list, np.ndarray, tuple)):
        if len(L) == n:
            pass
        else:
            print("Warning: n != length of L for Planar Dynamic Arm Constructor!")
            return None
    else:
        L = [L] * n

    mass = L
    r_coms = []
    link_inertias = []
    dh = []
    for ll in L:
        r_coms.append(np.array([-ll / 2, 0, 0]))
        In = np.array([[0, 0, 0],
                       [0, ll**3/12, 0],
                       [0, 0, ll**3/12]])
        link_inertias.append(In)
        dh.append([0, 0, ll, 0])

    arm = SerialArmDyn(dh, link_inertia=link_inertias, mass=mass, r_com=r_coms, joint_damping=joint_damping)
    return arm