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

def SimpleDynArm(dh, linear_density=1.0, damping=0.1):
    mass = []
    r_com = []
    link_inertia = []
    joint_damping = []

def panda():
    # d theta a alpha
    dh = np.array([[0.333, 0, 0, -np.pi / 2], # 1-2
                   [0, 0, 0, np.pi / 2], # 2-3
                   [0.316, 0, 0.0825, np.pi / 2], # 3-4
                   [0, 0, -0.0825, -np.pi / 2], # 4-5
                   [0.384, 0, 0, np.pi / 2], # 5-6
                   [0, 0, 0.088, np.pi / 2], # 6-7
                   [0.107, 0, 0, 0]]) #7-F
    dh[:, 0]
    dh[:, 2]
    tool = transl([0, 0, 0.107])

    arm = SerialArm(dh=dh, tip=tool)
    return arm

