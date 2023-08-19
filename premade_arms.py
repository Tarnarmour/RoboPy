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
                       [0, ll**2/12, 0],
                       [0, 0, ll**2/12]])
        link_inertias.append(In)
        dh.append([0, 0, ll, 0])

    arm = SerialArmDyn(dh, link_inertia=link_inertias, mass=mass, r_com=r_coms, joint_damping=joint_damping)
    return arm


def SimpleDynArm(dh, jt=None, linear_density=1.0, motor_mass=0.0, damping=0.0, **kwargs):
    """A generic constructor that takes in DH parameters and joint types and makes an assumption of linear density,
    then calculates the dynamic parameters for each link assuming slender links."""
    mass = []
    r_com = []
    link_inertia = []
    joint_damping = []

    n = len(dh)

    if isinstance(motor_mass, (int, float)):
        motor_mass = [motor_mass] * n

    if jt is None:
        jt = ['r'] * n

    """
    The general idea is this: we can find the length of each link by assuming that the link forms a straight line from
    frame i - 1 to frame i with linear density. Then the mass will be simple to find and the rotational inertia can be
    found by finding the required shift and rotation that transforms the slender body to the right orientation
    """

    for i in range(n):
        d = dh[i][0]
        theta = dh[i][1]
        a = dh[i][2]
        alpha = dh[i][3]

        cth = np.cos(theta)
        sth = np.sin(theta)
        cal = np.cos(alpha)
        sal = np.sin(alpha)

        Ap2c = np.array([[cth, -sth * cal, sth * sal, a * cth],
                          [sth, cth * cal, -cth * sal, a * sth],
                          [0, sal, cal, d],
                          [0, 0, 0, 1]])
        Ac2p = inv(Ap2c)

        rc2p = Ac2p[0:3, 3]  # vector from frame i to frame i-1
        r_lcom = rc2p * 0.5  # vector to COM of link
        r_m = rc2p  # vector to motor
        ll = np.linalg.norm(rc2p)
        lm = ll * linear_density
        mm = motor_mass[i]

        com = (r_lcom * lm + r_m * mm) / (mm + lm)  # find COM by weighted average of link COM and motor COM

        I_link_body = np.array([[0, 0, 0],
                                [0, ll**2 / 12 * lm, 0],
                                [0, 0, ll**2 / 12 * lm]])  # inertia tensor of link at the link COM in the link body frame

        i_b = rc2p / ll if not np.isclose(ll, 0.0) else Ac2p[0:3, 0]  # link body frame i unit vector is aligned with the vector from i to i-1 frame (along link length)
        k_b = np.cross(i_b, np.array([1, 0, 0]))  # find z unit vector by crossing i_b and i_i
        if np.linalg.norm(k_b) == 0:  # unless i_b and i_i are already parallel, then cross with j_i vector
            k_b = np.cross(i_b, np.array([0, 1, 0]))
        k_b = k_b / np.linalg.norm(k_b)
        j_b = np.cross(k_b, i_b)
        R_b = np.vstack([i_b, j_b, k_b]).T
        I_link_i = R_b.T @ I_link_body @ R_b
        r_link2com = com - r_lcom
        r_motor2com = com - r_m
        I_combined = I_link_i + lm * (r_link2com @ r_link2com * np.eye(3) - np.outer(r_link2com, r_link2com)) \
                     + mm * (r_motor2com @ r_motor2com * np.eye(3) - np.outer(r_motor2com, r_motor2com))

        mass.append(lm + mm)
        r_com.append(com)
        link_inertia.append(I_combined)
        joint_damping.append(damping)

    return SerialArmDyn(dh=dh,
                        jt=jt,
                        mass=mass,
                        r_com=r_com,
                        link_inertia=link_inertia,
                        joint_damping=joint_damping,
                        **kwargs)


def Panda(s=1):
    """Franka emika Panda robot"""
    # d theta a alpha
    dh = np.array([[0.333, 0, 0, -np.pi / 2], # 1-2
                   [0, 0, 0, np.pi / 2], # 2-3
                   [0.316, 0, 0.0825, np.pi / 2], # 3-4
                   [0, 0, -0.0825, -np.pi / 2], # 4-5
                   [0.384, 0, 0, np.pi / 2], # 5-6
                   [0, 0, 0.088, np.pi / 2], # 6-7
                   [0.107, 0, 0, 0]]) #7-F
    dh[:, 0] = dh[:, 0] * s
    dh[:, 2] = dh[:, 2] * s

    jt = ['r'] * 7

    base = se3()
    tip = se3(R=rotz(-np.pi / 4), p=[0, 0, 0.107])

    joint_limits = np.array([[-2.8973, 2.8973],
                             [-1.7628, 1.7628],
                             [-2.8973, 2.8973],
                             [-3.0718, -0.0698],
                             [-2.8973, 2.8973],
                             [-0.0175, 3.7525],
                             [-2.8973, 2.8973]])

    arm = SerialArm(dh=dh, jt=jt, base=base, tip=tip, joint_limits=joint_limits)
    return arm


def Baxter(rl='right', s=1):
    """
    Rethink Robotics Baxter robot.

    References:
    R.L. Williams II, “Baxter Humanoid Robot Kinematics”, Internet Publication,
    https://www.ohio.edu/mechanical-faculty/williams/html/pdf/BaxterKinematics.pdf, April 2017.
    """
    L0 = 0.27035
    L1 = 0.069
    L2 = 0.36435
    L3 = 0.069
    L4 = 0.37429
    L5 = 0.010
    L6 = 0.36830

    L = 0.278
    h = 0.064
    H = 1.104
    # d theta a alpha
    if rl == 'left':
        dh = np.array([[L0, 0, L1, -np.pi / 2],
                       [0, np.pi / 2, 0, np.pi / 2],
                       [L2, 0, L3, -np.pi / 2],
                       [0, 0, 0, np.pi / 2],
                       [L4, 0, L5, -np.pi / 2],
                       [0, 0, 0, np.pi / 2],
                       [L6, 0, 0, 0]])

        base = se3(R=rotz(np.pi / 4), p=[L, -h, H])

    elif rl == 'right':
        dh = np.array([[L0, 0, L1, -np.pi / 2],
                       [0, np.pi / 2, 0, np.pi / 2],
                       [L2, 0, L3, -np.pi / 2],
                       [0, 0, 0, np.pi / 2],
                       [L4, 0, L5, -np.pi / 2],
                       [0, 0, 0, np.pi / 2],
                       [L6, 0, 0, 0]])

        base = se3(R=rotz(3*np.pi / 4), p=[-L, -h, H])

    dh[:, 0] *= s
    dh[:, 2] *= s

    jt = ['r', 'r', 'r', 'r', 'r', 'r', 'r']
    tip = se3()

    jointlim = np.array([[-141, 51],
                         [-123, 60],
                         [-173, 173],
                         [-3, 150],
                         [-175, 175],
                         [-90, 120],
                         [-175, 175]]) * np.pi / 180

    arm = SerialArm(dh=dh, jt=jt, base=base, tip=tip, joint_limits=jointlim)
    return arm


def Puma560(s=1):
    """
    Puma560 arm

    references:
    https://github.com/4rtur1t0/ARTE/blob/master/robots/UNIMATE/puma560/parameters.m

    Peter Corke robotics package
    """
    # d theta a alpha
    dh = np.array([[0.67183, 0, 0, np.pi / 2],
                   [0, 0, 0.4318, 0],
                   [0.15005, 0, 0.0203, -np.pi / 2],
                   [0.4318, 0, 0, np.pi / 2],
                   [0, 0, 0, -np.pi / 2],
                   [0, 0, 0, 0]])

    dh[:, 0] = dh[:, 0] * s
    dh[:, 2] = dh[:, 2] * s

    joint_limits = np.array([[-160, 160],
                            [-110, 110],
                            [-135, 135],
                            [-266, 266],
                            [-100, 100],
                            [-266, 266]]) * np.pi / 180

    jt = ['r', 'r', 'r', 'r', 'r', 'r']

    base = se3()
    tip = se3()

    arm = SerialArm(dh=dh, jt=jt, base=base, tip=tip, joint_limits=joint_limits)
    return arm


def Stanford(s=1):
    """
    Stanford manipulator

    References:
    https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/models/DH/Stanford.py
    """
    # d theta a alpha
    dh = np.array([[0.412, 0, 0, -np.pi / 2],
                   [0.154, 0, 0, np.pi / 2],
                   [0.3048, -np.pi / 2, 0.0203, 0],
                   [0, 0, 0, -np.pi / 2],
                   [0, 0, 0, np.pi / 2],
                   [0, 0, 0, 0]])

    dh[:, 0] *= s
    dh[:, 2] *= s

    jt = ['r', 'r', 'p', 'r', 'r', 'r']
    ang = 170 * np.pi / 180
    joint_limits = np.array([[-ang, ang],
                             [-ang, ang],
                             [0, 0.9652],
                             [-ang, ang],
                             [-np.pi / 2, np.pi / 2],
                             [-ang, ang]])

    base = se3()
    tip = se3()

    arm = SerialArm(dh=dh, jt=jt, base=base, tip=tip, joint_limits=joint_limits)
    return arm


def Kuka_KR5(s=1):
    """
    Kuka KR5 arm

    References:
        https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/models/DH/KR5.py

    """
    dh = np.array([[0.4, 0, 0.18, -np.pi / 2],
                   [0, 0, 0.6, 0],
                   [0, 0, 0.12, np.pi / 2],
                   [-0.62, 0, 0, -np.pi / 2],
                   [0, 0, 0, np.pi / 2],
                   [-0.115, 0, 0, 0]])

    dh[:, 0] *= s
    dh[:, 2] *= s

    jt = ['r', 'r', 'r', 'r', 'r', 'r']

    joint_limits = np.array([[-155, 155],
                             [-180, 65],
                             [-15, 158],
                             [-350, 350],
                             [-130, 130],
                             [-350, 350]]) * np.pi / 180

    base = se3()
    tip = se3()

    arm = SerialArm(dh=dh, jt=jt, base=base, tip=tip, joint_limits=joint_limits)
    return arm


def ABB(s=1):
    """
    ABB IRB140 Manipulator

    References:
        https://github.com/4rtur1t0/ARTE/blob/master/robots/ABB/IRB140/parameters.m
    """

    d1 = 0.352
    a1 = 0.070
    a2 = 0.360
    d4 = 0.380
    d6 = 0.065

    # d theta a alpha
    dh = np.array([[d1, 0, a1, -np.pi / 2],
                   [0, 0, a2, 0],
                   [0, 0, 0, -np.pi / 2],
                   [d4, 0, 0, np.pi / 2],
                   [0, 0, 0, -np.pi / 2],
                   [d6, 0, 0, 0]])

    joint_limits = np.array([[-180, 180],
                             [-100, 100],
                             [-220, 60],
                             [-200, 200],
                             [-120, 120],
                             [-400, 400]]) * np.pi / 180

    dh[:, 0] *= s
    dh[:, 2] *= s

    jt = ['r', 'r', 'r', 'r', 'r', 'r']

    base = se3()
    tip = se3()

    arm = SerialArm(dh=dh, jt=jt, base=base, tip=tip, joint_limits=joint_limits)
    return arm


def Canadarm2(s=1):
    """
    A model of the Canadarm2 on the ISS

    References:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6739427&tag=1
    """

    # d theta a alpha
    dh = np.array([[0.65, np.pi / 2, 0, np.pi / 2],
                   [0.3, np.pi / 2, 0, np.pi / 2],
                   [0.9, 0, 2.3, 0],
                   [0, 0, 2.3, 0],
                   [0, np.pi, 0, np.pi / 2],
                   [0.3, -np.pi / 2, 0, np.pi / 2],
                   [0.65, np.pi, 0, np.pi / 2]])

    dh[:, 0] *= s
    dh[:, 2] *= s

    jt = ['r', 'r', 'r', 'r', 'r', 'r', 'r']

    base = se3()
    tip = se3()

    arm = SerialArm(dh=dh, jt=jt, base=base, tip=tip)
    return arm

"""Kuka KR5, LWR, Stanford, PUMA, Cobra600, Fanuc AM120iB, M16, ABB IRB140, Kinova Jaco, Mico, offset6 spherical wrist, Trossen phantomX Pincher, UR10 universal robotics, Canadarm and Canadarm2 space shuttle"""