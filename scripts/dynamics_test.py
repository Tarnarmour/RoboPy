import RoboPy as rp
import numpy as np
from numpy import pi, eye
from time import perf_counter

dh = [[0, 0, 0.4, 0]] * 3
link_masses = [1, 1, 1]
r_coms = [np.array([-0.2, 0, 0])] * 3
link_inertias = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.01]])] * 3
joint_damping = np.array([0, 0, 0])
arm = rp.SerialArmDyn(dh, mass=link_masses, r_com=r_coms, link_inertia=link_inertias, joint_damping=joint_damping)
q = np.array([pi / 4, pi / 4, pi / 4])
qd = np.array([pi / 6, -pi / 4, pi / 3])
qdd = np.array([-pi / 6, pi / 3, pi / 6])
g = np.array([0, -9.81, 0])

M = arm.get_M(q)
C, Mdot = arm.get_C(q, qd, get_mdot=True)
G = arm.get_G(q, g)

print(f"M: \n{M}\n")
print(f"C: \n{C}\n")
print(f"G: \n{G}\n")
print(f"Mdot: \n{Mdot}\n")

print(f"Mdot - 2C: \n{Mdot - 2 * C}\n")

Wext = np.array([0, 0, 0, 0, 0, 0])

tau1 = arm.rne(q, qd, qdd, Wext=Wext, g=g)
tau2 = arm.EL(q, qd, qdd, Wext=Wext, g=g)
print(f"tau rne: {tau1}")
print(f"tau EL: {tau2}")
