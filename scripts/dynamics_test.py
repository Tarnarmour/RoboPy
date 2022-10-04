import RoboPy as rp
import numpy as np
from numpy import pi, eye
import sympy as sp
from time import perf_counter

dh = [[0, 0, 1, 0], [0, 0, 1, 0]]
link_masses = [1, 1]
r_coms = [np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0])]
link_inertias = [eye(3), eye(3)]
joint_damping = np.array([1, 1])
arm = rp.SerialArmDyn(dh, mass=link_masses, r_com=r_coms, link_inertia=link_inertias, joint_damping=joint_damping)
q = np.array([1, 2])
qd = np.array([3, 4])
qdd = np.array([5, 6])
g = np.array([7, 8, 9])

M = arm.get_M(q)
C, Mdot = arm.get_C(q, qd)
G = arm.get_G(q, g)

print(f"M: \n{M}\n")
print(f"C: \n{C}\n")
print(f"G: \n{G}\n")
print(f"Mdot: \n{Mdot}\n")

print(f"Mdot - 2C: \n{Mdot - 2 * C}\n")

Wext = np.array([1, 2, 3, 4, 5, 6])

tau1 = arm.rne(q, qd, qdd, Wext=Wext, g=g)
tau2 = arm.EL(q, qd, qdd, Wext=Wext, g=g)
print(f"tau rne: {tau1}")
print(f"tau EL: {tau2}")
print(f"qdd inv: {qdd}")
qdd2 = arm.forward_rne(q, qd, tau1, g=g, Wext=Wext)
qdd3 = arm.forward_EL(q, qd, tau2, g=g, Wext=Wext)
print(f"qdds forward: {qdd2}, {qdd3}")

tick = perf_counter()

for i in range(10):
    qdd = arm.forward_rne(q, qd, tau1, g=g, Wext=Wext)

print(perf_counter() - tick)

tick = perf_counter()

for i in range(10):
    qdd = arm.forward_EL(q, qd, tau2, g=g, Wext=Wext)

print(perf_counter() - tick)

