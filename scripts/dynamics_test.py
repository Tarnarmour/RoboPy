import RoboPy as rp
import numpy as np
from numpy import pi, eye
import sympy as sp
from time import perf_counter

dh = [[0, 0, 1, 0], [0, 0, 1, 0]]
link_masses = [1, 1]
r_coms = [np.array([-0.5, 0, 0]), np.array([-0.5, 0, 0])]
link_inertias = [eye(3), eye(3)]
arm = rp.SerialArmDyn(dh, mass=link_masses, r_com=r_coms, link_inertia=link_inertias)
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

tau = arm.rne(q, qd, qdd, Wext, g)
print(f"tau: {tau}")
print(f"qdd inv: {qdd}")
qdd2 = arm.forward_rne(q, qd, tau, g=g, Wext=Wext)
qdd3 = arm.forward_EL(q, qd, tau, g=g, Wext=Wext)
print(f"qdds forward: {qdd2}, {qdd3}")

tick = perf_counter()

for i in range(1000):
    qdd = arm.forward_rne(q, qd, tau, g=g, Wext=Wext)

print(perf_counter() - tick)

tick = perf_counter()

for i in range(1000):
    qdd = arm.forward_EL(q, qd, tau, g=g, Wext=Wext)

print(perf_counter() - tick)




# import TimingAnalysis as TA
# timer = TA.TimingAnalysis()
# timer.time_in("MAIN")
# for i in range(1000):
#     timer.time_in("rne")
#     tau_rne, wrench = arm.rne(q, qd, qdd, g=g)
#     timer.time_out("rne")
#     timer.time_in("el")
#     tau_el = arm.EL(q, qd, qdd, g=g)
#     timer.time_out("el")
#
# print(f"tau from RNE: \n{tau_rne}\n")
# print(f"tau from EL: \n{tau_el}\n")
#
# timer.report_all()