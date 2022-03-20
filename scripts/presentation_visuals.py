from RoboPy import *
import RoboPy as rp
import numpy as np
from numpy import pi
from john_radlab.Jacobian_Orthogonality.source import AIM, findY

# dh = [[0, 0, 2, 0], [0, 0, 1, 0], [0, 0, 0.3, 0], [0, 0, 1, 0]]
# dh = [[0, 0, 2.41497930, 0], [0, 0, 1.71892394e+00, 0], [0, 0, 1.38712293e-03, 0], [0, 0, 3.89431195e-01, 0]]
# dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
dh = np.random.random_sample((4,4))
arm = SerialArm(dh)

viz = VizScene()
viz.add_arm(arm)
viz.add_frame

q0 = np.array([0, pi/2, pi/2, pi/2])
J = arm.jacob(q0)
Y = findY(J)
w = AIM(J, 'w')
np.set_printoptions(precision=2)

print(f"q: {q0}\nJ: \n{clean_rotation_matrix(J, 0.01)}\n Y: \n{clean_rotation_matrix(Y)}\n---- Scores ----\nW: {w}, Fro: {AIM(J,'fro')}, Min: {AIM(J,'min')}")
viz.update(q0)
dq = np.array([pi/1000, 2*pi/1000, 4*pi/1000, 8*pi/1000])
q = q0
while True:
    q += dq
    viz.update(q)
viz.hold()
