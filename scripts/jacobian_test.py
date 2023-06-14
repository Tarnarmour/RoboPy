import numpy as np
from numpy import sin, cos, sqrt, pi
from numpy.linalg import norm
from RoboPy import *
from scipy import optimize

dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0.5, 0], [0, 0, 0.25, 0]]
arm = SerialArm(dh)
arm.set_qlim_warnings(False)
q = np.array([np.pi / 2, np.pi / 4, 0, -np.pi / 3])
x = arm.fk(q)[0:2, 3]

J = arm.jacob(q)[0:2]
H = arm.hessian(q)[0:2]
Jdag = np.linalg.pinv(J)

print('Jacobian:\n', J)
print('Hessian:')
print(H)

dq = np.array([0.5, 0.5, 0.5, 0.5]) * 0.5

print("error comparison")
print(x + J @ dq, np.linalg.norm(x + J @ dq - arm.fk(q + dq)[0:2, 3]))
print(x + J @ dq + 0.5 * dq @ H @ dq, np.linalg.norm(x + J @ dq + 0.5 * dq @ H @ dq - arm.fk(q + dq)[0:2, 3]))
print(arm.fk(q + dq)[0:2, 3])

qd = np.random.random((arm.n,))
Jdot = arm.jacobdot(q, qd)
dt = 1e-5
print(Jdot * dt)
print(arm.jacob(q + dt * qd) - arm.jacob(q))

