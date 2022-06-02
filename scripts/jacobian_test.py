import numpy as np
from numpy import sin, cos, sqrt, pi
from numpy.linalg import norm
from RoboPy import *

dh = [[0, 0, 1, 0], [0, 0, 1, 0]]
arm = SerialArm(dh)
q = [-1, 2]
qp = [-1, 2.1]

J = arm.jacob(q)
Jq = arm.jacoba(q, rep='q')

A1 = arm.fk(q)
q1 = arm.fk(q, rep='q')

A2 = arm.fk(qp)
q2 = arm.fk(qp, rep='q')

print(J)
print(A2 - A1)

print(Jq)
print(q2 - q1)
