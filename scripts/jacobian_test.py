import numpy as np
from numpy import sin, cos, sqrt, pi
from numpy.linalg import norm
from RoboPy import *

dh = [[0, 0, 1, 0], [0, 0, 1, 0]]
arm = SerialArm(dh)
q = [0, 0]

J = arm.jacob(q, 2)

print(J)
