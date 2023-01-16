from RoboPy import *
import numpy as np
from numpy import pi

arm1 = PlanarDyn(n = 2, L = 1, joint_damping=[0.1, 0.1])
dh = [[0, 0, 1, 0], [0, 0, 1, 0]]
den = 1.0
mm = 0
damping = 0.1
arm2 = SimpleDynArm(dh, linear_density=den, damping=damping)

