import RoboPy as rp
import numpy as np


dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
q = tuple(np.radians([0, 60, -30, 30]))
arm = rp.SerialArm(dh)
rep = 'planar'
print(arm.yoshikawa(q, rep), arm.rcond(q, rep), arm.aim(q, rep, 'min'))
