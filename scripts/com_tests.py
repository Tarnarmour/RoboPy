import RoboPy as rp
import numpy as np


dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0.5, 0]]
arm = rp.SerialArm(dh)

q = [0, 0, np.pi / 2]
print(arm.fk_com(q))
print(arm.jacob_com(q))
