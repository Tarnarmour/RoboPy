import RoboPy as rp
import numpy as np

dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
qlim = None

arm = rp.SerialArm(dh, joint_limits=qlim)

print(arm.qlim)

print(arm.randq(100))
