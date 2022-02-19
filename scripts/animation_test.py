from RoboPy import *
import numpy as np
from numpy import pi, sin, cos, sqrt, cross
from numpy.linalg import norm

n = 9
max_length = 0.5
dh = np.random.random_sample((n,4)) * 2 * max_length - max_length
q = np.zeros((n,))
jt = ['r'] * len(dh)
tip = se3()
arm = SerialArm(dh, jt, tip=tip)

viz = VizScene()
viz.add_arm(arm, draw_frames=False)

viz.update()

A_target = arm.fk(np.random.random_sample((n,)))
viz.add_frame(A_target)
viz.update(q)

max_speed = 0.05

dq = np.random.random_sample((n,)) * 2 * max_speed - max_speed
print(dq)

# while True:
#     q = q + dq
#     viz.update(q)

sol = arm.ik(A_target,
            method='pinv',
            rep='rpy',
            max_iter=np.inf,
            tol=1e-3,
            viz=viz,
            min_delta=1e-5,
            max_delta=10)

print(sol)

viz.hold()
