from RoboPy import *
import numpy as np
from numpy import pi, sin, cos, sqrt, cross
from numpy.linalg import norm

n = 7
max_length = 0.5
dh = np.random.random_sample((n,4)) * 2 * max_length - max_length
dh = [[pi/6, 0, 0.25, pi/3]] * n
q = np.zeros((n,))
jt = ['r'] * len(dh)
tip = se3()
arm = SerialArm(dh, jt, tip=tip)

viz = VizScene()
viz.add_arm(arm, draw_frames=False)

viz.update()

viz.update(q)

max_speed = 0.05

dq = np.random.random_sample((n,)) * 2 * max_speed - max_speed
count = 0
while count < 300:
    q = q + dq
    viz.update(q)
    count += 1


count_q = 0
count_rpy = 0

for i in range(3):
    A_target = arm.fk(np.random.random_sample((n,)))
    viz.add_frame(A_target)
    sol = arm.ik(A_target,
                 q0=q,
                method='pinv',
                rep='cart',
                max_iter=np.inf,
                tol=1e-3,
                viz=viz,
                min_delta=1e-5,
                max_delta=0.01)
    count_q += sol.status
    q = sol.qf

    viz.remove_frame()


print(count_q)
print(count_rpy)


viz.wander(duration=10.0, q0=[q])



viz.hold()
