from RoboPy import *
import time

viz = VizScene()

R0 = rpy2R([0, 0, 0])
R1 = rpy2R(np.random.random_sample((3,)) * pi)

A0 = se3(R0)
A1 = se3(R1)

q0 = R2q(R0)
q1 = R2q(R1)

print(quat_dist(q0, q1))

viz.add_frame(A0)
viz.add_frame(A1)
viz.add_frame(A0)

ts = np.linspace(0, 1, 500)

while True:
    for t in ts:
        q = slerp(q0, q1, t)
        R = q2R(q)
        A = se3(R)
        viz.update(As=[A0, A1, A])
        time.sleep(0.001)
    for t in ts:
        q = slerp(q1, q0, t)
        R = q2R(q)
        A = se3(R)
        viz.update(As=[A0, A1, A])
        time.sleep(0.001)
