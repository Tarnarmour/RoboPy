import RoboPy as rp
import numpy as np
import matplotlib.pyplot as plt


arm = rp.PlanarDyn(n=3, L=2.0)
viz = rp.VizScene()
viz.add_arm(arm)

tau = np.array([0, 0, -0.05])
def tau(t, x):
    r = np.array([4, -0.5, 2.2])
    q = x[0:3]
    qd = x[3:6]
    kp, kd = 12.0, 20.0
    qrel = rp.wrap_relative(q, r)
    T = kp * (r - qrel) - kd * qd
    T = np.clip(T, -10, 10)
    return T

q0 = np.zeros((3,))
qd0 = np.zeros((3,))
g = np.array([0, 0, 0])
Wext = np.array([0, 0, 0, 0, 0, 0])

dt = 0.05
tspan = [0, 20]

sys = rp.RobotSys(arm, dt, step_method='rk4', dyn_method='EL')
ts, qs, qds = sys.simulate(tspan, q0, qd0, tau, viz=viz, gravity=g, Wext=Wext, real_time=True)

fig, ax = plt.subplots()
ax.plot(ts, qs[:, 0], color=np.array([0, 0.1, 0.9, 1]))
ax.plot(ts, qs[:, 1], color=np.array([0, 0.3, 0.7, 1]))
ax.plot(ts, qs[:, 2], color=np.array([0, 0.5, 0.5, 1]))

ax.plot(ts, qds[:, 0], color=np.array([0.9, 0.1, 0, 1]))
ax.plot(ts, qds[:, 1], color=np.array([0.7, 0.1, 0, 1]))
ax.plot(ts, qds[:, 2], color=np.array([0.5, 0.1, 0, 1]))

ax.legend(['q1', 'q2', 'q3', 'qd1', 'qd2', 'qd3'])
plt.show()
