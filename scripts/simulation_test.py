import RoboPy as rp
import numpy as np
import matplotlib.pyplot as plt

n = 3
b = np.ones((n,)) * 1.0
arm = rp.PlanarDyn(n=n, L=3 / n, joint_damping=b)
viz = rp.VizScene()
viz.add_arm(arm)

# tau = np.zeros((n,))

target = np.array([1.0, 0.5]) #+ np.array([np.sin(t), np.cos(t), 0])
viz.add_marker(np.hstack([target, 0]), pxMode=False, size=0.1)

ce = []
tpos = []
global count
count = 1

def tau(t, x):
    global count
    q = x[0:n]
    qd = x[n:2*n]

    w = 0.5
    tar = target + np.array([np.sin(t * w), np.cos(t * w)]) * 0.5


    e = tar - arm.fk(q, rep='cart')[0:2] + 0.5 * w * np.array([np.cos(t * w), -np.sin(t * w)])

    if count == 4:
        ce.append(np.linalg.norm(e))
        viz.update(poss=[np.hstack([tar, 0])])
        tpos.append(tar)
        count = 1
    else:
        count += 1

    J = arm.jacob(q)[0:2]

    M, C, G = arm.get_MCG(q, qd, g)

    # W = M
    W = np.eye(arm.n) # np.array([[1000, 0, 0], [0, 1, 0], [0, 0, 1]])
    # W = W / np.linalg.norm(W)
    Winv = np.linalg.inv(W)
    Jdag = Winv @ J.T @ np.linalg.inv(J @ Winv @ J.T + np.eye(2) * 1e-6)

    qd_des = Jdag @ e # + (np.eye(n) - Jdag @ J) @ -q

    qd_des = qd_des / np.linalg.norm(qd_des) * np.linalg.norm(e) * 1

    dt = 0.1

    qdd = (qd_des - qd) / dt

    T = M @ qdd + C @ qd + G

    T = np.clip(T, -10, 10)

    return T

q0 = np.random.random((n,)) * 0.2 - 0.1
qd0 = np.zeros((n,))
g = np.array([0, 0, 0])
Wext = np.array([0, 0, 0, 0, 0, 0])

dt = 0.05
tspan = [0, 15]

sys = rp.RobotSys(arm, dt, step_method='rk4', dyn_method='EL')
ts, qs, qds = sys.simulate(tspan, q0, qd0, tau, viz=viz, gravity=g, Wext=Wext, real_time=False)

xs = arm.fk(qs, rep='cart')
ce = np.array(ce)
tpos = np.array(tpos)

fig, axs = plt.subplots(1, 3)
axs[0].plot(ts, qs[:, 0], color=np.array([0, 0.1, 0.9, 1]), label='q1')
axs[0].plot(ts, qs[:, 1], color=np.array([0, 0.3, 0.7, 1]), label='q2')
axs[0].plot(ts, qs[:, 2], color=np.array([0, 0.5, 0.5, 1]), label='q3')

axs[1].plot(xs[:, 0], xs[:, 1], label='path')
axs[1].plot(tpos[:, 0], tpos[:, 1], label='target')
axs[1].axis('equal')

axs[2].plot(ts, ce)
axs[2].set_xlabel('time')
axs[2].set_ylabel('error (m)')

axs[0].legend()
axs[1].legend()

print(f"Score: {np.sum(ce)}")

plt.show()
