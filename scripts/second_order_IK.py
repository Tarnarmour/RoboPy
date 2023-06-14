import RoboPy as rp
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph.opengl as gl


dh = [[0, 0, 1, 0], [0, 0, 0.75, 0], [0, 0, 0.5, 0], [0, 0, 0.25, 0]]
arm = rp.SerialArm(dh)

viz = rp.VizScene()
viz.add_arm(arm)

fig, ax = plt.subplots()
ax.set_yscale('log')


def draw_path(qs, es, color=(0, 0, 1, 1), label=""):
    color_array = np.full((len(qs), 4), color)
    xs = arm.fk(qs, rep='cart')
    viz.window.addItem(gl.GLLinePlotItem(pos=xs, color=color_array))
    viz.add_marker(xs, color=color, size=0.05, pxMode=False)

    emag = np.asarray([np.linalg.norm(e) for e in es])
    ax.plot(emag, c=color, label=label)
    ax.legend()
    plt.pause(0.01)


def first_order_pseudo_ik(xt, q0, eps=1e-2):
    q = np.copy(q0)
    viz.update(q)
    e = xt - arm.fk(q, rep='xy')

    qs = [q]
    es = [e]
    dt = 1.0

    count = 0

    while np.linalg.norm(e) > eps and count < 1000:
        count += 1

        J = arm.jacob(q, rep='xy')
        Jdag = np.linalg.pinv(J)
        qdhat = Jdag @ e
        qdhat = qdhat / np.linalg.norm(qdhat)
        qd = qdhat * np.linalg.norm(e)

        q = q + qd * dt
        e = xt - arm.fk(q, rep='xy')

        qs.append(q)
        es.append(e)

    return q, qs, es


def first_order_transpose_ik(xt, q0, eps=1e-2):
    q = np.copy(q0)
    viz.update(q)
    e = xt - arm.fk(q, rep='xy')

    qs = [q]
    es = [e]
    dt = 1.0

    count = 0

    while np.linalg.norm(e) > eps and count < 1000:
        count += 1

        J = arm.jacob(q, rep='xy')
        qdhat = J.T @ e
        qdhat = qdhat / np.linalg.norm(qdhat)
        qd = qdhat * np.linalg.norm(e) * 0.1

        q = q + qd * dt
        e = xt - arm.fk(q, rep='xy')

        qs.append(q)
        es.append(e)

    return q, qs, es


def second_order_ik(xt, q0, eps=1e-2):
    q = np.asarray(q0)
    qd = np.zeros_like(q)

    Kp = np.diag([1, 1])
    Kd = np.diag([1, 1])

    dt = 0.001

    viz.update(q)
    e = xt - arm.fk(q, rep='xy')
    enorm = np.linalg.norm(e)

    qs = [q]
    es = [e]

    count = 0

    while enorm > eps and count < 1000:
        count += 1

        J = arm.jacob(q, rep='xy')

        try:
            Jdag = np.linalg.pinv(J)
        except np.linalg.LinAlgError:
            Jdag = J.T @ np.linalg.inv(J @ J.T + np.eye(2) * 0.01)

        Jdot = arm.jacobdot(q, qd)[0:2]
        ehat = e / enorm

        qdd = Jdag @ ()
        qd = qd + qdd * dt
        q = q + qd * dt

        q = rp.wrap_angle(q)

        qs.append(q)
        es.append(e)

    return q, qs, es


q0 = np.array([2 * np.pi / 3, np.pi / 2, -np.pi / 2, 0.0])
xt = np.array([0.8, 1.6])
xt3 = np.hstack([xt, [0]])

eps = 1e-4

viz.add_marker(xt3, color=[1, 1, 0, 1], size=0.1, pxMode=False)
viz.add_marker(arm.fk(q0, rep='cart'), color=[1, 1, 1, 1], size=0.1, pxMode=False)

qf, qs, es = first_order_pseudo_ik(xt, q0, eps)
draw_path(qs, es, [0, 0, 1, 1], "first order pseudo")

qf, qs, es = first_order_transpose_ik(xt, q0, eps)
draw_path(qs, es, [0, 1, 0, 1], "first order transpose")

qf, qs, es = second_order_ik(xt, q0, eps)
draw_path(qs, es, [1, 0, 0, 1], "second order")

while True:
    for q in qs:
        viz.update(q)
        viz.hold(0.01)
    viz.hold(1.0)
