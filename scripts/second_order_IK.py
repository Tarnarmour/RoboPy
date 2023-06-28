import RoboPy as rp
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph.opengl as gl
import time


dh = [[0, 0, 1, 0], [0, 0, 1.0, 0], [0, 0, 0.5, 0], [0, 0, 0.5, 0], [0, 0, 0.25, 0.], [0., 0., 0.25, 0.]]
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


def first_order_pseudo_ik(ft, q0, dt, eps=1e-2, tf=10.):
    q = np.copy(q0)

    t = 0.0

    xt = ft(t)[0]
    vt = ft(t)[1]
    e = xt - arm.fk(q, rep='xy')

    qs = [q]
    es = [e]
    ts = [0]

    count = 0

    while t < tf and np.linalg.norm(e) > eps:
        count += 1

        if np.linalg.norm(e) < eps:
            pass
        else:
            J = arm.jacob(q, rep='xy')
            try:
                Jdag = np.linalg.pinv(J)
            except np.linalg.LinAlgError:
                Jdag = J.T @ np.linalg.inv(J @ J.T + np.eye(2) * 1e-4)
            qdhat = Jdag @ e
            qdhat = qdhat / np.linalg.norm(qdhat)
            qd = qdhat * np.linalg.norm(e) + Jdag @ vt / (1 + np.linalg.norm(e))
            q = q + qd * dt

            qs.append(q)
            es.append(e)
            ts.append(t)

        e = xt - arm.fk(q, rep='xy')
        t = t + dt
        xt = ft(t)[0]
        vt = ft(t)[1]

    return q, qs, es, ts


def first_order_transpose_ik(ft, q0, dt, eps=1e-2, tf=10.0):
    q = np.copy(q0)

    t = 0.0

    xt = ft(t)[0]
    e = xt - arm.fk(q, rep='xy')

    qs = [q]
    es = [e]
    ts = [t]

    count = 0

    while t < tf and np.linalg.norm(e) > eps:
        count += 1

        if np.linalg.norm(e) < eps:
            pass
        else:
            J = arm.jacob(q, rep='xy')
            qdhat = J.T @ e
            qdhat = qdhat / np.linalg.norm(qdhat)
            qd = qdhat * np.linalg.norm(e) * 5
            q = q + qd * dt

            qs.append(q)
            es.append(e)
            ts.append(t)

        e = xt - arm.fk(q, rep='xy')
        t = t + dt
        xt = ft(t)[0]

    return q, qs, es, ts


def second_order_ik(ft, q0, dt, eps=1e-2, tf=10.0):
    q = np.copy(q0)
    qd = np.zeros_like(q)

    Kp = np.diag([5., 5.])
    Kd = np.diag([1., 1.])
    kd = 1.0

    t = 0.0

    xtc, vtc, atc = ft(t)

    e = xtc - arm.fk(q, rep='xy')
    enorm = np.linalg.norm(e)

    qs = [q]
    es = [e]
    ts = [t]

    count = 0

    while t < tf and np.linalg.norm(e) > eps:
        count += 1

        if enorm < eps:
            pass
        else:
            J = arm.jacob(q, rep='xy')

            try:
                Jdag = np.linalg.pinv(J)
            except np.linalg.LinAlgError:
                Jdag = J.T @ np.linalg.inv(J @ J.T + np.eye(2) * 0.01)

            Jdot = arm.jacobdot(q, qd)[0:2]

            qdd = Jdag @ (atc + Kp @ e + Kd @ (vtc - J @ qd) - Jdot @ qd) - (np.eye(arm.n) - Jdag @ J) @ (kd * qd)
            qd = qd + qdd * dt
            q = q + qd * dt + 0.5 * qdd * dt**2

            qs.append(q)
            es.append(e)
            ts.append(t)

        e = xtc - arm.fk(q, rep='xy')
        enorm = np.linalg.norm(e)

        t = t + dt
        xtc, vtc, atc = ft(t)

    return q, qs, es, ts


def random_ik(ft, q0, dt, eps=1e-2, tf=10.0):
    q = np.copy(q0)
    t = 0.0

    xtc, vtc, atc = ft(t)
    e = xtc - arm.fk(q, rep='xy')

    qs = [q]
    es = [e]
    ts = [t]

    while t < tf and np.linalg.norm(e) > eps:

        dqs = (np.random.random((10, arm.n)) * 2 - 1) * np.linalg.norm(e)
        des = np.asarray([np.linalg.norm(xtc - arm.fk(q + dq, rep='xy')) for dq in dqs])

        q = q + dqs[np.argmin(des)]

        qs.append(q)
        es.append(e)
        ts.append(t)

        e = xtc - arm.fk(q, rep='xy')
        t = t + dt
        xtc, vtc, atc = ft(t)

    return q, qs, es, ts


def fabrik_ik(ft, q0, dt, eps=1e-2, tf=10.0):
    q = np.copy(q0)
    t = 0.0

    xtc, vtc, atc = ft(t)
    e = xtc - arm.fk(q, rep='xy')

    qs = [q]
    es = [e]
    ts = [t]

    rs = [dh[2] for dh in arm.dh]

    while t < tf and np.linalg.norm(e) > eps:

        points = np.zeros((arm.n + 1, 2))
        ppoints = np.zeros_like(points)
        pppoints = np.zeros_like(points)

        for i in range(arm.n):
            points[i + 1] = arm.fk(q, index=i + 1, rep='xy')

        ppoints[-1] = xtc
        for i in range(arm.n, 0, -1):
            v = points[i - 1] - ppoints[i]
            v = v / np.linalg.norm(v) * rs[i - 1]
            ppoints[i - 1] = ppoints[i] + v

        for i in range(arm.n):
            v = ppoints[i + 1] - pppoints[i]
            v = v / np.linalg.norm(v) * rs[i]
            pppoints[i + 1] = pppoints[i] + v

        thp = np.zeros_like(q)
        for i in range(arm.n):
            w = pppoints[i + 1] - pppoints[i]
            thp[i] = np.arctan2(w[1], w[0])

        qp = np.zeros_like(q)
        qp[0] = thp[0]
        for i in range(1, arm.n):
            qp[i] = thp[i] - thp[i - 1]

        q = qp

        qs.append(q)
        es.append(e)
        ts.append(t)

        e = xtc - arm.fk(q, rep='xy')
        t = t + dt
        xtc, vtc, atc = ft(t)

    return q, qs, es, ts


def ft(t):
    w = 0.25
    a = 0.25
    b = 0.25
    xt = np.array([0.8, 1.6]) + np.array([a * np.sin(w * t), b * np.cos(w * t) - b])
    vt = w * np.array([a * np.cos(w * t), b * -np.sin(w * t)])
    at = w * w * np.array([a * -np.sin(w * t), b * -np.cos(w * t)])
    return xt, vt, at
    # xt = np.array([1, -1.0])
    # vt = np.array([0.0, 0.0])
    # at = np.array([0.0, 0.0])
    # return xt, vt, at


q0 = np.array([2 * np.pi / 3, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0])
xt = np.array([0.8, 1.6])
xt3 = np.hstack([xt, [0]])

eps = 1e-8
dt = 0.05
tf = 10

viz.add_marker(xt3, color=[1, 1, 0, 1], size=0.1, pxMode=False)
viz.add_marker(arm.fk(q0, rep='cart'), color=[1, 1, 1, 1], size=0.1, pxMode=False)

k_trial = 20

tic = time.perf_counter()
for i in range(k_trial):
    _, qs_fop, es, ts1 = first_order_pseudo_ik(ft, q0, dt, eps, tf)
draw_path(qs_fop, es, [0, 0, 1, 1], "first order pseudo")
print(f"first order pseudo: {(time.perf_counter() - tic) / k_trial}")

tic = time.perf_counter()
for i in range(k_trial):
    _, qs_fot, es, ts2 = first_order_transpose_ik(ft, q0, dt, eps, tf)
draw_path(qs_fot, es, [0, 1, 0, 1], "first order transpose")
print(f"first order transpose: {(time.perf_counter() - tic) / k_trial}")

tic = time.perf_counter()
for i in range(k_trial):
    _, qs_sop, es, ts3 = second_order_ik(ft, q0, dt, eps, tf)
draw_path(qs_sop, es, [1, 0, 0, 1], "second order")
print(f"second order pseudo: {(time.perf_counter() - tic) / k_trial}")

tic = time.perf_counter()
for i in range(k_trial):
    _, qs_ran, es, ts4 = random_ik(ft, q0, dt, eps, tf)
draw_path(qs_ran, es, [0.0, 0.707, 0.707, 1], "random")
print(f"random: {(time.perf_counter() - tic) / k_trial}")

tic = time.perf_counter()
for i in range(k_trial):
    _, qs_fab, es, ts5 = fabrik_ik(ft, q0, dt, eps, tf)
draw_path(qs_fab, es, [0.707, 0.707, 0, 1], "fabrik")
print(f"fabrik: {(time.perf_counter() - tic) / k_trial}")

while True:
    for q, t in zip(qs_fop, ts1):
        xt = ft(t)[0]
        xt3 = np.hstack([xt, [0]])
        viz.markers[0].setData(pos=xt3)
        viz.update(q)
        viz.hold(0.02)
    viz.hold(1.0)

    for q, t in zip(qs_fot, ts2):
        xt = ft(t)[0]
        xt3 = np.hstack([xt, [0]])
        viz.markers[0].setData(pos=xt3)
        viz.update(q)
        viz.hold(0.02)
    viz.hold(1.0)

    for q, t in zip(qs_sop, ts3):
        xt = ft(t)[0]
        xt3 = np.hstack([xt, [0]])
        viz.markers[0].setData(pos=xt3)
        viz.update(q)
        viz.hold(0.02)
    viz.hold(1.0)

    for q, t in zip(qs_ran, ts4):
        xt = ft(t)[0]
        xt3 = np.hstack([xt, [0]])
        viz.markers[0].setData(pos=xt3)
        viz.update(q)
        viz.hold(0.02)
    viz.hold(1.0)

    for q, t in zip(qs_fab, ts5):
        xt = ft(t)[0]
        xt3 = np.hstack([xt, [0]])
        viz.markers[0].setData(pos=xt3)
        viz.update(q)
        viz.hold(0.02)
    viz.hold(1.0)
