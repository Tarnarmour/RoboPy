import numpy as np
import mpmath as mp
import sympy as sp
from scipy.optimize import minimize

def wrap_angle(q):
    def f(q):
        q = np.mod(q, 2 * np.pi)
        if q > np.pi:
            q -= 2 * np.pi
        return q

    if isinstance(q, (np.ndarray, list, tuple)):
        for i in range(len(q)):
            q[i] = f(q[i])
    else:
        q = f(q)
    return q

def wrap_relative(q1, q2):
    if hasattr(q1, '__iter__'):
        if hasattr(q2, '__iter__'):
            for i, (r, w) in enumerate(zip(q1, q2)):
                while r - w > np.pi:
                    r = r - 2 * np.pi
                while r - w < -np.pi:
                    r = r + 2 * np.pi
                q1[i] = r
        else:
            for i, r in enumerate(q1):
                while r - q2 > np.pi:
                    r = r - 2 * np.pi
                while r - q2 < -np.pi:
                    r = r + 2 * np.pi
                q1[i] = r
    else:
        while q1 - q2 > np.pi:
            q1 = q1 - 2 * np.pi
        while q1 - q2 < -np.pi:
            q1 = q1 + 2 * np.pi
    return q1

def wrap_diff(q1, q2):
    q = wrap_relative(q1, q2)
    return q - q2

def skew(v):
    if hasattr(v[0], '__len__'):
        print("Input to skew(v) must be 1 dimensional!")
        return None
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def clean_rotation_matrix(R, eps=1e-12):
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.abs(R[i, j]) < eps:
                R[i, j] = 0.
            elif np.abs(R[i, j] - 1) < eps:
                R[i, j] = 1.
    return R

def mprint(A, p=3, eps=1e-12):
    if isinstance(A, (np.ndarray)):
        if A.ndim == 2:
            with np.printoptions(precision=p, suppress=True, floatmode='fixed', linewidth=200):
                print(clean_rotation_matrix(A, eps))
                return
        elif A.ndim > 2:
            print('[ ')
            for i in range(len(A)):
                mprint(A[i], p)
                if i < len(A) - 1:
                    print(', ')
            print(']\n')
        else:
            with np.printoptions(precision=p, suppress=True, floatmode='fixed', linewidth=200):
                print(A)
                return
    elif hasattr(A, '__len__'):
        print('[ ')
        for i in range(len(A)):
            mprint(A[i], p)
            if i < len(A) - 1:
                print(', ')
        print(']\n')
    else:
        with np.printoptions(precision=p, suppress=True, floatmode='fixed'):
            print(A)
            return

def piecewise_func(funcs, xs):
    def g(t):
        for f, x in zip(funcs, xs):
            if t < x:
                return f(t)
        return funcs[-1](t)
    return g

def cubic_interpolation(t0, tf, y0, yf, ms=0.0, mf=0.0, deriv=False):
    h = yf - y0
    yf = wrap_relative(yf, y0)

    A = (3*h*tf*t0**2 - h*t0**3 - mf*tf**2*t0**2 + mf*tf*t0**3 - ms*tf**3*t0 + ms*tf**2*t0**2 + tf**3*y0 - 3*tf**2*t0*y0 + 3*tf*t0**2*y0 - t0**3*y0)/(tf**3 - 3*tf**2*t0 + 3*tf*t0**2 - t0**3)
    B = (-6*h*tf*t0 + 2*mf*tf**2*t0 - mf*tf*t0**2 - mf*t0**3 + ms*tf**3 + ms*tf**2*t0 - 2*ms*tf*t0**2)/(tf**3 - 3*tf**2*t0 + 3*tf*t0**2 - t0**3)
    C = (3*h*tf + 3*h*t0 - mf*tf**2 - mf*tf*t0 + 2*mf*t0**2 - 2*ms*tf**2 + ms*tf*t0 + ms*t0**2)/(tf**3 - 3*tf**2*t0 + 3*tf*t0**2 - t0**3)
    D = (-2*h + mf*tf - mf*t0 + ms*tf - ms*t0)/(tf**3 - 3*tf**2*t0 + 3*tf*t0**2 - t0**3)

    def cubic(t):
        if t < t0:
            return y0
        elif t > tf:
            return yf
        else:
            y = wrap_angle(A + B*t + C*t**2 + D*t**3)
            # y = A + B*t + C*t**2 + D*t**3
            return y

    def velocity(t):
        if t < t0:
            return 0.0
        elif t > tf:
            return 0.0
        else:
            y = B + 2*C*t + 3*D*t**2
            return y

    def acceleration(t):
        if t < t0:
            return 0.0
        elif t > tf:
            return 0.0
        else:
            y = 2*C + 6*D*t
            return y

    if deriv:
        return cubic, velocity, acceleration
    else:
        return cubic

def vector_cubic_interpolation(t0, tf, v0, vf, ms=None, mf=None, deriv=False):
    funcs = []
    dfuncs = []
    ddfuncs = []

    n = v0.shape[0]

    if ms is None:
        ms = np.zeros((n,))
    if mf is None:
        mf = np.zeros((n,))

    for i in range(n):
        if deriv:
            f, fd, fdd = cubic_interpolation(t0, tf, v0[i], vf[i], ms[i], mf[i], deriv=True)
            funcs.append(f)
            dfuncs.append(fd)
            ddfuncs.append(fdd)
        else:
            f = cubic_interpolation(t0, tf, v0[i], vf[i], ms[i], mf[i], deriv=False)
            funcs.append(f)

    def g(t):
        y = np.zeros((n,))
        for i in range(n):
            y[i] = funcs[i](t)
        return y

    if deriv:
        def gd(t):
            y = np.zeros((n,))
            for i in range(n):
                y[i] = dfuncs[i](t)

            return y

        def gdd(t):
            y = np.zeros((n,))
            for i in range(n):
                y[i] = ddfuncs[i](t)
            return y

        return g, gd, gdd

    else:
        return g

def quartic_interpolation(t0, tf, x0, xf, v0=0.0, vf=0.0, a0=0.0, deriv=False):

    T = tf - t0

    c0 = x0
    c1 = v0
    c2 = a0/2
    c3 = (-T**2*a0 - 3*T*v0 - T*vf - 4*x0 + 4*xf)/T**3
    c4 = (T**2*a0 + 4*T*v0 + 2*T*vf + 6*x0 - 6*xf)/(2*T**4)

    def f(x):
        if x < t0:
            return x0
        elif x > tf:
            return xf
        else:
            x = x - t0
            return c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4

    def fd(x):
        if x < t0:
            return v0
        elif x > tf:
            return vf
        else:
            x = x - t0
            return c1 + 2 * c2 * x ** 1 + 3 * c3 * x ** 2 + 4 * c4 * x ** 3

    def fdd(x):
        if x < t0:
            return a0
        elif x > tf:
            return 2 * c2 + 6 * c3 * tf + 12 * c4 * tf ** 2
        else:
            x = x - t0
            return 2 * c2 + 6 * c3 * x + 12 * c4 * x ** 2

    if deriv:
        return f, fd, fdd
    else:
        return f

def vector_quartic_interpolation(t0, tf, x0, xf, v0, vf, a0, deriv=False):
    funcs = []
    dfuncs = []
    ddfuncs = []

    n = x0.shape[0]

    for i in range(n):
        f, fd, fdd = quartic_interpolation(t0, tf, x0[i], xf[i], v0[i], vf[i], a0[i], deriv=True)
        funcs.append(f)
        dfuncs.append(fd)
        ddfuncs.append(fdd)

    def g(t):
        y = np.zeros((n,))
        for i in range(n):
            y[i] = funcs[i](t)
        return y

    if deriv:
        def gd(t):
            y = np.zeros((n,))
            for i in range(n):
                y[i] = dfuncs[i](t)
            return y

        def gdd(t):
            y = np.zeros((n,))
            for i in range(n):
                y[i] = ddfuncs[i](t)
            return y

        return g, gd, gdd
    else:
        return g

def quintic_interpolation(t0, tf, x0, xf, v0=None, vf=None, a0=None, af=None, deriv=False, cost='minjerk'):

    # x will be a list of polynomial coefficients in this order: x = [c3, c4, c5, c2, c1], because if we have
    # some boundary conditions the first few coefficients may be predetermined.

    xf = wrap_relative(xf, x0)

    def f(t, x):
        c = np.zeros((6,))
        c[0] = x0
        c[1] = v0 if v0 is not None else x[-1]
        c[2] = a0 / 2 if a0 is not None else x[-2]
        c[3:None] = x[0:3]
        return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5

    def fd(t, x):
        c = np.zeros((6,))
        c[0] = x0
        c[1] = v0 if v0 is not None else x[-1]
        c[2] = a0 / 2 if a0 is not None else x[-2]
        c[3:None] = x[0:3]
        return c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4

    def fdd(t, x):
        c = np.zeros((6,))
        c[0] = x0
        c[1] = v0 if v0 is not None else x[-1]
        c[2] = a0 / 2 if a0 is not None else x[-2]
        c[3:None] = x[0:3]
        return 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3

    def fddd(t, x):
        c = np.zeros((6,))
        c[0] = x0
        c[1] = v0 if v0 is not None else x[-1]
        c[2] = a0 / 2 if a0 is not None else x[-2]
        c[3:None] = x[0:3]
        return 6 * c[3] + 24 * c[4] * t + 60 * c[5] * t ** 2

    dt = 0.01
    ts = np.linspace(0, 1, 100)

    if cost == 'minjerk':
        def g(x):
            H = 0.0
            for t in ts:
                H += fddd(t, x) ** 2
            return H
    elif cost == 'minacc':
        def g(x):
            H = 0.0
            for t in ts:
                H += fdd(t, x) ** 2
            return H
    elif cost == 'maxacc':
        def g(x):
            H = 0.0
            for t in ts:
                if abs(fdd(t, x)) > H:
                    H = abs(fdd(t, x))
            return H
    elif hasattr(cost, '__call__'):
        g = cost
    elif cost == 'linear':
        def g(x):
            H = 0.0
            ys = np.linspace(x0, xf, len(ts))
            for t, y in zip(ts, ys):
                H += (f(t, x) - y)**2
            return H
    elif cost == 'minvel':
        def g(x):
            H = 0.0
            for t in ts:
                H += fd(t, x)**2
            return H


    con = []
    con.append({'type': 'eq', 'fun': lambda x: f(1, x) - xf})
    if vf is not None:
        con.append({'type': 'eq', 'fun': lambda x: fd(1, x) - vf})
    if af is not None:
        con.append({'type': 'eq', 'fun': lambda x: fdd(1, x) - af})

    xStart = np.zeros((sum(y is None for y in [v0, a0]) + 3))
    sol = minimize(g, xStart, method='SLSQP', constraints=con)
    # print(sol.status)
    if sol.status != 0:
        print(sol.message)

    c = np.zeros((6,))
    c[0] = x0
    c[1] = v0 if v0 is not None else sol.x[-1]
    c[2] = a0 / 2 if a0 is not None else sol.x[-2]
    c[3:None] = sol.x[0:3]

    def f(x):
        if x < t0:
            return x0
        elif x > tf:
            return xf
        else:
            t = (x - t0) / (tf - t0)
            return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5

    def fd(x):
        if x < t0:
            return 0.0
        elif x > tf:
            return 0.0
        else:
            t = (x - t0) / (tf - t0)
            return c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4

    def fdd(x):
        if x < t0:
            return 0.0
        elif x > tf:
            return 0.0
        else:
            t = (x - t0) / (tf - t0)
            return 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3

    if deriv:
        return f, fd, fdd
    else:
        return f

def vector_quintic_interpolation(t0, tf, x0, xf, v0=None, vf=None, a0=None, af=None, deriv=False, cost='minjerk'):
    funcs = []
    dfuncs = []
    ddfuncs = []

    n = x0.shape[0]

    v0 = [None] * n if v0 is None else v0
    vf = [None] * n if vf is None else vf
    a0 = [None] * n if a0 is None else a0
    af = [None] * n if af is None else af

    for i in range(n):
        f, fd, fdd = quintic_interpolation(t0, tf, x0[i], xf[i], v0[i], vf[i], a0[i], deriv=True, cost=cost)
        funcs.append(f)
        dfuncs.append(fd)
        ddfuncs.append(fdd)

    def g(t):
        y = np.zeros((n,))
        for i in range(n):
            y[i] = funcs[i](t)
        return y

    if deriv:
        def gd(t):
            y = np.zeros((n,))
            for i in range(n):
                y[i] = dfuncs[i](t)
            return y

        def gdd(t):
            y = np.zeros((n,))
            for i in range(n):
                y[i] = ddfuncs[i](t)
            return y

        return g, gd, gdd
    else:
        return g

def cubic_spline(ts, qs, qds=None, deriv=False):

    if len(qs.shape) > 1:
        m, n = qs.shape[0], qs.shape[1]
        oneD = False
        ms = np.zeros((n,))
        mf = np.zeros((n,))
        funcs = [lambda t: qs[0]]
        dfuncs = [lambda t: np.zeros((n,))]
        ddfuncs = [lambda t: np.zeros((n,))]
    else:
        m, n = qs.shape[0], 1
        oneD = True
        ms = 0.0
        mf = 0.0
        funcs = [lambda t: qs[0]]
        dfuncs = [lambda t: 0.0]
        ddfuncs = [lambda t: 0.0]

    for i in range(1, m):
        if qds is None:
            if i > 1:
                if oneD:
                    ms = mf
                else:
                    ms = np.copy(mf)
            if i == m - 1:
                if oneD:
                    mf = 0.0
                else:
                    mf = np.zeros((n,))
            elif not oneD:
                for j in range(n):
                    if (qs[i-1, j] > qs[i, j] < qs[i+1, j]) or (qs[i-1, j] < qs[i, j] > qs[i+1, j]):
                        mf[j] = 0.0
                    elif i == 1:
                        mf[j] = (qs[i+1, j] - qs[i, j]) / (ts[i+1] - ts[i])
                        # mf[j] = (qs[i + 2, j] - qs[i + 1, j]) / (ts[i + 2] - ts[i + 1])
                    else:
                        # mf[j] = (qs[i + 1, j] - qs[i-1, j]) / (ts[i + 1] - ts[i-1])
                        mf[j] = (qs[i, j] - qs[i - 1, j]) / (ts[i] - ts[i - 1])
                        # mf[j] = (qs[i + 1, j] - qs[i, j]) / (ts[i + 1] - ts[i])

            else:
                if (qs[i-1] > qs[i] < qs[i+1]) or (qs[i-1] < qs[i] > qs[i+1]):
                    mf = 0.0
                elif i == 1:
                    mf = (qs[i+1] - qs[i]) / (ts[i+1] - ts[i])
                else:
                    # mf = (qs[i + 1] - qs[i-1]) / (ts[i + 1] - ts[i-1])
                    mf = (qs[i] - qs[i - 1]) / (ts[i] - ts[i - 1])
                    # mf = (qs[i + 1] - qs[i]) / (ts[i + 1] - ts[i])
        else:
            ms = qds[i - 1]
            mf = qds[i]

        if oneD:
            f, fd, fdd = cubic_interpolation(ts[i - 1], ts[i], qs[i - 1], qs[i], ms, mf, deriv=True)
        else:
            f, fd, fdd = vector_cubic_interpolation(ts[i-1], ts[i], qs[i-1], qs[i], ms, mf, deriv=True)
        funcs.append(f)
        dfuncs.append(fd)
        ddfuncs.append(fdd)

    pf = piecewise_func(funcs, ts)
    pfd = piecewise_func(dfuncs, ts)
    pfdd = piecewise_func(ddfuncs, ts)

    if deriv:
        return pf, pfd, pfdd
    else:
        return pf

def quartic_spline(ts, qs, qds, qdd0, deriv=False):

    m = qs.shape[0]

    funcs = [lambda t: qs[0]]
    dfuncs = [lambda t: qds[0]]
    ddfuncs = [lambda t: qdd0]
    qdd = qdd0

    for i in range(1, m):
        f, fd, fdd = vector_quartic_interpolation(ts[i-1], ts[i], qs[i-1], qs[i], qds[i-1], qds[i], qdd, True)
        qdd = fdd(ts[i])
        funcs.append(f)
        dfuncs.append(fd)
        ddfuncs.append(fdd)

    pf = piecewise_func(funcs, ts)
    pfd = piecewise_func(dfuncs, ts)
    pfdd = piecewise_func(ddfuncs, ts)

    if deriv:
        return pf, pfd, pfdd
    else:
        return pf

def quintic_spline(ts, qs, deriv=False, cost='minjerk'):
    m = qs.shape[0]
    n = qs.shape[1]

    if not isinstance(ts, (np.ndarray,)):
        ts = np.asarray(ts)

    qds = np.zeros_like(qs)
    qdds = np.asarray([[None] * n] * m)
    qds[1:-1] = (qs[2:None] - qs[0:-2]) / (np.full((n, m - 2), ts[2:None] - ts[0:-2]).T)

    funcs = [lambda t: qs[0]]
    dfuncs = [lambda t: qds[0]]
    ddfuncs = [lambda t: qdds[0]]

    for i in range(1, m):
        f, fd, fdd = vector_quintic_interpolation(ts[i - 1], ts[i], qs[i - 1], qs[i], qds[i - 1], qds[i], qdds[i - 1], qdds[i], True, cost)
        funcs.append(f)
        dfuncs.append(fd)
        ddfuncs.append(fdd)

    pf = piecewise_func(funcs, ts)
    pfd = piecewise_func(dfuncs, ts)
    pfdd = piecewise_func(ddfuncs, ts)

    if deriv:
        return pf, pfd, pfdd
    else:
        return pf

def angle_linspace(q0, qf, n, endpoint=True):
    qf = wrap_relative(qf, q0)
    qs = np.linspace(q0, qf, n, endpoint=endpoint)
    return qs

def quat_product(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    q3 = np.array([a1*a2 - b1*b2 - c1*c2 - d1*d2,
                   a1*b2 + b1*a2 + c1*d2 - d1*c2,
                   a1*c2 - b1*d2 + c1*a2 + d1*b2,
                   a1*d2 + b1*c2 - c1*b2 + d1*a2])
    return q3

def quat_conj(q1):
    a, b, c, d = q1
    qconj = np.array([a, -b, -c, -d])
    return qconj

def quat_norm(q1):
    n = np.linalg.norm(q1)
    return n

def quat_inverse(q1):
    qinv = quat_conj(q1) / quat_norm(q1)
    return qinv

def quat_polar(q1):
    a, b, c, d = q1
    n = np.array([b, c, d]) / np.linalg.norm(np.array([b, c, d]))
    varphi = np.arccos(a / quat_norm(q1))
    return n, varphi

def quat_power(q1, x):
    n, varphi = quat_polar(q1)
    q2 = quat_norm(q1)**x * np.append(np.cos(x * varphi), n * np.sin(x * varphi))
    return q2

def quat_ln(q1):
    a1, v1 = q1[0], q1[1:4]
    a2 = np.log(quat_norm(q1))
    v2 = v1 / np.linalg.norm(v1) * np.arccos(a1 / quat_norm(q1))
    q2 = np.array([a2, v2[0], v2[1], v2[2]])
    return q2

def quat_dist(q1, q2):
    d = quat_norm(quat_ln(quat_product(quat_inverse(q1), q2)))
    return d


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    p = 10

    q_knot = np.array([[0, 1, 2, 0, -2, -1]]).T

    t_knot = [0, 1, 2, 3, 4, 5]
    # f, fd, fdd = cubic_spline(t_knot, q_knot, deriv=True)
    f, fd, fdd = quintic_spline(t_knot, q_knot, deriv=True, cost='minjerk')

    ts = np.linspace(t_knot[0], t_knot[-1], 1000)
    qs = np.asarray([f(t) for t in ts])
    qds = np.asarray([fd(t) for t in ts])
    qdds = np.asarray([fdd(t) for t in ts])

    fig, ax = plt.subplots()
    ax.plot(ts, qs, color=[1, 0, 0, 1], ls='-')
    ax.plot(ts, qds, color=[0, 1, 0, 1], ls='-')
    ax.plot(ts, qdds, color=[0, 0, 1, 1], ls='-')
    ax.scatter(t_knot, q_knot)
    plt.axis('equal')
    # ax.legend(['q0', 'qd0', 'qdd0', 'qpoint'])
    plt.show()
