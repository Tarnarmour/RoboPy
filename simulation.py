"""
simulation module - contains code for:
- Creating dynamic system representations of robot arms
- Executing simulation using RK4, SciPy odeint, etc
- Generating linearized models of arms
- Input respones, etc.

John Morrell, May 2 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
import scipy
from scipy.integrate import odeint
from time import perf_counter, sleep

from .kinematics import SerialArm, shift_gamma
from .dynamics import SerialArmDyn
from .transforms import *
from .utility import *
from .premade_arms import PlanarDyn

default_arm = PlanarDyn()


class RobotSys:
    """RobotSys is a class representing a robot arm as a dynamical system"""

    def __init__(self, arm=default_arm, dt=1e-3, step_method='rk4', dyn_method='rne'):

        self.n = arm.n
        self.arm = arm
        self.dt = dt

        self.g = np.array([0, 0, -9.81])

        self.x = np.zeros((self.n * 2,))
        self.t = 0.0


        if step_method in ['rk4', 'RK4', 'Rk4', 'rK4']:
            self.step = self.rk4_step
        elif step_method in ['scipy', 'SciPy', 'odeint', 'ode45', 'ODE45']:
            self.step = self.odeint_step
        elif step_method in ['euler', 'Euler']:
            self.step = self.euler_step
        else:
            self.step = self.rk4_step

        self.dyn_method = dyn_method

    def eom(self, x, t, f, g):

        q, qd = x[0:self.n], x[self.n:None]
        tau = f(t, x)
        Wext = g(t, x)

        tau

        if self.dyn_method == 'rne':
            qdd = self.arm.forward_rne(q, qd, tau, g=self.g, Wext=Wext)
        elif self.dyn_method == 'EL':
            qdd = self.arm.forward_EL(q, qd, tau, g=self.g, Wext=Wext)
        else:
            print("Invalid forward kinematics method, ", self.dyn_method, "!")
            return None

        xd = np.zeros_like(x)
        xd[0:self.n] = qd
        xd[self.n:None] = qdd

        return xd

    def rk4_step(self, f, g):

        x0 = np.copy(self.x)
        t0 = self.t
        t1 = self.t + self.dt / 2
        t2 = self.t + self.dt / 2
        t3 = self.t + self.dt

        f1 = self.eom(x0, t0, f, g)
        f2 = self.eom(x0 + f1 * self.dt / 2, t1, f, g)
        f3 = self.eom(x0 + f2 * self.dt / 2, t2, f, g)
        f4 = self.eom(x0 + f3 * self.dt, t3, f, g)

        self.x = self.x + self.dt / 6 * (f1 + 2*f2 + 2*f3 + f4)

    def euler_step(self, f, g):

        x0 = np.copy(self.x)
        dx = self.eom(x0, self.t, f, g)
        self.x = self.x + dx * self.dt

    def odeint_step(self, f, g):

        # ts = np.arange(tspan[0], tspan[1], self.dt)
        tspan = [self.t, self.t + self.dt]
        x0 = np.copy(self.x)
        func = lambda x, t: self.eom(x, t, f, g)

        yout = odeint(func, x0, tspan)
        self.x = yout[-1, :]

    def simulate(self, tspan, q0, qd0, tau, gravity=np.zeros((3,)), Wext=None,
                 viz=None,
                 max_iter=np.inf,
                 real_time=False):

        self.g = gravity

        # tau should be a function of t and x so that tau = f(t, x)
        if not callable(tau):
            f = lambda t, x: tau
        else:
            f = tau

        # Wext should be a function
        if Wext is None:
            g = lambda t, x: np.zeros((6,))
        elif not callable(Wext):
            g = lambda t, x: Wext
        else:
            g = Wext

        if not tspan[1] - self.dt > tspan[0]:
            print("tspan must be a pair of numbers with tspan[1] - self.dt > tspan[0]")
            return None
        self.t = tspan[0]

        if len(q0) != self.n or len(qd0) != self.n:
            print(f"q0 or qd0 not the right length! q0 = {q0}, qd0 = {qd0}")
            return None
        self.x = np.hstack([q0, qd0])

        ts = np.array([], dtype=np.float)
        xs = np.zeros((0, self.n * 2))

        count = 0
        if real_time:
            twall = perf_counter()
            tstart = twall
            tlast = 0.0

        while self.t < tspan[1] and count < max_iter:

            self.step(f, g)
            self.x[0:self.n] = wrap_angle(self.x[0:self.n])

            ts = np.append(ts, self.t)
            xs = np.vstack([xs, self.x])

            if not viz is None:
                q = self.x[0:self.n]
                viz.update(q)

            self.t = self.t + self.dt
            count += 1

            if real_time:
                tcur = perf_counter()
                if tcur < twall + self.dt:
                    sleep(twall + self.dt - tcur)
                    if np.floor(tcur - tstart) > tlast:
                        tlast = np.floor(tcur - tstart)
                        print(f"t = {tlast}")
                else:
                    if np.floor(tcur - tstart) > tlast:
                        tlast = np.floor(tcur - tstart)
                        print(f"t = {tlast}: Behind Real Time!")
                twall = perf_counter()


        qs = xs[:, 0:self.n]
        qds = xs[:, self.n:None]

        return ts, qs, qds


