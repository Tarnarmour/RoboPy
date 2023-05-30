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
from dataclasses import dataclass
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from time import sleep, perf_counter

from .dynamics import SerialArmDyn


@dataclass
class SimulationOutput:
    ts: np.ndarray
    qs: np.ndarray
    qds: np.ndarray
    qdds: np.ndarray

class SingleArmSimulation:
    def __init__(self, arm: SerialArmDyn):
        self.n = arm.n
        self.arm = arm
        self.q = np.zeros((self.n,))
        self.qd = np.zeros((self.n,))
        self.qdd = np.zeros((self.n,))
        self.t = 0.0
        self.g = np.array([0.0, 0.0, 0.0])

    def simulate(self, q0=None, qd0 = None, tstop=np.inf, dt=0.01, realtime=False, callback=None, get_forces=None, gravity=np.zeros((3,))):
        if q0 is None:
            self.q = np.zeros((self.n,))
        elif isinstance(q0, (list, tuple)):
            self.q = np.asarray(q0)
        else:
            self.q = q0

        if qd0 is None:
            self.qd = np.zeros((self.n,))
        elif isinstance(qd0, (list, tuple)):
            self.qd = np.asarray(qd0)
        else:
            self.qd = qd0

        if callback is None:
            callback = self.callback_default

        if get_forces is None:
            get_forces = self.get_forces_default

        self.g = gravity

        output = SimulationOutput(np.zeros((0,)),
                                   np.zeros((0, self.n)),
                                   np.zeros((0, self.n)),
                                   np.zeros((0, self.n)))

        self.t = 0.0
        k = 0
        while self.t < tstop:
            if realtime:
                tick = perf_counter()

            f_tau = get_forces(self)
            self.step(dt, f_tau)
            self.update_history(output)
            callback(self, output)
            self.t = self.t + dt
            k += 1

            if realtime:
                real_dt = perf_counter() - tick
                if real_dt < dt:
                    sleep(dt - real_dt)

        return output

    def callback(self, *args, **kwargs):
        pass

    @staticmethod
    def get_forces_default(sim, *args, **kwargs):
        f_tau = lambda *args: np.zeros((sim.n))
        return f_tau

    @staticmethod
    def callback_default(sim, *args, **kwargs):
        pass

    def step(self, dt, f_tau):
        q = self.q + self.qd * dt + self.qdd * 0.5 * dt * dt
        qdd = self.arm.forward_EL(q, self.qd, tau=f_tau(q, self.qd, self.t), g=self.g)
        qd = self.qd + (qdd + self.qdd) * 0.5 * dt

        self.q = q
        self.qd = qd
        self.qdd = qdd

    def update_history(self, output: SimulationOutput):
        output.ts = np.hstack([output.ts, self.t])
        output.qs = np.vstack([output.qs, self.q])
        output.qds = np.vstack([output.qds, self.qd])
        output.qdds = np.vstack([output.qdds, self.qdd])
