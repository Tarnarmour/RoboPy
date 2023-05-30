import RoboPy as rp
import numpy as np


dh = [[0, 0, 1, 0], [0, 0, 1, 0]]
arm = rp.SimpleDynArm(dh)
sim = rp.SingleArmSimulation(arm)
viz = rp.SimViz(arm)

def my_forces(insim: rp.SingleArmSimulation):
    def f_tau(q, qd, t):
        Kp = np.diag([20.0, 30.0])
        Kd = np.diag([10.0, 5.0])
        tau = Kp @ (np.array([np.sin(5 * t), np.cos(5 * t)]) - q) + Kd @ (np.array([np.cos(5 * t), -np.sin(5 * t)]) - qd)
        return tau

    return f_tau


sim.simulate(qd0=np.array([2 * np.pi, 20.0]), tstop=np.inf, dt=0.01, realtime=True, callback=viz.callback_function, get_forces=my_forces, gravity=np.array([10, 0, 0]))

