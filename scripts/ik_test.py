import RoboPy as rp
import numpy as np
from numpy import pi
import time
from pyqtgraph import opengl as gl
from scipy.optimize import approx_fprime


dh = [[0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 0.5, 0],
      [0, 0, 0.25, 0],
      [0, 0, 0.1, 0]]

arm = rp.SerialArm(dh)
arm.qlim_warning = False

q0 = [0, 0, 0, 0, 0]
qt = [np.pi * 3 / 4, -np.pi / 4, np.pi / 4, 0, 0]

target = arm.fk(qt)

def nullq(q, qd):

      def f(q):
            return np.linalg.norm(arm.fk(q)[0:3, 3] - target[0:3, 3])

      qdstar = np.zeros((arm.n,))
      qdstar = approx_fprime(q, f)
      qdstar = qdstar / max(np.linalg.norm(qdstar), 0.01) * np.linalg.norm(qd)
      return qdstar

max_delta = 0.1
max_iter = 5000

sol1 = arm.ik(target, q0, method='pinv', rep='cart', max_iter=max_iter, max_delta=max_delta, nullq=None)
sol2 = arm.ik(target, q0, method='pinv', rep='cart', max_iter=max_iter, max_delta=max_delta, nullq=nullq)

viz = rp.VizScene()
viz.add_arm(arm)
viz.add_arm(arm)
viz.add_marker(pos=target[0:3, 3], color=[1, 0, 0, 1], size=0.1, pxMode=False)

qs1 = sol1.qs
qs2 = sol2.qs

if qs1.shape[0] > qs2.shape[0]:
      qs2 = np.vstack([qs2, np.full((qs1.shape[0] - qs2.shape[0], arm.n), qs2[-1])])

if qs2.shape[0] > qs1.shape[0]:
      qs1 = np.vstack([qs1, np.full((qs2.shape[0] - qs1.shape[0], arm.n), qs1[-1])])

xs1 = np.array([arm.fk(q, rep='cart') for q in qs1])
xs2 = np.array([arm.fk(q, rep='cart') for q in qs2])

viz.add_marker(pos=xs1, size=0.05, color=[0, 0, 1, 1], pxMode=False)
viz.add_marker(pos=xs2, size=0.05, color=[0, 1, 0, 1], pxMode=False)

viz.window.addItem(gl.GLLinePlotItem(pos=xs1, color=[0, 0, 1, 1]))
viz.window.addItem(gl.GLLinePlotItem(pos=xs2, color=[0, 1, 0, 1]))

print(sol1)
print(sol2)

while True:
      for q1, q2 in zip(qs1, qs2):
            viz.update([q1, q2])
            time.sleep(0.001)
      viz.hold(2.0)
