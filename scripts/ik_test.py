import RoboPy as rp
import numpy as np
from numpy import pi
import time
from pyqtgraph import opengl as gl
from scipy.optimize import approx_fprime


dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0.5, 0], [0, 0, 0.25, 0], [0, 0, 0.1, 0]]
arm = rp.SerialArm(dh)
q0 = [0] * 5

# arm = rp.Panda(s=1)
# arm.qlim_warning = False
# q0 = [0, 0, 0, 0, 0, 0, 0]

qf = np.random.random((arm.n,)) * 2 * np.pi - np.pi

target = arm.fk(qf)

max_delta = 0.05
max_iter = np.inf

sol = arm.ik(target, q0=q0, method='pinv', maxdel=max_delta, mit=max_iter, retry=5)
print(sol)

viz = rp.VizScene()
viz.add_arm(arm)
# viz.add_marker(pos=target, color=[1, 1, 0, 1], pxMode=False, size=0.15)
viz.add_frame(target)
xs = arm.fk(sol.qs)
viz.window.addItem(gl.GLLinePlotItem(pos=xs[:, 0:3, 3], color=[0, 1, 0, 1]))

while True:
      for q in sol.qs:
            viz.update(q)
            time.sleep(0.01)

      viz.hold(0.5)
