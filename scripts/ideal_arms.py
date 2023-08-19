import RoboPy as rp
import numpy as np


dh = [[0, 0, 1, 0], [0, 0, 0.1, 0], [0, 0, 0.9, 0]]
jt = ['r', 'r', 'r']
qlim = [(-np.pi, np.pi), (0, 1), (-np.pi, np.pi), (0, 1)]

arm = rp.SerialArm(dh, jt)

q = [0, 1, 0]
J = arm.jacob(q, rep='xy')
print(J)
print(np.sqrt(np.linalg.det(J @ J.T)))

viz = rp.VizScene()
viz.window.setBackgroundColor(255, 255, 255, 100)
viz.grid.setColor([50, 50, 50, 76])
viz.add_arm(arm)
viz.update(q)
viz.hold()
