import RoboPy as rp
import numpy as np
import time


viz = rp.VizScene()
A = np.eye(4, dtype=np.float64)
viz.add_frame(A, label='0')

for i in range(1000):
    t = i / 100
    A[0:3, 0:3] = rp.rotz(t) @ rp.rotx(t) @ rp.roty(t)
    A[0:3, 3] = [np.sin(t), np.cos(t), np.sin(t) * np.cos(t)]
    viz.update(As=A)
    time.sleep(0.01)

viz.close()