from RoboPy import *
import numpy as np
from numpy import pi

viz = VizScene()
human_arm = HumanArm(0.9, 0.285*3)
viz.add_arm(human_arm)
q = np.array([0, 0, 0, 0, 0, 0, pi/2])
# dq = np.random.random_sample((7,)) * 0.01 - 0.005
dq = np.array([0, 0, 0, 0, 0.001, 0, 0])
while True:
    q = q + dq
    viz.update(q)
viz.hold()