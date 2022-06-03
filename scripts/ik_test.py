import RoboPy as rp
import numpy as np
from numpy import pi
import time


dh = [[0, 0, 1, 0],
      [0, 0, 1, pi/2],
      [0, 0, 1, pi/2],
      [0.5, 0, 1, 0]]

n = len(dh)

arm = rp.SerialArm(dh)
# player = rp.ArmPlayer(arm)
viz = rp.VizScene()
viz.add_arm(arm)
viz.add_arm(arm)
viz.add_arm(arm)
viz.add_arm(arm)
viz.wander()
viz.hold()
