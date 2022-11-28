from RoboPy import *
import numpy as np
from numpy import pi

viz = VizScene()
rightArm = Baxter('right', 1)
leftArm = Baxter('left', 1)
rightArm.set_qlim_warnings(False)
leftArm.set_qlim_warnings(False)
viz.add_arm(rightArm)
viz.add_arm(leftArm)
viz.wander()
viz.hold()


