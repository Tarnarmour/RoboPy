import RoboPy as rp
import numpy as np
import time

arm = rp.panda()
# player = rp.ArmPlayer(arm)

player = rp.VizScene()
player.add_arm(arm)
player.add_frame(arm.fk(np.zeros((arm.n,))))
player.wander()
player.hold()
