import RoboPy as rp
import numpy as np
import time

arm = rp.PlanarDyn(n=5, L=1)

player = rp.VizScene()
player.add_arm(arm)
player.add_frame(arm.fk(np.zeros((5,))))
player.hold()
