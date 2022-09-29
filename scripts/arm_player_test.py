import RoboPy as rp
import numpy as np
import time

arm = rp.SerialArm([[0, 0, 1, 2], [1, 0, 1, 0]], ['r', 'p'])
player = rp.ArmPlayer(arm)

