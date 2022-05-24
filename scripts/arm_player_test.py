import RoboPy as rp
import numpy as np
import time

arm = rp.PlanarDyn(n=5, L=1)

player = rp.ArmPlayer(arm)
