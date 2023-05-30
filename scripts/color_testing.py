import RoboPy as rp
import numpy as np


viz = rp.VizScene()
viz.window.setBackgroundColor(255, 255, 255, 255)
viz.window.removeItem(viz.grid)

dh = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
arm = rp.SerialArm(dh)
viz.add_arm(arm, ee_color=[0.5, 0.5, 0.5, 0], link_colors=[0.5, 0.5, 0.5, 1], joint_colors=[0.1, 0.1, 0.1, 1])

viz.hold()
