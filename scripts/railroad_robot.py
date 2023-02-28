import numpy as np
import RoboPy as rp


dh = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
arm = rp.SerialArm(dh, joint_limits=[(-np.pi, np.pi)] * 4)


class CustomTrackJoint:
    def __init__(self, dh):
        self.r = r

    def custom_track_joint(self, q):
        p = np.array([self.r, 0, 0])
        R = rp.rotz(q)
        p = R @ p
        A = rp.se3(R, p) @ rp.se3(rp.rotx(np.pi / 2) @ rp.rotz(np.pi / 2))
        return A

arm.transforms[0] = custom_track_joint

viz = rp.VizScene()
viz.add_arm(arm)

q0 = [0, 0, 0, 0]
qt = [1, 1.2, 0.2, -0.7]
At = arm.fk(qt)
viz.add_frame(At)

sol = arm.ik(At, q0=q0, method='ccd', maxdel=0.01, viz=viz, mit=np.inf)
print(sol)
viz.hold()
