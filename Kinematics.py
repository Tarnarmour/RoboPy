"""
Kinematics Module - Contains code for:
- creating transforms from DH parameters
- SerialArm class, representing serial link robot arms
- Forward kinematics, Jacobian calculations, Inverse Kinematics

John Morrell, Jan 26 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
import sympy as sp
import mpmath as mp
import copy

eye = np.eye(4, dtype=np.float32)


class TForm:

    def __init__(self, dh, jt):

        if jt == 'r':
            def f(q):
                d = dh[0]
                theta = dh[1] + q
                a = dh[2]
                alpha = dh[3]

                return np.array(
                    [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]],
                    dtype=np.float32)
        else:
            def f(q):
                d = dh[0] + q
                theta = dh[1]
                a = dh[2]
                alpha = dh[3]

                return np.array(
                    [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]],
                    dtype=np.float32)

        self.f = f


class SerialArm:
    """
    SerialArm - A class designed to represent a serial link robot arm
    !!!! FINISH ME !!!!!

    SerialArms have frames 0 to n defined, with frame 0 located at the first joint and aligned with the robot body
    frame, and frame n located at the end of link n.

    """

    def __init__(self, dh, jt=None, base=eye, tip=eye, joint_limits=None):
        """
        arm = SerialArm(dh, joint_type, base=I, tip=I, radians=True, joint_limits=None)
        :param dh: n length list or iterable of length 4 list or iterables representing dh parameters, [d theta a alpha]
        :param jt: n length list or iterable of strings, 'r' for revolute joint and 'p' for prismatic joint
        :param base: 4x4 numpy or sympy array representing SE3 transform from world frame to frame 0
        :param tip: 4x4 numpy or sympy array representing SE3 transform from frame n to tool frame
        :param joint_limits: 2 length list of n length lists, holding first negative joint limit then positive, none for
        not implemented
        """
        self.dh = dh
        self.n = len(dh)
        self.transforms = []
        if jt is None:
            self.jt = ['r'] * self.n
        else:
            self.jt = jt
        for i in range(self.n):
            T = TForm(dh[i], self.jt[i])
            self.transforms.append(T.f)

        self.base = base
        self.tip = tip

    def __str__(self):
        # MUST DO!
        return("not here yet")

    def __repr__(self):
        # MUST DO!
        return("not here yet")

    def fk(self, q, index=None, base=False, tip=False):

        if self.n == 1 and not isinstance(q, (list, tuple)):
            q = [q]

        if isinstance(index, (list, tuple)):
            start_frame = index[0]
            end_frame = index[1]
        elif index == None:
            start_frame = 0
            end_frame = self.n
        else:
            start_frame = 0
            if index < 0:
                print("WARNING: Index less than 0!")
                print(f"Index: {index}")
                return None
            end_frame = index

        if end_frame > self.n:
            print("WARNING: Ending index greater than number of joints!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None
        if start_frame < 0:
            print("WARNING: Starting index less than 0!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None
        if start_frame > end_frame:
            print("WARNING: starting frame must be less than ending frame!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None

        if base and start_frame == 0:
            A = self.base
        else:
            A = eye

        for i in range(start_frame, end_frame):
            A = A @ self.transforms[i](q[i])

        if tip and end_frame == self.n:
            A = A @ self.tip

        return A

    def jacob(self, q, index=None, base=False, tip=False):

        if index is None:
            index = self.n
        elif index > self.n:
            print("WARNING: Index greater than number of joints!")
            print(f"Index: {index}")

        J = np.zeros((6, self.n), dtype=np.float32)
        Te = self.fk(q, index, base=base, tip=tip)
        pe = Te[0:3, 3]

        for i in range(index):
            if self.jt[i] == 'r':
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                p = T[0:3, 3]
                J[0:3, i] = np.cross(z_axis, pe - p, axis=0)
                J[3:6, i] = z_axis
            else:
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                J[0:3, i] = z_axis
                J[3:6, i] = np.zeros_like(z_axis)

        return J

    def jacoba(self, q, index, pose='rpy'):
        print("implement me!")
        return None


if __name__ == "__main__":

    dh = [[0, 0, 0.1, 0], [0, 0, 5, 0]]
    arm = SerialArm(dh)
    print(arm.fk([0, 0]))
