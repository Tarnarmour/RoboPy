"""
Kinematics Module - Contains code for:
- creating transforms from DH parameters
- SerialArm class, representing serial link robot arms
- Forward kinematics, Jacobian calculations, Inverse Kinematics

John Morrell, Jan 26 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""
import Visualization
import numpy as np
import sympy as sp
import mpmath as mp
from Transforms import *
from Utility import *
from Visualization import PlanarMPL

eye = np.eye(4, dtype=np.float32)
pi = np.pi


class TForm:
    def __init__(self, dh, jt):

        if jt == 'r':
            def f(q):
                d = dh[0]
                theta = dh[1] + q
                a = dh[2]
                alpha = dh[3]

                cth = np.cos(theta)
                sth = np.sin(theta)
                cal = np.cos(alpha)
                sal = np.sin(alpha)

                return np.array(
                    [[cth, -sth * cal, sth *sal, a * cth],
                     [sth, cth * cal, -cth * sal, a * sth],
                     [0, sal, cal, d],
                     [0, 0, 0, 1]],
                    dtype=np.float32)
        else:
            def f(q):
                d = dh[0] + q
                theta = dh[1]
                a = dh[2]
                alpha = dh[3]

                cth = np.cos(theta)
                sth = np.sin(theta)
                cal = np.cos(alpha)
                sal = np.sin(alpha)

                return np.array(
                    [[cth, -sth * cal, sth * sal, a * cth],
                     [sth, cth * cal, -cth * sal, a * sth],
                     [0, sal, cal, d],
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
        self.reach = 0
        for i in range(self.n):
            self.reach += np.sqrt(self.dh[i][0]**2 + self.dh[i][2]**2)

    def __str__(self):
        dh_string = """DH PARAMS\n"""
        dh_string += """d\t|\tth\t|\ta\t|\tal\t|\tJT\n"""
        dh_string += """---------------------------------------\n"""
        for i in range(self.n):
            dh_string += f"{self.dh[i][0]}\t|\t{self.dh[i][1]}\t|\t{self.dh[i][2]}\t|\t{self.dh[i][3]}\t|\t{self.jt[i]}\n"
        return "Serial Arm\n" + dh_string

    def __repr__(self):
        return(f"SerialArm(dh=" + repr(self.dh) + ", jt=" + repr(self.jt) + ", base=" + repr(self.base) + ", tip=" + repr(self.tip) + ")")

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

    def jacoba(self, q, rep='rpy', index=None, eps=1e-4):

        if rep == 'rpy':
            def get_pose(q):
                return A2rpy(self.fk(q, index))
        elif rep == 'planar':
            def get_pose(q):
                return A2planar(self.fk(q, index))
        elif rep == 'cart':
            def get_pose(q):
                return self.fk(q, index)[0:3, 3]
        elif rep == 'axis':
            def get_pose(q):
                return A2axis(self.fk(q, index))
        elif rep == 'q' or rep == 'quaternion':
            def get_pose(q):
                return A2q(self.fk(q, index))
        elif rep == 'x':
            def get_pose(q):
                return A2x(self.fk(q, index))
        else:
            def get_pose(q):
                return arm.fk(q)[0:2, 3]

        x0 = get_pose(q)
        m = x0.shape[0]
        J = np.zeros((m, self.n), dtype=np.float32)

        if not isinstance(q, np.ndarray):
            q = np.array(q, dtype=data_type)

        for i in range(self.n):
            q0 = np.zeros_like(q)
            q0[i] += eps
            J[:, i] = (get_pose(q + q0) - get_pose(q - q0)) / (2 * eps)

        return J

    def ik(self, A_target, q0=None, method='pinv', rep='planar', max_iter=100, tol=1e-3, viz=None, min_delta=1e-5):

        if q0 is None:
            q0 = np.zeros((self.n,), dtype=data_type)

        if rep == 'rpy':
            def get_pose(A):
                return A2rpy(A)
        elif rep == 'planar':
            def get_pose(A):
                return A2planar(A)
        elif rep == 'cart':
            def get_pose(A):
                return A[0:3, 3]
        elif rep == 'axis':
            def get_pose(A):
                return A2axis(A)
        elif rep == 'q' or rep == 'quaternion':
            def get_pose(A):
                return A2q(A)
        elif rep == 'x':
            def get_pose(A):
                return A2x(A)
        else:
            def get_pose(A):
                return A[0:2, 3]

        if method == 'pinv':
            def get_qd(q, e):
                J = self.jacoba(q, rep=rep)
                Jdag = np.linalg.pinv(J)
                qd = -Jdag @ e
                return qd

        elif method == 'jt':
            def get_qd(q, e):
                J = self.jacoba(q, rep=rep)
                qd = -J.T @ e * 0.15
                return qd

        x_target = get_pose(A_target)
        x0 = get_pose(arm.fk(q0))

        e = x0 - x_target
        q = q0
        count = 0

        status = 'Success'
        report = 'Successfully converged'

        while norm(e) > tol:
            count += 1
            qd = get_qd(q, e)
            # qd = qd / norm(qd) * norm(e) / 50
            qd = qd / norm(qd) / 50
            while norm(get_pose(arm.fk(q + qd)) - x_target) > norm(e) and norm(qd) > 1e-6:
                qd = qd * 0.5

            q = q + qd
            q = wrap_angle(q)
            x = get_pose(arm.fk(q))
            viz.update(q)

            e = x - x_target

            if count > max_iter:
                status = 'Failure'
                report = 'Did not converge within max_iter limit'
                break
            elif norm(qd) < min_delta:
                status = 'Failure'
                report = 'Terminated because change in q below minimum epsilon'
                break

        print(f"Status: {status}\nFinal Error Norm: {norm(e)}\nIter: {count}\n{report}")
        return q


if __name__ == "__main__":
    np.set_printoptions(precision=4, floatmode='maxprec', suppress=True)
    pi = np.pi
    dh = [[0, 0, 1.0, pi/4], [0, 0, 1.0, pi/4], [0, 0, 0.25, pi/4],[0, 0, 0.25, 0]]
    q0 = [0, 0, 0, 0]
    arm = SerialArm(dh)
    A_target = se3(rotz(np.pi/2), [1, 1, 1])

    viz = Visualization.ArmViz(arm, q0)
    viz.addScatter(A_target[0:3, 3], np.array([0, 1, 0, 1]), 10)
    qt = arm.ik(A_target, q0, 'pinv', 'cart', 500, viz=viz)

    viz.app.exec_()
