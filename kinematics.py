"""
Kinematics Module - Contains code for:
- creating transforms from DH parameters
- SerialArm class, representing serial link robot arms
- Forward kinematics, Jacobian calculations, Inverse Kinematics

John Morrell, Jan 26 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

from .utility import *
from .transforms import *
from dataclasses import dataclass
import scipy.optimize as optimize

eye = np.eye(4, dtype=np.float32)
pi = np.pi


class DH2Func:
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
        :param joint_limits: n length list of 2 length lists, holding first negative joint limit then positive, none for
        not implemented, as in [[-2, 2], [-2, 2], [-3.14, 1.56]]
        """
        self.dh = dh
        self.n = len(dh)
        self.transforms = []
        if jt is None:
            self.jt = ['r'] * self.n
        else:
            self.jt = jt
            if len(self.jt) != self.n:
                raise ValueError("WARNING! Joint Type list does not have the same size as dh param list!")
        for i in range(self.n):
            T = DH2Func(dh[i], self.jt[i])
            self.transforms.append(T)

        self.base = base
        self.tip = tip
        self.reach = 0
        for i in range(self.n):
            if self.jt[i] != 'c':  # don't attempt for custom joint types
                self.reach += np.sqrt(self.dh[i][0]**2 + self.dh[i][2]**2)
        # self.reach = self.reach * 1.1

        if joint_limits is None:
            joint_limits = None
        else:
            if not len(joint_limits) == self.n:
                raise ValueError("WARNING! Joint limits list does not have the same size as dh param list!")
            if not isinstance(joint_limits, np.ndarray):
                joint_limits = np.asarray(joint_limits, dtype=data_type)

        self.qlim = joint_limits

        self.qlim_warning = False

    def __str__(self):
        dh_string = """DH PARAMS\n"""
        dh_string += """d\t|\tth\t|\ta\t|\tal\t|\tJT\n"""
        dh_string += """---------------------------------------\n"""
        for i in range(self.n):
            dh_string += f"{self.dh[i][0]}\t|\t{self.dh[i][1]}\t|\t{self.dh[i][2]}\t|\t{self.dh[i][3]}\t|\t{self.jt[i]}\n"
        return "Serial Arm\n" + dh_string

    def __repr__(self):
        return(f"SerialArm(dh=" + repr(self.dh) + ", jt=" + repr(self.jt) + ", base=" + repr(self.base) + ", tip=" + repr(self.tip) + ", joint_limits=" + repr(self.qlim) + ")")

    def set_qlim_warnings(self, warnings_on):
        self.qlim_warning = bool(warnings_on)

    def clipq(self, q):
        if self.qlim is not None:
            q_out = np.clip(q, self.qlim[:, 0], self.qlim[:, 1])
            changed = not (q == q_out).all()
        else:
            q_out = q
            changed = False
        return q_out, changed

    def randq(self, p=1):
        qs = np.random.random((p, self.n))
        if self.qlim is not None:
            A = np.diag(self.qlim[:, 1] - self.qlim[:, 0])
            qs = qs @ A + self.qlim[:, 0]
        else:
            qs = (np.random.random((p, self.n)) * 2 - 1) * np.pi
        return qs

    def fk(self, q, index=None, base=False, tip=False, rep=None):

        # handle input: we want to accept any iterable or scalar and turn it into an np.array
        if not isinstance(q, np.ndarray):
            if hasattr(q, '__getitem__'):
                q = np.asarray(q, dtype=data_type)
            else:
                q = np.array([q], dtype=data_type)

        # If q is a 2D numpy array, assume each row is a set of q's and do this
        if len(q.shape) == 2:
            output_shape = self.fk(q[0], index, base, tip, rep).shape
            output = np.zeros(((q.shape[0],) + output_shape))
            for i, q_in in enumerate(q):
                output[i] = self.fk(q_in, index, base, tip, rep)
            return output

        if len(q) != self.n:
            raise ValueError("WARNING: q (input angle) not the same size as number of links!")
            return None

        q, clipped = self.clipq(q)
        if clipped and self.qlim_warning:
            raise ValueError("WARNING! Joint input to fk out of joint limits!")

        if isinstance(index, (list, tuple)):
            start_frame = index[0]
            end_frame = index[1]
        elif index == None:
            start_frame = 0
            end_frame = self.n
        else:
            start_frame = 0
            if index < 0:
                raise ValueError("WARNING: Index less than 0!")
                print(f"Index: {index}")
                return None
            end_frame = index

        if end_frame > self.n:
            raise ValueError("WARNING: Ending index greater than number of joints!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None
        if start_frame < 0:
            raise ValueError("WARNING: Starting index less than 0!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None
        if start_frame > end_frame:
            raise ValueError("WARNING: starting frame must be less than ending frame!")
            print(f"Starting frame: {start_frame}  Ending frame: {end_frame}")
            return None

        if base and start_frame == 0:
            A = self.base
        else:
            A = eye

        for i in range(start_frame, end_frame):
            A = A @ self.transforms[i].f(q[i])
        if rep is None:
            return A
        else:
            return A2pose(A, rep)

    def fk_com(self, q, base=False):
        """
        Find the center of mass, assuming weightless motors and links with constant linear density. Always returned
        with respect to frame 0 or world frame (if base is true). Only cartesian values [x y z]
        """
        # handle input: we want to accept any iterable or scalar and turn it into an np.array
        if not isinstance(q, np.ndarray):
            if hasattr(q, '__getitem__'):
                q = np.asarray(q, dtype=data_type)
            else:
                q = np.array([q], dtype=data_type)

        # If q is a 2D numpy array, assume each row is a set of q's and do this
        if len(q.shape) == 2:
            output_shape = self.fk_com(q[0], base).shape
            output = np.zeros(((q.shape[0],) + output_shape))
            for i, q_in in enumerate(q):
                output[i] = self.fk_com(q_in, base)
            return output

        if len(q) != self.n:
            raise ValueError("WARNING: q (input angle) not the same size as number of links!")
            return None

        q, clipped = self.clipq(q)
        if clipped and self.qlim_warning:
            raise ValueError("WARNING! Joint input to fk out of joint limits!")

        p_com = np.zeros((3,))
        p_prev = self.base[0:3, 3] if base else np.zeros((3,))
        mass = 0

        for i in range(self.n):
            p_cur = self.fk(q, i + 1, base=base, rep='cart')
            p_com += 0.5 * (p_cur + p_prev) * np.linalg.norm(p_cur - p_prev)  # location of com multiplied by weight, assuming straight line link between i-1 and i and constant linear density
            mass += np.linalg.norm(p_cur - p_prev)
            p_prev = p_cur

        p_com = p_com / mass
        return p_com

    def jacob(self, q, index=None, base=False, tip=False, rep='full'):
        # handle input: we want to accept any iterable or scalar and turn it into an np.array
        if not isinstance(q, np.ndarray):
            if hasattr(q, '__getitem__'):
                q = np.asarray(q, dtype=data_type)
            else:
                q = np.array([q], dtype=data_type)

        # If q is a 2D numpy array, assume each row is a set of q's and do this
        if len(q.shape) == 2:
            output_shape = self.jacob(q[0], index, base, tip).shape
            output = np.zeros(((q.shape[0],) + output_shape))
            for i, q_in in enumerate(q):
                output[i] = self.jacob(q_in, index, base, tip)
            return output

        if len(q) != self.n:
            raise ValueError("WARNING: q (input angle) not the same size as number of links!")
            return None

        q, clipped = self.clipq(q)
        if clipped and self.qlim_warning:
            raise ValueError("WARNING! Joint input to jacob out of joint limits!")

        if index is None:
            index = self.n
        elif index > self.n:
            raise ValueError("WARNING: Index greater than number of joints!")
            print(f"Index: {index}")

        J = np.zeros((6, self.n), dtype=data_type)
        Te = self.fk(q, index, base=base, tip=tip)
        pe = Te[0:3, 3]

        for i in range(index):
            if self.jt[i] == 'r':
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                p = T[0:3, 3]
                J[0:3, i] = np.cross(z_axis, pe - p, axis=0)
                J[3:6, i] = z_axis
            elif self.jt[i] == 'p':
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                J[0:3, i] = z_axis
                J[3:6, i] = np.zeros_like(z_axis)
            elif self.jt[i] == 'c':  # custom joint type, use predefined partials stored in self.Ts
                pass
            else:
                raise ValueError("WARNING: Unknown joint type!")
                print(f"Joint type for joint {i} is '{self.jt[i]}'")

        if rep == 'cart':
            J = J[0:3]
        elif rep == 'xy':
            J = J[0:2]
        elif rep == 'planar':
            J = J[[0, 1, 5]]
        else:
            pass

        return J

    def jacob_com(self, q, base=False):
        """
        Find the jacobian of the center of mass, assuming weightless joints and constant linear density. Always returns
        [x y z] with respect to either world frame or base frame
        :param q:
        :param base:
        :return:
        """
        # handle input: we want to accept any iterable or scalar and turn it into an np.array
        if not isinstance(q, np.ndarray):
            if hasattr(q, '__getitem__'):
                q = np.asarray(q, dtype=data_type)
            else:
                q = np.array([q], dtype=data_type)

        # If q is a 2D numpy array, assume each row is a set of q's and do this
        if len(q.shape) == 2:
            output_shape = self.jacob_com(q[0], base).shape
            output = np.zeros(((q.shape[0],) + output_shape))
            for i, q_in in enumerate(q):
                output[i] = self.jacob_com(q_in, base)
            return output

        if len(q) != self.n:
            raise ValueError("WARNING: q (input angle) not the same size as number of links!")

        q, clipped = self.clipq(q)
        if clipped and self.qlim_warning:
            raise ValueError("WARNING! Joint input to jacob out of joint limits!")

        J = np.zeros((6, self.n), dtype=data_type)
        mass = 0

        p_prev = self.base[0:3, 3] if base else np.zeros((3,))

        for i in range(self.n):
            p_cur = self.fk(q, i + 1, base=base, rep='cart')
            r_com = 0.5 * (p_prev - p_cur)
            J += shift_gamma(r_com) @ self.jacob(q, i + 1) * np.linalg.norm(p_prev - p_cur)
            mass += np.linalg.norm(p_prev - p_cur)

        J = J[0:3] / mass

        return J

    def jacobdot(self, q, qd, index=None, base=False, tip=False, rep=None):

        if index is None:
            index = self.n
        elif index > self.n:
            raise ValueError(f"WARNING: Index greater than number of joints!\nIndex = {index}, n = {self.n}")

        Jdot = np.zeros((6, self.n), dtype=np.float32)
        pe = self.fk(q, index, base=base, tip=tip, rep='cart')
        Je = self.jacob(q, index, base, tip)
        ve = Je[0:3, :] @ qd

        for i in range(index):
            if self.jt[i] == 'r':
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                pc = T[0:3, 3]
                xd = self.jacob(q, i, base, tip) @ qd
                vi = xd[0:3]
                wi = xd[3:6]
                Jdot[0:3, i] = np.cross(z_axis, ve - vi) + np.cross(np.cross(wi, z_axis), pe - pc)
                Jdot[3:6, i] = np.cross(xd[3:6], z_axis)
            else:
                Ji = self.jacob(q, i, base, tip)
                z_axis = Ji[0:3, i]
                wi = (Ji @ qd)[3:6]
                Jdot[0:3, i] = np.cross(wi, z_axis)

        return Jdot

    def jacoba(self, q, rep='rpy', index=None, base=False, tip=False, eps=1e-6):

        if rep == 'rpy':
            def get_pose(q):
                return A2rpy(self.fk(q, index, base=base, tip=tip))
        elif rep == 'planar':
            def get_pose(q):
                return A2planar(self.fk(q, index, base=base, tip=tip))
        elif rep == 'cart':
            def get_pose(q):
                return self.fk(q, index, base=base, tip=tip)[0:3, 3]
        elif rep == 'axis':
            def get_pose(q):
                return A2axis(self.fk(q, index, base=base, tip=tip))
        elif rep == 'q' or rep == 'quaternion' or rep == 'quat':
            return padE(invEquat(A2q(self.fk(q, index, base=base, tip=tip))[3:7])) @ self.jacob(q, index, base=base, tip=tip)
            def get_pose(q):
                return A2q(self.fk(q, index, base=base, tip=tip))
        elif rep == 'x':
            def get_pose(q):
                return A2x(self.fk(q, index, base=base, tip=tip))
        else:
            def get_pose(q):
                return arm.fk(q, index, base=base, tip=tip)[0:2, 3]

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

    def hessian(self, q, index=None, base=False, tip=False):
        if not isinstance(q, np.ndarray):
            if hasattr(q, '__getitem__'):
                q = np.asarray(q, dtype=data_type)
            else:
                q = np.array([q], dtype=data_type)

        # If q is a 2D numpy array, assume each row is a set of q's and do this
        if len(q.shape) == 2:
            output_shape = self.jacob(q[0], index, base, tip).shape
            output = np.zeros(((q.shape[0],) + output_shape))
            for i, q_in in enumerate(q):
                output[i] = self.jacob(q_in, index, base, tip)
            return output

        if len(q) != self.n:
            raise ValueError("WARNING: q (input angle) not the same size as number of links!")
            return None

        q, clipped = self.clipq(q)
        if clipped and self.qlim_warning:
            raise ValueError("WARNING! Joint input to jacob out of joint limits!")

        if index is None:
            index = self.n
        elif index > self.n:
            raise ValueError("WARNING: Index greater than number of joints!")
            print(f"Index: {index}")

        H = np.zeros((6, self.n, self.n))
        J = self.jacob(q, base=base, tip=tip)
        Ae = self.fk(q)
        pe = Ae[0:3, 3]

        for j in range(self.n):
            Jj = self.jacob(q, index=j, base=base, tip=tip)
            Aj = self.fk(q, index=j, base=base, tip=tip)
            zj = Aj[0:3, 2]
            pj = pe - Aj[0:3, 3]
            for k in range(self.n):
                zk = Jj[3:6, k]
                delz = np.cross(zk, zj)
                delv = J[0:3, k] - Jj[0:3, k]
                H[0:3, j, k] = np.cross(delz, pj) + np.cross(zj, delv)
                H[3:6, j, k] = delz

        return H

    def ik(self, target, q0=None, method='pinv', rep=None, tol=1e-4, mit=1000, maxdel=2*np.pi, mindel=1e-6, force=False, retry=0, viz=None, base=False, tip=False, **kwargs):
        """
        Wrapper for general IK function
        :param target: iterable or np.ndarray[size: (4, 4)], can be [x y], [x y z], [x y theta] (use rep='planar'), 4x4 transform, [x y z r p y] or [x y z q0 q1 q2 q3]
        :param q0: iterable, initial joint position, defaults to [0] * self.n
        :param method: string, method to use. ['pinv', 'jt', 'SLSQP', 'CCD', 'fabrik', 'hessian']
        :param tol: float, stopping tolerance of norm(error)
        :param mit: int, maximum iteration
        :param maxdel: float, maximum norm of qdot for single iteration
        :param mindel: float, minimum norm of qdot before termination
        :param force: bool, attempt even when target is outside of arm naive reach test
        :param retry: int, how many times to retry with random starting position if failed
        :param kwargs: additional key word arguments to pass to specific methods
        :return: sol, IKOutput object
        """

        # handle target argument
        """
        four cases, full 3D pose (x y z q0 q1 q2 q3), 3D cartesian (x y z), planar (x y theta) and planar cartesian (x y)
        """
        if not isinstance(target, np.ndarray):
            target = np.array(target, dtype=float)

        if len(target.shape) == 2: # treat target as a 4x4 homogeneous transform
            quat = R2q(target[0:3, 0:3])
            target = np.hstack([target[0:3, 3], quat])
            getx = lambda q: A2q(self.fk(q, base=base, tip=tip))
            getJ = lambda q: self.jacoba(q, rep='q', base=base, tip=tip)
            dist = np.linalg.norm(target[0:3] - self.base[0:3, 3])
        elif target.shape[0] == 2:  # treat as a x-y target
            getx = lambda q: self.fk(q, base=base, tip=tip)[0:2, 3]
            getJ = lambda q: self.jacob(q, base=base, tip=tip)[0:2]
            dist = np.linalg.norm(self.base[0:2, 3] - target)
        elif target.shape[0] == 3:
            if rep is None or rep == 'cart' or rep == 'xyz':  # assume xyz
                getx = lambda q: self.fk(q, base=base, tip=tip)[0:3, 3]
                getJ = lambda q: self.jacob(q, base=base, tip=tip)[0:3]
                dist = np.linalg.norm(target - self.base[0:3, 3])
            elif rep == 'planar':
                def getx(q):
                    A = self.fk(q, base=base, tip=tip)
                    theta = rot2theta(A[0:2, 0:2])
                    return np.array([A[0, 3], A[1, 3], theta])
                def getJ(q):
                    return self.jacob(q)[[0, 1, 5]]
                dist = np.linalg.norm(target[0:2] - self.base[0:2, 3])
        elif target.shape[0] == 6:  # rpy input
            quat = R2q(rpy2R(target[3:6]))
            target = np.hstack([target[0:3], quat])
            getx = lambda q: A2q(self.fk(q, base=base, tip=tip))
            getJ = lambda q: self.jacoba(q, rep='q', base=base, tip=tip)
            dist = np.linalg.norm(target[0:3] - self.base[0:3, 3])
        elif target.shape[0] == 7:  # quat input
            getx = lambda q: A2q(self.fk(q, base=base, tip=tip))
            getJ = lambda q: self.jacoba(q, rep='q', base=base, tip=tip)
            dist = np.linalg.norm(target[0:3] - self.base[0:3, 3])

        if q0 is None:
            q0 = np.array([0] * self.n)
        elif not isinstance(q0, np.ndarray):
            q0 = np.array(q0)

        # if not force:
        #     if dist > self.reach + tol:
        #         raise ValueError('Target fails naive reach test!')

        if method == 'pinv':
            if 'K' not in kwargs.keys():
                K = np.eye(len(target))
            else:
                K = kwargs['K']
                if not isinstance(K, np.ndarray):
                    K = K * np.eye(len(target))
            sol = ik_pinv(target, getx, getJ, q0, tol, mit, maxdel, mindel, retry, viz, K)
        elif method == 'jt':
            if 'K' not in kwargs.keys():
                K = np.eye(len(target)) * 0.15
            else:
                K = kwargs['K']
                if not isinstance(K, np.ndarray):
                    K = K * np.eye(len(target))
            if 'Kd' not in kwargs.keys():
                Kd = np.eye(len(target)) * 0.1
            else:
                Kd = kwargs['Kd']
                if not isinstance(Kd, np.ndarray):
                    Kd = Kd * np.eye(len(target))
            sol = ik_jt(target, getx, getJ, q0, tol, mit, maxdel, mindel, retry, viz, K, Kd)
        elif method == 'CCD' or method == 'ccd':
            sol = ik_ccd(target, getx, q0, tol, mit, maxdel, mindel, retry, viz)
        elif method == 'scipy':
            getH = lambda q: self.hessian(q, base=base, tip=tip)[:, 0:3, :]
            if 'opt' not in kwargs.keys():
                opt = 'BFGS'
            else:
                opt = kwargs['opt']
            sol = ik_scipy(target, getx, getJ, getH, q0, tol, mit, maxdel, mindel, retry, viz, opt)

        return sol


def shift_gamma(*args):
    gamma = np.eye(6)
    for x in args:
        if len(x.shape) == 1:
            # shifting by an offset
            gamma_new = np.block([[np.eye(3), -skew(x)], [np.zeros((3,3)), np.eye(3)]])
        else:
            # rotation of jacobian
            gamma_new = np.block([[x.T, np.zeros((3,3))], [np.zeros((3,3)), x.T]])
        gamma = gamma @ gamma_new
    return gamma

"""
All IK method functions can trust getting some specific types of inputs, which will be handled in SerialArm.ik

target: specific shape for the target
getx: function of q that returns position in the same form as target
getJ: function of q that gives analytic jacobian of getx

These are set up in such a way that the error = ||target - getx(qc)||
"""
def ik_pinv(target, getx, getJ, q0, tol, mit, maxdel, mindel, retry, viz, K):

    qs = [q0]
    qc = q0
    xc = getx(qc)
    xt = target
    nit = 0
    e = xt - xc

    while np.linalg.norm(e) > tol and nit < mit:
        J = getJ(qc)
        # Jdag = np.linalg.pinv(J)
        Jdag = J.T @ np.linalg.inv(J @ J.T + np.eye(len(target)) * 1e-8)
        qd = Jdag @ K @ e

        qdnorm = np.linalg.norm(qd)

        if qdnorm > maxdel:
            qd = qd / qdnorm * maxdel

        while np.linalg.norm(getx(qc + qd) - target) > np.linalg.norm(e) and qdnorm > mindel * 5:
            qd = qd * 0.75

        qdnorm = np.linalg.norm(qd)

        if qdnorm < mindel:
            break

        qc = qc + qd
        wrap_angle(qc)
        qs.append(qc)
        nit += 1

        if viz is not None:
            viz.update(qc)

        xc = getx(qc)
        e = xt - xc

    en = np.linalg.norm(e)
    if en <= tol:
        status = True
        message = 'Successfully converged'
    else:
        status = False
        if nit == mit:
            message = 'Failed to converge within maximum iteration limit'
        elif qdnorm < mindel:
            message = 'Failed to converge, qdot less than minimum'

    if retry > 0 and not status:
        q0 = np.random.random((len(q0),)) * 2 * np.pi - np.pi
        output = ik_pinv(target, getx, getJ, q0, tol, mit, maxdel, mindel, retry - 1, viz, K)
    else:
        output = IKOutput(qc, qs, e, en, nit, status, message)

    return output


def ik_jt(target, getx, getJ, q0, tol, mit, maxdel, mindel, retry, viz, K, Kd):
    qs = [q0]
    qc = q0
    xc = getx(qc)
    xt = target
    nit = 0
    e = xt - xc
    de = np.zeros_like(e)

    if maxdel == np.inf:
        maxdel = 2 * np.pi

    while np.linalg.norm(e) > tol and nit < mit:
        J = getJ(qc)
        qd = J.T @ K @ e - J.T @ Kd @ de

        qdnorm = np.linalg.norm(qd)

        def f(alpha):
            qdot = qd / qdnorm * alpha
            qopt = qc + qdot
            return np.linalg.norm(target - getx(qopt))**2

        alpha = optimize.minimize_scalar(f, bounds=(-maxdel, maxdel), method='bounded', options={'xatol':tol}).x
        qd = qd / qdnorm * alpha
        qdnorm = np.abs(alpha)

        if qdnorm < mindel:
            break

        qc = qc + qd
        qs.append(qc)
        nit += 1

        if viz is not None:
            viz.update(qc)

        xc = getx(qc)
        de = e - (xt - xc)
        e = xt - xc

    en = np.linalg.norm(e)
    if en <= tol:
        status = True
        message = 'Successfully converged'
    else:
        status = False
        if nit == mit:
            message = 'Failed to converge within maximum iteration limit'
        elif qdnorm < mindel:
            message = 'Failed to converge, qdot less than minimum'

    if retry > 0 and not status:
        q0 = np.random.random((len(q0),)) * 2 * np.pi - np.pi
        output = ik_jt(target, getx, getJ, q0, tol, mit, maxdel, mindel, retry - 1, viz, K, Kd)
    else:
        output = IKOutput(qc, qs, e, en, nit, status, message)

    return output


def ik_ccd(target, getx, q0, tol, mit, maxdel, mindel, retry, viz):
    qs = [q0]
    qc = q0
    xc = getx(qc)
    xt = target
    nit = 0
    e = xt - xc

    index = 0
    n = len(q0)

    if maxdel == np.inf:
        maxdel = 10.0

    while np.linalg.norm(e) > tol and nit < mit:

        def func(a):
            q = np.copy(qc)
            q[index] += a
            return np.linalg.norm(target - getx(q))**2

        a = optimize.minimize_scalar(func, bounds=(-maxdel, maxdel), method='bounded', tol=tol).x
        qdnorm = np.abs(a)
        qd = np.zeros((n,))
        qd[index] = a
        qc = qc + qd
        qc = wrap_angle(qc)
        qs.append(qc)
        nit += 1
        e = target - getx(qc)

        if viz is not None:
            viz.update(qc)

        index += 1
        if index == n:
            index = 0

    en = np.linalg.norm(e)
    if en <= tol:
        status = True
        message = 'Successfully converged'
    else:
        status = False
        if nit == mit:
            message = 'Failed to converge within maximum iteration limit'
        elif qdnorm < mindel:
            message = 'Failed to converge, qdot less than minimum'

    if retry > 0 and not status:
        q0 = np.random.random((len(q0),)) * 2 * np.pi - np.pi
        output = ik_ccd(target, getx, q0, tol, mit, maxdel, mindel, retry - 1, viz)
    else:
        output = IKOutput(qc, qs, e, en, nit, status, message)

    return output


def ik_scipy(target, getx, getJ, getH, q0, tol, mit, maxdel, mindel, retry, viz, opt):

    def fun(q):
        e = target - getx(q)
        output = 0.5 * e @ e
        return output

    def jac(q):
        J = getJ(q)
        e = target - getx(q)
        output = -e @ J
        return output

    def hess(q):
        J = getJ(q)
        H = getH(q)
        e = target - getx(q)
        output = J.T @ J - e @ H
        return output

    bounds = [(-np.pi, np.pi)] * len(q0)
    qs = [q0]

    def callback(qk, state=None):
        if viz is not None:
            viz.update(qk)
        qs.append(qk)

    options = {'maxiter':mit}

    sol = optimize.minimize(fun, q0, jac=jac, hess=hess, callback=callback, tol=tol / 1000, options=options, method=opt)
    qf = sol.x
    qs.append(qf)
    nit = sol.nit
    ef = target - getx(qf)
    en = np.linalg.norm(ef)
    status = np.linalg.norm(ef) <= tol
    message = sol.message
    if not status and retry > 0:
        return ik_scipy(target, getx, getJ, np.random.random((len(q0),)) * np.pi * 2 - np.pi, tol, mit, maxdel, mindel, retry - 1, viz)
    else:
        return IKOutput(qf, qs, ef, en, nit, status, message)


class IKOutput:
    def __init__(self, qf, qs, ef, en, nit, status, message):
        self.qf = qf
        self.qs = qs
        self.ef = ef
        self.en = en
        self.nit = nit
        self.status = status
        self.message = message

    def __str__(self):
        output = f"IK Solution:\nSuccess: {self.status}, with output message: {self.message}\n"
        output = output + f"Final q: {self.qf}\nFinal Error Norm: {self.en}\nIterations: {self.nit}\n"
        return output