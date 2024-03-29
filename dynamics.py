"""
dynamics Module - Contains code for:
- Dynamic SerialArm class
- RNE Algorithm
- Euler - Lagrange formulation

John Morrell, Jan 28 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
from .kinematics import SerialArm, shift_gamma
from .transforms import transl, se3
from .utility import skew
from itertools import combinations
from scipy.linalg import null_space

eye = np.eye(4)

class SerialArmDyn(SerialArm):
    """
    class representing a serial linkage arm with dynamic
    !!! finish me !!!
    """

    def __init__(self, dh, jt=None, base=eye, tip=eye, joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None,
                 joint_damping=None,
                 torque_scaling=None):

        SerialArm.__init__(self, dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia
        self.T = torque_scaling
        if self.T is None:
            self.T = np.eye(self.n)
        if joint_damping is None:
            self.B = np.zeros((self.n, self.n))
        else:
            self.B = np.diag(joint_damping)

    def rne(self, q, qd, qdd,
            Wext=np.zeros((6,)),
            g=np.zeros((3,)),
            omega_base=np.zeros((3,)),
            alpha_base=np.zeros((3,)),
            acc_base=np.zeros((3,)),
            return_wrench=False):

        omegas = []
        alphas = []
        v_ends = []
        v_coms = []
        acc_ends = []
        acc_coms = []

        forces = ['empty'] * self.n
        moments = ['empty'] * self.n
        b = self.B @ qd
        b = np.outer(b, np.array([0, 0, 1]))
        # print(b)

        Rs = []
        R0s = []
        rp2cs = []
        rp2coms = []
        zaxes = []

        for i in range(self.n):
            T = self.fk(q, [i, i+1])
            R = T[0:3, 0:3]
            p = T[0:3, 3]

            Rs.append(R)
            rp2cs.append(R.T @ p)
            rp2coms.append(R.T @ p + self.r_com[i])
            zaxes.append(Rs[i-1].T[0:3, 2])

            R0 = self.fk(q, i+1)[0:3, 0:3]
            R0s.append(R0)

        for i in range(0, self.n):
            if i == 0:
                omega_previous = omega_base
                alpha_previous = alpha_base
                acc_previous = acc_base
            else:
                omega_previous = omegas[i-1]
                alpha_previous = alphas[i-1]
                acc_previous = Rs[i].T @ acc_ends[i-1]

            omega_current = omega_previous + qd[i] * zaxes[i]
            alpha_current = alpha_previous + qdd[i] * zaxes[i] + np.cross(omega_current, zaxes[i], axis=0) * qd[i]
            acc_com_current = acc_previous + np.cross(alpha_current, rp2coms[i], axis=0) + np.cross(omega_current, np.cross(omega_current, rp2coms[i], axis=0), axis=0)
            acc_end_current = acc_previous + np.cross(alpha_current, rp2cs[i], axis=0) + np.cross(omega_current, np.cross(omega_current, rp2cs[i], axis=0), axis=0)

            omegas.append(omega_current)
            alphas.append(alpha_current)
            acc_coms.append(acc_com_current)
            acc_ends.append(acc_end_current)

        for i in range(self.n-1, -1, -1):

            if i == self.n-1:
                Rn_0 = R0s[i].T
                F_previous = -Rn_0 @ Wext[0:3]
                M_previous = -Rn_0 @ Wext[3:6]
                g_current = Rn_0 @ g
                drag = b[i]
            else:
                Ri_0 = R0s[i].T
                F_previous = Rs[i+1] @ forces[i+1]
                M_previous = Rs[i+1] @ moments[i+1]
                g_current = Ri_0 @ g
                drag = b[i] - b[i + 1]

            F_current = F_previous + self.mass[i] * (acc_coms[i] - g_current)
            dMomentum = self.link_inertia[i] @ alphas[i] + np.cross(omegas[i], self.link_inertia[i] @ omegas[i], axis=0)
            M_current = dMomentum + M_previous + np.cross(self.r_com[i], -F_previous, axis=0) + np.cross(rp2coms[i], F_current, axis=0) + drag

            forces[i] = F_current
            moments[i] = M_current

        tau = np.zeros((self.n,))

        for i in range(self.n):
            if self.jt[i] == 'r':
                tau[i] = zaxes[i] @ moments[i]
            else:
                tau[i] = zaxes[i] @ forces[i]

        # tau = tau - b

        Wrench = np.zeros((6,))
        Wrench[0:3] = -Rs[0] @ forces[0]
        Wrench[3:6] = -Rs[0] @ moments[0]

        if not return_wrench:
            output = tau
        else:
            output = (tau, Wrench)

        return output

    def jacobdot_com(self, q, qd, index, base=False, tip=False):

        if index is None:
            index = self.n
        elif index > self.n:
            print("WARNING: Index greater than number of joints!")
            print(f"Index: {index}")
            return None
        elif index < 1:
            print("WARNING: Index for Jacobdot cannot be less than 1!")
            print(f"Index: {index}")
            return None

        Jdot = np.zeros((6, self.n))
        Ae = self.fk(q, index) @ transl(self.r_com[index-1])
        re = Ae[0:3, 0:3] @ self.r_com[index-1]
        pe = Ae[0:3, 3]
        xde = shift_gamma(re) @ self.jacob(q, index) @ qd
        ve = xde[0:3]

        for i in range(index):
            Ac = self.fk(q, i)
            pc = Ac[0:3, 3]
            zc = Ac[0:3, 2]
            xdc = self.jacob(q, i) @ qd
            vc = xdc[0:3]
            wc = xdc[3:6]
            Jdot[0:3, i] = np.cross(zc, ve - vc) + np.cross(np.cross(wc, zc), pe - pc)
            Jdot[3:6, i] = np.cross(wc, zc)

        return Jdot

    def get_M(self, q):
        M = np.zeros((self.n, self.n))

        for i in range(self.n):
            A = self.fk(q, i+1)
            R = A[0:3, 0:3]
            ri = R @ self.r_com[i]
            J = shift_gamma(ri) @ self.jacob(q, i + 1)
            Jv = J[0:3, :]
            Jw = J[3:6, :]
            M += self.mass[i] * Jv.T @ Jv + Jw.T @ R @ self.link_inertia[i] @ R.T @ Jw

        return M

    def get_C(self, q, qd, get_mdot=False):
        # See paper A new Coriolis matrix factorization
        # Magnus Bjerkeng, member IEEE and Kristin Y. Pettersen, Senior member IEEE
        # for derivation of C matrix

        C = np.zeros((self.n, self.n))
        Mdot = np.zeros((self.n, self.n))
        for i in range(self.n):
            Ac = self.fk(q, i+1) @ transl(self.r_com[i])
            R = Ac[0:3, 0:3]
            rc = R @ self.r_com[i]
            J = shift_gamma(rc) @ self.jacob(q, i+1)
            Jv = J[0:3, :]
            Jw = J[3:6, :]
            Jd = self.jacobdot_com(q, qd, i+1)
            Jvd = Jd[0:3, :]
            Jwd = Jd[3:6, :]
            Rd = skew(Jw @ qd) @ R
            m = self.mass[i]
            In = self.link_inertia[i]
            C = C + m * Jv.T @ Jvd + Jw.T @ R @ In @ R.T @ Jwd + Jw.T @ Rd @ In @ R.T @ Jw

            if get_mdot:
                Ind = R @ In @ Rd.T + Rd @ In @ R.T
                Mdot = Mdot + m * (Jv.T @ Jvd + Jvd.T @ Jv) + Jw.T @ (In @ Jwd + Ind @ Jw) + Jwd.T @ In @ Jw

        if get_mdot:
            output = (C, Mdot)
        else:
            output = C
        return output

    def get_G(self, q, g=np.array([0, 0, 0])):

        G = np.zeros((self.n,))
        for i in range(self.n):
            ri = self.fk(q, i+1)[0:3, 0:3] @ self.r_com[i]
            J = shift_gamma(ri) @ self.jacob(q, i+1)
            G = G - J[0:3, :].T @ g * self.mass[i]

        return G

    def get_KE(self, q, qd):
        M = self.get_M(q)
        return 0.5 * qd.T @ M @ qd

    def get_PE(self, q, g=np.array([0, 0, 0])):
        PE = 0.0
        for i in range(1, self.n + 1):
            PE += self.fk(q, index=i)[0:3, 3] @ g * self.mass[i - 1]
        return PE

    def get_MCG(self, q, qd, g=np.array([0, 0, 0])):

        M = np.zeros((self.n, self.n))
        C = np.zeros_like(M)
        G = np.zeros((self.n,))

        for i in range(self.n):
            A = self.fk(q, i + 1) @ transl(self.r_com[i])
            R = A[0:3, 0:3]
            r = R @ self.r_com[i]
            J = shift_gamma(r) @ self.jacob(q, i + 1)
            Jv = J[0:3, :]
            Jw = J[3:6, :]
            Jd = self.jacobdot_com(q, qd, i + 1)
            Jvd = Jd[0:3, :]
            Jwd = Jd[3:6, :]
            Rd = skew(Jwd @ qd) @ R
            m = self.mass[i]
            In = self.link_inertia[i]

            M = M + m * Jv.T @ Jv + Jw.T @ R @ In @ R.T @ Jw
            C = C + m * Jv.T @ Jvd + Jw.T @ R @ In @ R.T @ Jwd + Jw.T @ Rd @ In @ R.T @ Jw
            G = G - J[0:3, :].T @ g * m

        return M, C, G

    def EL(self, q, qd, qdd, Wext=np.zeros((6,)), g=np.array([0, 0, 0])):

        M, C, G = self.get_MCG(q, qd, g=g)
        b = self.B @ qd
        J = self.jacob(q)
        tau = M @ qdd + C @ qd + G - J.T @ Wext + b
        return tau

    def forward_rne(self, q, qd, tau, g=np.array([0, 0, 0]), Wext=np.zeros((6,))):

        B = self.rne(q, qd, np.zeros((self.n,)), g=g, Wext=Wext)
        M = np.zeros((self.n, self.n))

        for i in range(self.n):
            qdd_fake = np.zeros((self.n,))
            qdd_fake[i] = 1.0
            M[:, i] = self.rne(q, np.zeros((self.n,)), qdd_fake)

        qdd = np.linalg.solve(M, tau - B)
        return qdd

    def forward_EL(self, q, qd, tau, g=np.array([0, 0, 0]), Wext=np.zeros((6,))):

        M, C, G = self.get_MCG(q, qd, g)
        b = self.B @ qd
        J = self.jacob(q)
        qdd = np.linalg.solve(M, tau - C @ qd - G + J.T @ Wext - b)
        return qdd

    def accel_r(self, q, qd=None,
                g=np.array([0.0, 0.0, 0.0]),
                Wext=np.zeros((6,)),
                planar=False,
                base=True,
                tip=True,
                verbose=False):
        """
        Calculate the acceleration radius (the maximum cartesian acceleration possible in any direction) for a given q
        :param q: length n np.ndarray of joint angles in radians
        :param qd: length n np.ndarray of joint velocities in rad / s
        :param g: [0, 0, 0] length 3 np.ndarray representing gravity vector (positive in the down direction)
        :param Wext: [0, 0, 0, 0, 0, 0] length 6 np.ndarray external wrench [F, Tau]^T exerted on the end effector
        :param planar: False, boolean value where true means the task space should be 2-dim, false means 3-dim
        :param base: True, whether to use base transform
        :param tip: True, whether to use tip transform
        :param verbose: False, whether to return potential accelerations
        :return: float r, acceleration radius
        """
        if qd is None:
            qd = np.zeros((self.n,))
        J = self.jacob(q, base=base, tip=tip)
        Jd = self.jacobdot(q, qd, base=base, tip=tip)
        m = 2 if planar else 3
        n = self.n
        J = J[0:m]
        Wext = Wext[0:m]
        Jd = Jd[0:m]
        M, C, G = self.get_MCG(q, qd, g)
        Minv = np.linalg.inv(M)
        Tinv = np.linalg.inv(self.T)

        L = J @ Minv @ self.T
        w = J @ Minv @ (C @ qd + G + J.T @ Wext) + Jd @ qd
        b = Tinv @ (C @ qd + G + J.T @ Wext)

        comb = [x for x in combinations(range(n), m - 1)]
        iterate = range(len(comb))
        k = len(comb)

        rs = np.zeros((k,)) + np.inf
        accs_possible = np.zeros((2 * k, m))

        if np.linalg.norm(b, ord=np.inf) < 1:
            for i in iterate:
                if np.linalg.matrix_rank(L[:, comb[i]]) == m - 1:
                    acc_orth = np.ravel(null_space(L[:, comb[i]].T))
                else:
                    continue

                h = np.linalg.norm(acc_orth)
                acc_orth = acc_orth / h
                c = np.sign(L.T @ acc_orth)
                z1 = acc_orth @ (L @ c + w)
                z2 = acc_orth @ (L @ -c + w)

                z = z1 if np.abs(z1) < np.abs(z2) else z2

                rs[i] = z
                accs_possible[i] = acc_orth * z
                accs_possible[i + k] = acc_orth * z1 if z == z2 else acc_orth * z2

            j = np.argmin(np.abs(rs))
            acc_radius = np.abs(rs[j])

            if verbose:
                return acc_radius, accs_possible[j], accs_possible
            return acc_radius
        else:
            return 1 - np.linalg.norm(b, ord=np.inf)

    def calc_planar_boundary(self, q, qd=None,
                                    g=np.array([0.0, 0.0, 0.0]),
                                    Wext=np.zeros((6,)),
                                    base=True,
                                    tip=True):
        if qd is None:
            qd = np.zeros((self.n,))
        J = self.jacob(q, base=base, tip=tip)
        Jd = self.jacobdot(q, qd, base=base, tip=tip)
        m = 2
        n = self.n
        J = J[0:m]
        Wext = Wext[0:m]
        Jd = Jd[0:m]
        M, C, G = self.get_MCG(q, qd, g)
        Minv = np.linalg.inv(M)
        Tinv = np.linalg.inv(self.T)

        L = J @ Minv @ self.T
        w = J @ Minv @ (C @ qd + G + J.T @ Wext) + Jd @ qd
        b = Tinv @ (C @ qd + G + J.T @ Wext)

        comb = [x for x in combinations(range(n), m - 1)]
        iterate = range(len(comb))
        k = len(comb)

        rs = np.zeros((k,)) + np.inf
        accs_possible = np.zeros((2 * k, m))

        if np.linalg.norm(b, ord=np.inf) < 1:
            for i in iterate:
                if np.linalg.matrix_rank(L[:, comb[i]]) == m - 1:
                    acc_orth = np.ravel(null_space(L[:, comb[i]].T))
                else:
                    continue

                h = np.linalg.norm(acc_orth)
                acc_orth = acc_orth / h
                c = np.sign(L.T @ acc_orth)
                z1 = acc_orth @ (L @ c + w)
                z2 = acc_orth @ (L @ -c + w)

                z = z1 if np.abs(z1) < np.abs(z2) else z2

                rs[i] = z
                accs_possible[i] = acc_orth * z
                accs_possible[i + k] = acc_orth * z1 if z == z2 else acc_orth * z2

            j = np.argmin(np.abs(rs))
            acc_radius = np.abs(rs[j])
        else:
            acc_radius = 1 - np.linalg.norm(b, ord=np.inf)

        return acc_radius
