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
                 motor_inertia=None):

        SerialArm.__init__(self, dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia

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
            else:
                Ri_0 = R0s[i].T
                F_previous = Rs[i+1] @ forces[i+1]
                M_previous = Rs[i+1] @ moments[i+1]
                g_current = Ri_0 @ g

            F_current = F_previous + self.mass[i] * (acc_coms[i] - g_current)
            dMomentum = self.link_inertia[i] @ alphas[i] + np.cross(omegas[i], self.link_inertia[i] @ omegas[i], axis=0)
            M_current = dMomentum + M_previous + np.cross(self.r_com[i], -F_previous, axis=0) + np.cross(rp2coms[i], F_current, axis=0)

            forces[i] = F_current
            moments[i] = M_current

        tau = np.zeros((self.n,))

        for i in range(self.n):
            if self.jt[i] == 'r':
                tau[i] = zaxes[i].T @ moments[i]
            else:
                tau[i] = zaxes[i].T @ forces[i]

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

            Ind = R @ In @ Rd.T + Rd @ In @ R.T
            Mdot = Mdot + m * (Jv.T @ Jvd + Jvd.T @ Jv) + Jw.T @ (In @ Jwd + Ind @ Jw) + Jwd.T @ In @ Jw

        if get_mdot:
            output = (C, Mdot)
        else:
            output = C
        return output

    def get_G(self, q, g=np.array([0, 0, -9.81])):

        G = np.zeros((self.n,))
        for i in range(self.n):
            ri = self.fk(q, i+1)[0:3, 0:3] @ self.r_com[i]
            J = shift_gamma(ri) @ self.jacob(q, i+1)
            G = G - J.T @ np.hstack((g, np.zeros((3,))))

        return G

    def get_MCG(self, q, qd, g=np.array([0, 0, -9.81])):

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
            G = G - J.T @ np.hstack((g, np.zeros(3, )))

        return M, C, G

    def EL(self, q, qd, qdd, g=np.zeros([0, 0, 0]), Wext=np.zeros((6,))):

        M, C, G = self.get_MCG(q, qd, g=np.array([0, 0, -9.81]))
        J = self.jacob(q)
        tau = M @ qdd + C @ qd + G - J.T @ Wext
        return tau

    def forward_rne(self, q, qd, tau, g=np.array([0, 0, -9.81]), Wext=np.zeros((6,))):

        B = self.rne(q, qd, np.zeros((self.n,)), g=g, Wext=Wext)
        M = np.zeros((self.n, self.n))

        for i in range(self.n):
            qdd_fake = np.zeros((self.n,))
            qdd_fake[i] = 1.0
            M[:, i] = self.rne(q, np.zeros((self.n,)), qdd_fake)

        qdd = np.linalg.solve(M, tau - B)
        return qdd

    def forward_EL(self, q, qd, tau, g=np.array([0, 0, -9.81]), Wext=np.zeros((6,))):

        M, C, G = self.get_MCG(q, qd, g)
        J = self.jacob(q)
        qdd = np.linalg.solve(M, tau - C @ qd - G + J.T @ Wext)
        return qdd
