"""
Dynamics Module - Contains code for:
- Dynamic SerialArm class
- RNE Algorithm
- Euler - Lagrange formulation

John Morrell, Jan 28 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
import Kinematics as kin

eye = np.eye(4)

class SerialArmDyn(kin.SerialArm):
    """
    class representing a serial linkage arm with dynamic
    !!! finish me !!!
    """

    def __init__(self, dh, jt=None, base=eye, tip=eye, joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None):
        kin.SerialArm.__init__(self, dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia

    def RNE(self, q, qd, qdd,
            Wext=np.zeros((6,)),
            g=np.zeros((3,)),
            omega_base=np.zeros((3,)),
            alpha_base=np.zeros((3,)),
            acc_base=np.zeros((3,))):

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

        return tau, Wrench


if __name__ == '__main__':

    from TimingAnalysis.TimingAnalysis import TimingAnalysis as TA
    timer = TA()
    timer.time_in("MAIN")

    r = np.array([-0.5, 0, 0])
    a = 1
    m = 1
    I = np.array([[0, 0, 0],
                  [0, a**2 * m / 12, 0],
                  [0, 0, a**2 * m / 12]])
    link_inertia = [I, I]

    dh = [[0, 0, a, 0],
          [0, 0, a, 0]]

    jt = ['r', 'r']
    link_masses = [m, m]
    r_com = [r, r]
    arm = SerialArmDyn(dh, jt, mass=link_masses, r_com=r_com, link_inertia=link_inertia)

    W = np.array([5, 6, 7, 8, 9, 10])
    g = np.array([10, 20, 30])

    q = [1, 2]
    qd = [3, 4]
    qdd = [5, 6]


    for i in range(100):
        timer.time_in("RNE")
        tau, W = arm.RNE(q, qd, qdd, W, g)
        timer.time_out("RNE")

    print(tau)
    print(W)

    timer.report_all()