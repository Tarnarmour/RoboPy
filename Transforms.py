"""
Transforms Module - Contains code for:
- representing SO2, SO3, and SE3 objects as numpy arrays
- converting between different representations of those objects,
- euler angles, quaternions, axis angle, etc.

John Morrell, Jan 28 2022
Tarnarmour@gmail.com
https://github.com/Tarnarmour/RoboPy.git
"""

import numpy as np
from numpy import sin, cos, sqrt
from numpy.linalg import norm

def rot2(th):
    return np.array([[cos(th), -sin(th)], [sin(th), cos(th)]], dtype=np.float32)

def rotx(th):
    return np.array([[1, 0, 0],
                     [0, cos(th), -sin(th)],
                     [0, sin(th), cos(th)]], dtype=np.float32)

def roty(th):
    return np.array([[cos(th), 0, sin(th)],
                     [0, 1, 0],
                     [-sin(th), 0, cos(th)]], dtype=np.float32)

def rotz(th):
    return np.array([[cos(th), -sin(th), 0],
                     [sin(th), cos(th), 0],
                     [0, 0, 1]], dtype=np.float32)

def R2rpy(R):
    return np.array([np.arctan2(R[1, 0], R[0, 0]),
                     np.arctan2(-R[2, 0], sqrt(R[2, 1]**2 + R[2, 2]**2)),
                     np.arctan2(R[2, 1], R[2, 2])])

def rpy2R(rpy):
    return rotx(rpy[0]) @ roty(rpy[1]) @ rotz(rpy[2])

def R2euler(R, order='xyz'):

    D = dict(x=(rotx, 0), y=(roty, 1), z=(rotz, 2))

    rotA, axis1 = D[order[0]]
    rotB, axis2 = D[order[1]]
    rotC, axis3 = D[order[2]]

    if axis1 >= axis3:
        s = -1
    else:
        s = 1

    Ri = np.eye(3)
    Rf = R

    v = np.cross(Rf[:, axis3], (s * Ri[:, axis1]))
    if norm(v) < 0.001:  # This indicates a rotation about the A axis ONLY.
        th1 = np.arccos(Ri[:, axis2] @ (Rf[:, axis2]))
        th2 = 0
        th3 = 0
        Ri = Ri @ rotA(th1)
    else:
        v = v / norm(v)
        th1 = np.arccos(Ri[:, axis2] @ v)
        Ri = Ri @ rotA(th1)

        th2 = np.arccos(Ri[:, axis3] @ Rf[:, axis3])
        Ri = Ri @ rotB(th2)

        th3 = np.arccos(Ri[:, axis2] @ Rf[:, axis2])
        Ri = Ri @ rotC(th3)

    return np.array([th1, th2, th3])


def euler2R(th1, th2, th3, order='xyz'):

    if order == 'xyx':
        R = rotx(th1) @ roty(th2) @ rotx(th3)
    elif order == 'xyz':
        R = rotx(th1) @ roty(th2) @ rotz(th3)
    elif order == 'xzx':
        R = rotx(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'xzy':
        R = rotx(th1) @ rotz(th2) @ roty(th3)
    elif order == 'yxy':
        R = roty(th1) @ rotx(th2) @ roty(th3)
    elif order == 'yxz':
        R = roty(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'yzx':
        R = roty(th1) @ rotz(th2) @ rotx(th3)
    elif order == 'yzy':
        R = roty(th1) @ rotz(th2) @ roty(th3)
    elif order == 'zxy':
        R = rotz(th1) @ rotx(th2) @ roty(th3)
    elif order == 'zxz':
        R = rotz(th1) @ rotx(th2) @ rotz(th3)
    elif order == 'zyx':
        R = rotz(th1) @ roty(th2) @ rotx(th3)
    elif order == 'zyz':
        R = rotz(th1) @ roty(th2) @ rotz(th3)

    return R

def R2q(R):
    return np.array([0.5*sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1),
                     0.5*np.sign(R[2, 1] - R[1, 2]) * sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1),
                     0.5*np.sign(R[0, 2] - R[2, 0]) * sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1),
                     0.5*np.sign(R[2, 1] - R[1, 2]) * sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)])

def q2R(q):
    nu = q[0]
    ex = q[1]
    ey = q[2]
    ez = q[3]
    return np.array([[2*(nu**2+ex**2)-1, 2*(ex*ey-nu*ez), 2*(ex*ez+nu*ey)],
                    [2*(ex*ey+nu*ez), 2*(nu**2+ey**2)-1, 2*(ey*ez-nu*ex)],
                    [2*(ex*ez-nu*ey), 2*(ey*ez+nu*ex), 2*(nu**2+ez**2)-1]])

def R2axis(R):
    ang = np.arccos(0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1))
    return np.array([ang,
                     (R[2, 1] - R[1, 2]) / (2 * sin(ang)),
                     (R[2, 0] - R[2, 0]) / (2 * sin(ang)),
                     (R[1, 0] - R[0, 1]) / (2 * sin(ang))])

def axis2R(ang, rx, ry, rz):
    c = cos(ang)
    s = sin(ang)
    return np.array([[rx**2 * (1-c) + c, rx*ry*(1-c)-rz*s, rx*rz*(1-c)+ry*s],
                     [rx*ry*(1-c)+rz*s, ry**2 * (1-c) + c, ry*rz*(1-c)-rx*s],
                     [rx*rz*(1-c)-ry*s, ry*rz*(1-c)+rx*s, rz**2 * (1-c) + c]])

if __name__ == '__main__':
    PI = np.pi
    R = rotx(1) @ roty(2) @ rotx(3)
    print(R2euler(R, 'xyx'))

