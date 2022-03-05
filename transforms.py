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

from .utility import *

data_type = np.float64

def rot2(th):
    R = np.array([[cos(th), -sin(th)], [sin(th), cos(th)]], dtype=data_type)
    return clean_rotation_matrix(R)

def rot2theta(R):
    return np.arctan2(R[1, 0], R[0, 0])

def rotx(th):
    R = np.array([[1, 0, 0],
                     [0, cos(th), -sin(th)],
                     [0, sin(th), cos(th)]], dtype=data_type)
    return clean_rotation_matrix(R)

def roty(th):
    R = np.array([[cos(th), 0, sin(th)],
                     [0, 1, 0],
                     [-sin(th), 0, cos(th)]], dtype=data_type)
    return clean_rotation_matrix(R)

def rotz(th):
    R = np.array([[cos(th), -sin(th), 0],
                     [sin(th), cos(th), 0],
                     [0, 0, 1]], dtype=data_type)
    return clean_rotation_matrix(R)

def R2rpy(R):
    return np.array([np.arctan2(R[1, 0], R[0, 0]),
                     np.arctan2(-R[2, 0], sqrt(R[2, 1]**2 + R[2, 2]**2)),
                     np.arctan2(R[2, 1], R[2, 2])])

def rpy2R(rpy):
    R = rotx(rpy[0]) @ roty(rpy[1]) @ rotz(rpy[2])
    return clean_rotation_matrix(R)

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
        th1 = np.arccos(min(max(Ri[:, axis2] @ v, 1), 0))
        Ri = Ri @ rotA(th1)

        th2 = np.arccos(min(max(Ri[:, axis3] @ Rf[:, axis3], 1), 0))
        Ri = Ri @ rotB(th2)

        th3 = np.arccos(min(max(Ri[:, axis2] @ Rf[:, axis2], 1), 0))
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

    return clean_rotation_matrix(R)

def R2q(R):
    return np.array([0.5*sqrt(np.abs(R[0, 0] + R[1, 1] + R[2, 2] + 1)),
                     0.5*np.sign(R[2, 1] - R[1, 2]) * sqrt(np.abs(R[0, 0] - R[1, 1] - R[2, 2] + 1)),
                     0.5*np.sign(R[0, 2] - R[2, 0]) * sqrt(np.abs(R[1, 1] - R[2, 2] - R[0, 0] + 1)),
                     0.5*np.sign(R[2, 1] - R[1, 2]) * sqrt(np.abs(R[2, 2] - R[0, 0] - R[1, 1] + 1))])

def q2R(q):
    nu = q[0]
    ex = q[1]
    ey = q[2]
    ez = q[3]
    R =  np.array([[2*(nu**2+ex**2)-1, 2*(ex*ey-nu*ez), 2*(ex*ez+nu*ey)],
                    [2*(ex*ey+nu*ez), 2*(nu**2+ey**2)-1, 2*(ey*ez-nu*ex)],
                    [2*(ex*ez-nu*ey), 2*(ey*ez+nu*ex), 2*(nu**2+ez**2)-1]])
    return clean_rotation_matrix(R)

def R2axis(R):
    ang = np.arccos(0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1))
    return np.array([ang,
                     (R[2, 1] - R[1, 2]) / (2 * sin(ang)),
                     (R[2, 0] - R[2, 0]) / (2 * sin(ang)),
                     (R[1, 0] - R[0, 1]) / (2 * sin(ang))])

def axis2R(ang, v):
    v = v / norm(v)
    V = skew(v)
    R = np.eye(3, dtype=data_type) + sin(ang) * V + (1 - cos(ang)) * V @ V
    return clean_rotation_matrix(R)

def se3(R=np.eye(3, dtype=data_type), p=np.array([0, 0, 0], dtype=data_type)):
    A = np.eye(4, dtype=data_type)
    if isinstance(R, (list, tuple)):
        R = np.array(R)
    A[0:3, 0:3] = R
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    A[0:3, 3] = p
    return clean_rotation_matrix(A)

def se2(R=np.eye(2), p=np.array([0, 0])):
    A = np.eye(2, dtype=data_type)
    if isinstance(R, (list, tuple)):
        R = np.array(R)
    A[0:2, 0:2] = R
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    A[0:2, 2] = p
    return A

def transl(p):
    return se3(p=p)

def inv(A):
    m = T.shape[0]
    n = T.shape[1]
    Ainv = np.zeros_like(A)
    if m == 3:
        R = A[0:2, 0:2].T
        p = R.T @ - A[0:2, 2]
        return se2(R, p)
    else:
        R = A[0:3, 0:3].T
        p = R.T @ - A[0:3, 3]
        return se3(R, p)

def A2rpy(A):
    return np.hstack((A[0:3, 3], R2rpy(A[0:3, 0:3])))

def A2planar(A):
    return np.hstack((A[0:2, 3], rot2theta(A[0:2, 0:2])))

def A2axis(A):
    return np.hstack((A[0:3, 3], R2axis(A[0:3, 0:3])))

def A2q(A):
    return np.hstack((A[0:3, 3], R2q(A[0:3, 0:3])))

def A2x(A):
    return np.hstack((A[0:3, 3], np.reshape(A[np.ix_([0,1,2], [0,2])], 6, 'F')))

def A2cart(A):
    return A[0:3, 3]

def A2pose(A, pose='q'):
    def f(x):
        return {
            'rpy': A2rpy(A),
            'planar': A2planar(A),
            'axis': A2axis(A),
            'q': A2q(A),
            'quat': A2q(A),
            'x': A2x(A),
            'cart': A2cart(A)
        }.get(pose, A2q(A))
    return f(pose)

def Erpy(x):
    spsi = sin(x[0])
    cpsi = cos(x[0])
    sth = sin(x[1])
    cth = cos(x[1])
    E = np.array([[0, -spsi, cth*cpsi],
                  [0, cpsi, cth*spsi],
                  [1, 0, -sth]])
    return E

def invErpy(x):
    spsi = sin(x[0])
    cpsi = cos(x[0])
    sth = sin(x[1])
    cth = cos(x[1])
    if np.abs(np.abs(np.mod(x[1], np.pi)) - np.pi) < 1e-6:
        E = np.array([[0, 0, 0.5],
                      [-spsi, cpsi, 0],
                      [0, 0, 0.5]])
    else:
        E = np.array([[cpsi*sth/cth, sth*spsi/cth, 1],
                      [-spsi, cpsi, 0],
                      [cpsi/cth, spsi/cth, 0]])
    return E

def Eaxis(x):
    E = np.hstack((x[1:4,None], sin(x[0])*np.eye(3) + (1-cos(x[0]))*skew(x[1:4])))
    return E

def invEaxis(x):
    E = np.vstack((x[1:4], -sin(x[0]) / 2 / (1 - cos(x[0])) * skew(x[1:4])**2 - skew(x[1:4]) / 2))
    return E

def Equat(x):
    H = np.hstack((-x[1:4, None], skew(x[1:4]) + x[0]*np.eye(3)))
    return 2 * H

def invEquat(x):
    H = np.hstack((-x[1:4, None], skew(x[1:4]) + x[0] * np.eye(3)))
    return H.T / 2

def padE(E):
    m = E.shape[0]
    n = E.shape[1]
    return np.block([[np.eye(3), np.zeros((3,n))],
                     [np.zeros((m,3)), E]])
