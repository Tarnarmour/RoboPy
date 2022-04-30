import numpy as np
import mpmath as mp

def wrap_angle(q):
    def f(q):
        q = np.mod(q, 2 * np.pi)
        if q > np.pi:
            q -= 2 * np.pi
        return q

    if isinstance(q, (np.ndarray, list, tuple)):
        for i in range(len(q)):
            q[i] = f(q[i])
    else:
        q = f(q)
    return q

def wrap_relative(q1, q2):
    if hasattr(q1, '__iter__'):
        if hasattr(q2, '__iter__'):
            for i, (r, w) in enumerate(zip(q1, q2)):
                while r - w > np.pi:
                    r = r - 2 * np.pi
                while r - w < -np.pi:
                    r = r + 2 * np.pi
                q1[i] = r
        else:
            for i, r in enumerate(q1):
                while r - q2 > np.pi:
                    r = r - 2 * np.pi
                while r - q2 < -np.pi:
                    r = r + 2 * np.pi
                q1[i] = r
    else:
        while q1 - q2 > np.pi:
            q1 = q1 - 2 * np.pi
        while q1 - q2 < -np.pi:
            q1 = q1 + 2 * np.pi
    return q1


def skew(v):
    if hasattr(v[0], '__len__'):
        print("Input to skew(v) must be 1 dimensional!")
        return None
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def clean_rotation_matrix(R, eps=1e-12):
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.abs(R[i, j]) < eps:
                R[i, j] = 0.
            elif np.abs(R[i, j] - 1) < eps:
                R[i, j] = 1.
    return R

def mprint(A, p=3, eps=1e-12):
    if isinstance(A, (np.ndarray)):
        if A.ndim == 2:
            with np.printoptions(precision=p, suppress=True, floatmode='fixed', linewidth=200):
                print(clean_rotation_matrix(A, eps))
                return
        elif A.ndim > 2:
            print('[ ')
            for i in range(len(A)):
                mprint(A[i], p)
                if i < len(A) - 1:
                    print(', ')
            print(']\n')
        else:
            with np.printoptions(precision=p, suppress=True, floatmode='fixed', linewidth=200):
                print(A)
                return
    elif hasattr(A, '__len__'):
        print('[ ')
        for i in range(len(A)):
            mprint(A[i], p)
            if i < len(A) - 1:
                print(', ')
        print(']\n')
    else:
        with np.printoptions(precision=p, suppress=True, floatmode='fixed'):
            print(A)
            return

if __name__ == '__main__':
    q1 = np.pi + 1
    q2 = -(q1)
    q3 = np.pi - 1
    q4 = -np.pi + 1

    print(w(q1))
    print(w(q2))
    print(w(q3))
    print(w(q4))