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


if __name__ == '__main__':
    q1 = np.pi + 1
    q2 = -(q1)
    q3 = np.pi - 1
    q4 = -np.pi + 1

    print(w(q1))
    print(w(q2))
    print(w(q3))
    print(w(q4))