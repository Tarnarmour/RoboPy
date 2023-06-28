import timeit
import RoboPy as rp
import numpy as np

arm = rp.SerialArm(dh=[[0, 0, 1, 0], [0, 0, 1, 6], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

q = [1, 2, 3, 4, 5]
N = 1000


oldfk = timeit.repeat("[arm.fk(q, index=i, base=True, tip=False) for i in range(6)]", globals=globals(), number=N, repeat=10)
newfk = timeit.repeat("[arm.fk2(q, index=i, base=True, tip=False) for i in range(6)]", globals=globals(), number=N, repeat=10)

print(min(oldfk) / N)
print(min(newfk) / N)
print(min(oldfk) / min(newfk))

print(arm._fk.cache_info())
