import RoboPy as rp
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm


"""
Goal is to test the IK algorithms and compare them, to see what improves them and what makes them worse.

Requirements for methods:
arm.ik argument list
target: representation of the target, flexible formatting (iterable cast to numpy array), and different options available
including 4x4, cartesian, length 7 cart-quaternion, length 6 cart-rpy, etc.
rep: optional, if you need to specify how the target is being represented
q0: initial q values, optional
tol: tolerance on norm of final error
mit: maximum iteration
mdel: maximum delta, max change in joint position for a single iteration
method: string, which of the algorithms to use
force: try even when target is out of naive reach, returning closest solution
retry: integer specifying how often to retry the solution when failure to converge, with random starting position
**kwargs: arguments to pass to the specific internal algorithm

Each IK method should accept a target (which could be represented as either a cartesian position, a homogeneous transform,
a dual quaternion or a 7 element xyz-quaternion, etc.), some general parameters like tolerance and maximum iterations,
and optionally a starting position.

Each method should output an IK solution object, which is a dataclass holding the following properties:
qf: final q value
qs: list of q values for each iteration
ef: error at final step
en: norm of error at final step
nit: number of major iterations
status: boolean for successful convergene
message: string describing exit status

Testing:
The algorithms will be tested on the following:
find a list of all common arms (PUMA, Panda, Baxter, Stanford, SCARA, etc.) and make each one 3 times at different scales
3 scales of two link and three link planar arms
3 scales of 10 DOF arm

For each arm, targets will be selected from the reachable workspace using joint space sampling (static seed). Appropriate
target representation will be used. 5 tests will start at singular configurations, and 5 tests will end at singular 
configurations. The rest of the 40 tests will have randomly generated starting and ending positions.

Scoring:
Each algorithm will get back a score for the percentage of successful trials and the total time required for each type of
arm. So the final report will look like this:

PUMA: % finished, avg. wall time
PANDA: % finished, avg. wall time
etc.

Additionally, a final score will be reported giving the total time and total percentage success
"""

two_link_arms = []
dh_two_link = np.array([[0, 0, 1, 0], [0, 0, 1, 0]])
two_link_arms.append(rp.SerialArm(dh_two_link))
two_link_arms.append(rp.SerialArm(dh_two_link * 3))
two_link_arms.append(rp.SerialArm(dh_two_link * 0.3))

three_link_arms = []
dh_three_link = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0.5, 0]])
three_link_arms.append(rp.SerialArm(dh_three_link))
three_link_arms.append(rp.SerialArm(dh_three_link * 3))
three_link_arms.append(rp.SerialArm(dh_three_link * 0.3))

common_arm_funcs = [rp.Panda, rp.Stanford, rp.Baxter, rp.Puma560, rp.Canadarm2, rp.Kuka_KR5, rp.ABB]
common_arms = []
for f in common_arm_funcs:
    arm = f(s=1)
    arm.set_qlim_warnings(False)
    common_arms.append(arm)
    arm = f(s=3)
    arm.set_qlim_warnings(False)
    common_arms.append(arm)
    arm = f(s=0.3)
    arm.set_qlim_warnings(False)
    common_arms.append(arm)

dh_ten_dof = np.array([[0, 0, 0.2, np.pi / 2]] * 10)
ten_dof_arms = []
for s in [0.3, 1, 3]:
    dh = np.copy(dh_ten_dof)
    dh[:, 2] *= s
    ten_dof_arms.append(rp.SerialArm(dh))

## declare constant parameters
tol = 1e-3
mit = 50
maxdel = np.inf
mindel = 1e-8
force = False
retry = 0

## declare kwargs for specific test
# Pseudo-inverse method
kwargs = {'method':'pinv'}

# Jacobian Transpose
# kwargs = {'method':'jt', 'K':0.15, 'Kd':0.0}
# mit = 1000

# CCD Method
# kwargs = {'method':'ccd'}

# Scipy Method
# kwargs = {'method':'scipy'}


## Set up tracking numbers
global total_trials
global success_trials
global total_time
global success_time

total_trials = 0
success_trials = 0
total_time = 0
success_time = 0


## Setup generic calling function
def call_ik(arm, target, q0=None):
    tick = time.perf_counter()
    sol = arm.ik(target, q0=q0, tol=tol, mit=mit, maxdel=maxdel, mindel=mindel, force=force, retry=retry, **kwargs)
    tock = time.perf_counter()
    if not sol.status:
        print(sol.message)
    # print(arm, sol.message)
    return sol.status, tock - tick


def call_arm_general(arm):
    global total_time
    global total_trials
    global success_time
    global success_trials
    for i in range(1):
        q0 = [0] * arm.n
        qf = np.random.random((arm.n,)) * np.pi * 2 - np.pi
        target = arm.fk(qf)
        success, time = call_ik(arm, target, q0)

        total_trials += 1
        total_time += time
        if success:
            success_trials += 1
            success_time += time

    for i in range(1):
        q0 = np.random.random((arm.n,)) * np.pi * 2 - np.pi
        qf = [0] * arm.n

        target = arm.fk(qf)
        success, time = call_ik(arm, target, q0)

        total_trials += 1
        total_time += time
        if success:
            success_trials += 1
            success_time += time

    for i in range(10):
        q0 = np.random.random((arm.n,)) * np.pi * 2 - np.pi
        qf = np.random.random((arm.n,)) * np.pi * 2 - np.pi

        target = arm.fk(qf)
        success, time = call_ik(arm, target, q0)

        total_trials += 1
        total_time += time
        if success:
            success_trials += 1
            success_time += time


if __name__ == '__main__':

    np.random.seed(0)

    for arm in tqdm.tqdm(two_link_arms + three_link_arms + common_arms + ten_dof_arms):
        call_arm_general(arm)

    print(f"Test Results:\nTotal Trials: {total_trials}\t Successful Trials: {success_trials}\t Percentage Success: {100 * success_trials / total_trials}%")
    print(f"Total Time: {total_time}\t Average Time for Successful Trial: {success_time / success_trials}")

"""
pinv v1:
45.5%, 0.019

pinv v2: tried out scalar_optimize
43.1%, 0.024

pinv v3: no optimize, VERY simple line search
58.0%, 0.017

pinv v4: using manual pseudo-inverse
58.8%, 0.016

jt v1:
10.8%, 0.275

jt v2:
33.6%, 0.646

jt v3: simple line search (mainly failed to minimum qdot norm)
4.44%, 0.211

ccd v1:
25.8%, 0.598

scipy v1: Note, fails very quickly, so the overall test is very fast
40.8%, 0.055
"""