# READ ME: Instructions on use here
import numpy as np

if __name__ == '__main__':
    ## Tutorial for basic operation.
    """ RoboPy is designed to be a lightweight and convenient tool for visualizing and simulating serial linkage robots.
    The goal is that in a few lines you should be able to create a robot arm and use it to do any of the common 
    boilerplate robotics operations, like forward and inverse kinematics, simple dynamics, and visualization. This 
    tutorial will step through some common tools. Look at the individual files for more details documentation."""

    # To use RoboPy, simply import like you would normally "import RoboPy as rp"
    # or
    from RoboPy import *

    """RoboPy is currently organized into 5 main modules: transforms, kinematics, dynamics, simulation, and visualization.
    """

    # Transforms
    """Transforms includes helper functions for SO2, SO3, and SE3 type objects (frames, basically). All objects are stored
     as numpy arrays."""
    pi = np.pi
    R = rotx(pi / 4)  # The rotx function creates a rotation about the x axis. All rotations default to radians as inputs
    print(f"Rotation of pi / 4 about the x axis\n Rotation Matrix (SO3):\n {R}")

    quaternion = R2q(R)
    axis_angle = R2axis(R)
    euler = R2euler(R, 'xyz')
    rpy = R2rpy(R)

    print(f"Quaternion: {quaternion}\nAxis Angle: {axis_angle}\nEuler xyz: {euler}\nRPY: {rpy}")

    # Kinematics
    """Kinematics includes the base SerialArm class, which is what lets us do forward kinematics, jacobians, etc. It is 
    created using dh parameters in the order dh = [d theta a alpha]"""

    dh = [[0, 0, 1, 0], [0, 0, 1, 0]] # Planar arm with a = 1 for two links
    jt = ['r', 'r'] # jt is the joint type, and can be 'r' for rotational or 'p' for prismatic, and defaults to 'r'
    arm = SerialArm(dh, jt=jt)
    print(arm)

    # We can now do things like forward kinematics or jacobians
    q = [pi/4, -pi/2] # q can be any iterable, e.g. list, tuple, numpy array, etc.
    print(f"Forward kinematics: q = {q}\n{arm.fk(q)}")
    print(f"Jacobian:\n{arm.jacob(q)}")




