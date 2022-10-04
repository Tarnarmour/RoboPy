# READ ME: Instructions on use here
import numpy as np

if __name__ == '__main__':
    ## Tutorial for basic operation.
    """ RoboPy is designed to be a lightweight and convenient tool for visualizing and simulating serial linkage robots.
    The goal is that in a few lines you should be able to create a robot arm and use it to do any of the common 
    boilerplate robotics operations, like forward and inverse kinematics, simple dynamics, and visualization. This 
    tutorial will step through some common tools. Look at the individual files for more details documentation."""

    # To use RoboPy, simply import like you would normally "import RoboPy as rp"
    # or "from RoboPy import *"
    import RoboPy as rp

    """RoboPy is currently organized into 5 main modules: transforms, kinematics, dynamics, simulation, and visualization.
    """

    # Transforms
    """Transforms includes helper functions for SO2, SO3, and SE3 type objects (frames, basically). All objects are stored
     as numpy arrays."""
    pi = np.pi
    R = rp.rotx(pi / 4)  # The rotx function creates a rotation about the x axis. All rotations default to radians as inputs
    print(f"Rotation of pi / 4 about the x axis\n Rotation Matrix (SO3):\n {R}")

    quaternion = rp.R2q(R)
    axis_angle = rp.R2axis(R)
    euler = rp.R2euler(R, 'xyz')
    rpy = rp.R2rpy(R)

    print(f"Quaternion: {quaternion}\nAxis Angle: {axis_angle}\nEuler xyz: {euler}\nRPY: {rpy}")

    # Kinematics
    """Kinematics includes the base SerialArm class, which is what lets us do forward kinematics, jacobians, etc. It is 
    created using dh parameters in the order dh = [d theta a alpha], and a list of joint types for rotary or prismatic
    joints"""

    dh = [[0, 0, 1, 0], [0, 0, 1, 0]] # Planar arm with a = 1 for two links
    jt = ['r', 'r'] # jt is the joint type, and can be 'r' for rotational or 'p' for prismatic, and defaults to 'r'
    joint_limits = [[-np.pi, np.pi]] * 2  # joint limits are given as (low, high) pairs for each link
    arm = rp.SerialArm(dh, jt=jt, joint_limits=joint_limits)
    print(arm)

    # We can now do things like forward kinematics or jacobians
    q = [pi/4, -pi/2] # q can be any iterable, e.g. list, tuple, numpy array, etc.
    print(f"Forward kinematics: q = {q}\n{arm.fk(q)}")
    print(f"Jacobian:\n{arm.jacob(q)}")

    # Dynamics

    """A SerialArmDyn object is a serial arm with added information on the dynamics and methods for computing torques,
    motion, etc. It is created by adding info about the inertia and centers of mass into the definition."""

    dh = [[0, 0, 1, 0]] * 2
    jt = ['r'] * 2
    mass = [1.0, 2.0]
    link_inertia = [np.eye(3)]  * 2 # the inertia argument takes a list of 3x3 numpy arrays representing the inertia tensor
    r_com = [np.array([-0.5, 0, 0])] * 2  # r_com takes a list of the vectors pointing from frame i to COM of frame i expressed in frame i

    dynamic_arm = rp.SerialArmDyn(dh=dh, jt=jt, mass=mass, link_inertia=link_inertia, r_com=r_com)

    # Dynamic arms have access to both recursive newton euler and euler lagrange methods for finding joint torques
    q = [1, 2]
    qdot = [3, 4]
    qdotdot = [5, 6]
    external_wrench = np.array([1, 2, 3, 4, 5, 6])
    gravity = np.array([1, 2, 3])
    joint_torque_rne = dynamic_arm.rne(q, qdot, qdotdot, Wext=external_wrench, g=gravity)
    joint_torque_el = dynamic_arm.EL(q, qdot, qdotdot, Wext=external_wrench, g=gravity)

    print(f"Joint Torques from RNE: tau = {joint_torque_rne}\nJoint Torques from EL: tau = {joint_torque_el}")

    # We can likewise do forward kinematics with either method
    qdd_rne = dynamic_arm.forward_rne(q, qdot, joint_torque_rne, g=gravity, Wext=external_wrench)
    qdd_el = dynamic_arm.forward_EL(q, qdot, joint_torque_el, g=gravity, Wext=external_wrench)

    print(f"Joint acceleration from RNE: qdd = {qdd_rne}\n Joint acceleration from EL: qdd = {qdd_el}")

    # Visualization
    """
    RoboPy uses PyQt5 and pyqtgraph to animate coordinate frames and serial arms. This is done through a VizScene object,
    which can accept frames, points, and robot arms and update their coordinates
    """

    viz = rp.VizScene()
    viz.add_marker(pos=np.array([1, 2, 3]), color=[1, 1, 0, 1], size=20, pxMode=True)  # arguments get passed to the pyqtgraph.opengl.GLScatterPlotItem API
    viz.add_frame(np.eye(4))
    viz.add_arm(arm)

    viz.hold(2.0)  # hold can be used to display for a certain time

    # To update, pass the coordinates of objects whose positions have changed; qs for joints, poss for markers, As for frames
    viz.update(qs=q, As=rp.se3(rp.rotx(1)), poss=[np.array([3, 2, 1])])

    viz.hold(2.0)

    # if multiple objects are passed in, then they must be updated by passing a list of relevent coordinates
    viz.add_arm(arm)
    q1 = [1, 2]
    q2 = [3, 4]
    viz.update(qs=[q1, q2])

    viz.hold(2.0)

    viz.quit()  # quit closes a visualization

    # An ArmPlayer object can be used to interactively play with a SerialArm
    player = rp.ArmPlayer(arm)



