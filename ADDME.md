1) <s> Simulation (RK4 stepping) </s>
2) Trajectory
3) <s> Sliders </s> 
4) <s> Forward Dynamics: RNE and EL </s>
5) <s> List of premade arms, like panda, baxter, stanford, HUMAN ARM, etc. </s>
6) Wrap auto-differentiation around fk, jacob, and dynamics methods
7) <s> Pull IK into its own module, it's just too messy currently, and give it a wrapper function that calls different internal methods, right now the parameters list for IK is just nuts. Maybe an opts function? like an IKopts dataclass that gets put in to clean up the API for IK? </s> 
   Pull methods into their own functions in kinematics
8) Need to rethink how VizScene is done, specifically for updating when there are multiple objects
in the scene. Maybe you should be getting handles to the specific objects inside instead of needing to pass a
list of update parameters in. This would make using VizScenes inside of stuff like IK or simulation work better for 
multiple object scenes.
9) <s> Viscous damping on dynamics </s>
10) Improve simulation setup, should have an integrated method to take care of weird
damping stuff going on. Or maybe that needs to be in forward dynamics?
11) <s> Joint Limits, with integrated FK, </s> IK,  and simulation stuff
12) Control module, enabling control functions or some common control methods to be passed in
13) Simulation module should include linearization and step responses
14) Make sliders on play function taller when possible.
15) <s> Make general dynamic arm creator with linear density </s>
16) <s> Kinetic and Potential Energy calculation </s>
17) Make real decision about gravity and external wrench convention and standardize
18) <s> quaternion utility; slerp, product, etc </s>
19) IK uses quaternions internally
20) **** Dual quaternions ****
21) Random joint angle helper function, with list of joint limits. I'm tired of redoing this. BTW this should maybe be a method attached to Serial arm, so that a specific arm can generate sample points for itself
22) <s> Reachable distance estimate as a property of Serial Arm </s>
23) <s> Okay the reachable distance thing was great but it's not so awesome that you need two copies of it! Figure out where arm.reach and arm.max_reach are being used and consolidate, please! </s>
24) Modified DH parameter conversion and arm generation
25) Github landing page with tutorial
26) Finish tutorial script and keep up to date
27) __str__ and __repr__ functions for SerialArmDyn class need additional details
28) ArmPlayer: It would be nice to be able to add other static objects to the scene, like markers, lines, etc. Likewise it would be good to add a callback function that executes each step given the joint angles (so we can add things like moving vectors, etc)
29) Functionality for saving images from VizScene. First, a function that holds an app.processEvents() loop until user terminal input so that I can set up the image how I want. Then, a function that takes the current GLViewWindow, saves to an image, and puts it somewhere. It would be really cool if you could do this with GIF's as well, execute for a set time and save as a video or GIF. See https://groups.google.com/g/pyqtgraph/c/dKT1Z3nIeow/m/OErAgRPAbB8J
30) SerialArm object should have preset joint configurations, e.g. zeros, maybe more for the premade arms
31) ArmPlayer display Jacobian
32) Rigid body kinematics, given q, qd, qdd, a, omega, and alpha return v, a, omega, and alpha for each frame
33) Null space option on IK function
34) FrameMeshObject should just be part of FrameViz
35) Lie Groups: I think (emphasis on think) that Lie groups are the natural way to solve IK in that they provide the most natural definition for smooth error between a target pose and a current pose. It would also be great to learn the math by applying it to a concrete problem. Therefore, I'd like to make an IK3 function that uses Lie Groups to solve for pose-related inverse kinematics. This would be the same for purely translational targets, like cartesian or planar IK, but would replace (or maybe be equivalent to?) the quaternion analytic jacobian method for full SE3 stuff.
36) Geometric Algebra: Same idea as with Lie Algebras, I'd love to see how Geometric Algebra can be applied to serial arm robotics and if that would yield any good performance or simplified calculation.
37) ***IMPORTANT*** Caching for common functions like fk and jacob. fk gets recalled a *lot* during normal operation, and this could be a dramatic increase in performance especially for dynamics methods that require recalculation of jacobians over and over again. Caching is not going to be super simple, since numpy arrays can't be directly cached. I'm going to need to keep an input-standardization function, because I want inputs to still be able to be called with lists, numpy arrays, etc.
38) call sign change. I am inevitably evolving into the convoluted (but hopefully well designed) monster package that I 
criticised when using robotics toolbox the first time. So, right now I have a ton of boiler plate code that does
input standardization, error checking, etc. for fk or jacob calls. Instead, I should have one function which checks 
input sizes and converts inputs to a common data type (which should probably be a tuple for reasons in 37).
Then the output of that function will go to other internal methods that can be leaner because they know the 
input type. For example, arm.fk(q, index, base, tip, rep) will first call self._input_standardization(q) which
would turn q into a tuple, check that len(q) == n, etc. Then that standardized output comes back to the fk
function, (which will also check for the dimensions involved) and calls self._fk(q: tuple, index: int, ...)
all fully known function inputs which can be cached using @lru_cache. Actually we can cut out base and tip from this as 
well to speed things up more, and just manually pre or post multiply at the end (makes cache more dense).
