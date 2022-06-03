1) <s> Simulation (RK4 stepping) </s>
2) Trajectory
3) <s> Sliders </s> 
4) <s> Forward Dynamics: RNE and EL </s>
5) List of premade arms, like panda, baxter, stanford, HUMAN ARM, etc.
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
11) Joint Limits, with integrated FK, IK, and simulation stuff
12) Control module, enabling control functions or some common control methods to be passed in
13) Simulation module should include linearization and step responses
14) Make sliders on play function taller when possible.
15) Make general dynamic arm creator with linear density
16) <s> Kinetic and Potential Energy calculation </s>
17) Make real decision about gravity and external wrench convention and standardize
18) <s> quaternion utility; slerp, product, etc </s>
19) IK uses quaternions internally
20) Dual quaternions
21) Random joint angle helper function, with list of joint limits. I'm tired of redoing this