1) Simulation (RK4 stepping)
2) Trajectory
3) <s> Sliders </s> 
4) Forward Dynamics: RNE and EL
5) List of premade arms, like panda, baxter, stanford, HUMAN ARM, etc.
6) Wrap auto-differentiation around fk, jacob, and dynamics methods
7) Pull IK into its own module, it's just too messy currently, and give it a wrapper function that calls different 
internal methods, right now the parameters list for IK is just nuts. Maybe an opts function? like an IKopts dataclass that 
gets put in to clean up the API for IK?
8) Need to rethink how VizScene is done, specifically for updating when there are multiple objects
in the scene. Maybe you should be getting handles to the specific objects inside instead of needing to pass a
list of update parameters in. This would make using VizScenes inside of stuff like IK or simulation work better for 
multiple object scenes.
9) Viscous damping on dynamics