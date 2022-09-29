# Redesign for Visualization Module

<i> This document contains a more detailed explanation of the redesign featurs for the next 
iteraiont of the visualization module, including object hierarchy and API</i>

## Structure

### Hierarchy

The current hierarchy of the visualization module is a bit haphazard and introduces difficulties in use and modification.
Currently, we have a VizScene object which contains lists of arm, frame, and scatter point objects.
The arm objects are instances of ArmMeshObjects, which are themselves composed of LinkMeshObjects and FrameMeshObjects.
The Link and Frame MeshObjects are convenient objects that create the points and triangle mesh for each object and move them when given a transform.
The ArmMeshObject takes all the points and mesh arrays from its constituent links and frames and appends them together.
This allows the ArmMeshObject to be drawn as a single GLMeshObject instead of multiple GLMeshObjects, but has the downside of making it hard to access individual components and difficult to add new parts.

