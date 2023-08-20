import pyqtgraph as pg
from pyqtgraph import opengl as gl
import numpy as np


def cube_vertex(a, b, c):
    """Returns a 8 x 3 numpy array of points describing the vertices of a rectangular prism, centered on the origin,
    with x side length of a, y side length of b, and z side length of c."""

    x = a / 2
    y = b / 2
    z = c / 2

    vertices = np.array([[-x, -y, -z],
                         [x, -y, -z],
                         [-x, y, -z],
                         [x, y, -z],
                         [-x, -y, z],
                         [x, -y, z],
                         [-x, y, z],
                         [x, y, z]])

    return vertices


def cube_vertex_to_mesh(v):
    """Takes a set of vertices as defined in cube_vertex and converts it to a numpy array suitable for use as a meshData
    for GLMeshItem objects"""

    mesh = np.array([[v[0], v[2], v[1]],
                     [v[1], v[2], v[3]],
                     [v[0], v[4], v[1]],
                     [v[1], v[4], v[5]],
                     [v[0], v[6], v[4]],
                     [v[0], v[2], v[6]],
                     [v[2], v[6], v[3]],
                     [v[3], v[6], v[7]],
                     [v[3], v[7], v[5]],
                     [v[1], v[3], v[5]],
                     [v[4], v[6], v[5]],
                     [v[5], v[6], v[7]]])

    return mesh


class VizSceneObject:
    """A VizSceneObject is the parent class for an object that can be added to a viz scene and updated using some
    parameter. For example, a robot arm is a VizSceneObject instance that has a mesh and can be updated by giving it
    a set of joint values. A single link is also a VizSceneObject that returns a mesh based on a homogeneous transform.

    Every VizSceneObject must define the following methods:
    __init__(*args, **kwargs)
    get_mesh(*args, **kwargs)
    get_colors(*args, **kwargs)

    The following methods are defined already:
    get_GLMeshItem()
    update(*args, **kwargs)
    add(window)

    additionally, for most objects it makes sense to have a function to redefine __init__ parameters (e.g. change color
    or shape of different parts)

    To give an example, we consider a VizFrame object. The VizFrame object is defined by a scale, colors, and label:

    __init__(self, scale=1, colors=[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], label=None)

    This function defines the points making up the mesh when the frame is at the origin. The get_GLMeshItem() method
    returns a gl.GLMeshItem with the correct mesh and colors. The add function then adds this object to an existing
    GLViewWidget.

    The update function for a frame accepts a homogeneous transform and calls the get_mesh and get_colors methods based
    on that information to update things:

    def self.update(*args, **kwargs):
        mesh = self.get_mesh(*args, **kwargs)
        colors = self.get_colors(*args, **kwargs)
        self.glItem.setMeshData(vertexes=mesh, vertexColors=colors)

    VizSceneObjects should update a boolean property "fresh" that details if they have been changed, which the VizScene
    can use to intelligently refresh only when things have changed.
    """

    def __init__(self):
        self.glItem = gl.GLMeshItem()
        self.fresh = False

    def get_mesh(self, *args, **kwargs):
        raise NotImplementedError("VizSceneObject get_mesh function not implemented!")

    def get_colors(self, *args, **kwargs):
        raise NotImplementedError("VizSceneObject get_colors function not implemented!")

    def get_GLMeshItem(self):
        return self.glItem

    def add(self, window):
        window.addItem(self.glItem)

    def update(self, *args, **kwargs):
        mesh = self.get_mesh(*args, **kwargs)
        colors = self.get_colors(*args, **kwargs)
        self.glItem.setMeshData(vertexes=mesh,
                                  vertexColors=colors)

    def redefine(self):
        pass


class VizSceneRigidBody(VizSceneObject):
    """
    VizSceneRigidBody class describes an object to be added to a viz scene, which has a mesh that can be totally defined
    by a homogeneous transform. In other words, a single rigid body with no moving or deformable parts whose update
    method accepts a single 4x4 numpy array as input. This applies to frames, robot links, and revolute joints.

    The get_mesh method for this class works by transforming a fixed array of vertex points by an accepted 4x4 matrix,
    then generating a mesh based on these points.

    The following methods are required by this class:

    __init__()
    get_points()
    points_to_mesh()
    get_colors()

    The following methods are already defined by this class:

    get_mesh()
    """

    @staticmethod
    def points_to_mesh(points):
        raise NotImplementedError("VizSceneRigidBody points_to_mesh function not implemented!")
        return np.eye(3)

    def get_mesh(self, *args, **kwargs):
        A = args[0]
        R = A[0:3, 0:3]
        p = A[0:3, 3]
        points = self.get_points() @ R.T + p
        mesh = self.points_to_mesh(points)
        return mesh
