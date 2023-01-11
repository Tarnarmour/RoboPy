import RoboPy as rp
from pyqtgraph import opengl as gl
import numpy as np

red = [1, 0, 0, 1]
green = [0, 1, 0, 1]
blue = [0, 0, 1, 1]

viz = rp.VizScene()
viz.add_frame(A=rp.transl([0, 0, 0]), label=None)

points_original = np.array([[-0.5, -0.5, 0],
                   [0.5, -0.5, 0],
                   [-0.5, 0.5, 0],
                   [0.5, 0.5, 0]])

p = np.array([0, 0, 1])
R = rp.rotx(np.pi / 2)

points = points_original @ R.T + p

mesh = np.array([[points[0], points[1], points[2]],
                 [points[2], points[3], points[1]]])


color = np.array([[red, green, blue],
                  [blue, red, green]])

plane = gl.GLMeshItem(vertexes=mesh, vertexColors=color, drawEdges=False, computeNormals=False)

viz.window.addItem(plane)

t = 0

while True:
    t += 0.001
    p = np.array([0, 0, np.sin(t) + 1])
    R = rp.rotx(np.pi / 2 + t) @ rp.roty(-t) @ rp.rotz(t**2)

    points = points_original @ R.T + p

    mesh = np.array([[points[0], points[1], points[2]],
                     [points[2], points[3], points[1]]])

    plane.setMeshData(vertexes=mesh, vertexColors=color)

    viz.app.processEvents()


viz.hold()
