import PySide2 as ps
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.widgets import Slider
import numpy as np
from Transforms import *


class TransformMPL:

    def __init__(self, A, ax=None):
        self.A = A

    def update(self, A):
        self.A = A

    def show(self):
        plt.show()


class PlanarMPL:

    def __init__(self, arm, q0=None, trace=False, ax=None):
        self.arm = arm
        if ax is not None:
            fig = ax.figure
            flag = True
        else:
            fig, ax = plt.subplots()
            flag = False
        self.fig = fig
        self.ax = ax
        self.n = arm.n
        self.reach = arm.reach
        if not flag:
            plt.axis('equal')
            plt.xlim([-arm.reach * 1.5, arm.reach * 1.5])
            plt.ylim([-arm.reach * 1.5, arm.reach * 1.5])

        self.joints = []
        self.links = []
        self.joints.append(self.ax.add_patch(Circle([0, 0], 0.15, color=[0,0,0,1])))

        if q0 is None:
            q0 = [0] * self.n
        self.q0 = q0
        for i in range(self.n):
            A = self.arm.fk(q0, index=i)
            A_next = self.arm.fk(q0, index=i+1)

            if i != 0:
                joint = Circle(A[0:2, 3], 0.1, color=[0,0,0,1])
                self.joints.append(self.ax.add_patch(joint))
            link, = self.ax.plot([A[0,3], A_next[0,3]], [A[1,3], A_next[1,3]], lw=3, color=[0,0,0,1])
            self.links.append(link)

        end_effector = Circle(A_next[0:2,3], 0.1, color=[0.5,0,0,1])
        self.joints.append(self.ax.add_patch(end_effector))

        self.do_trace = trace
        if trace:
            self.start_trace(q0)

    def start_trace(self, q0):
        self.xs = []
        self.ys = []
        self.traces = []
        for i in range(self.n):
            pos = self.arm.fk(q0, index=i + 1)[0:2, 3]
            C = [0, 0, 0, 0.4]
            C[i] = 0.5
            line, = self.ax.plot(pos[0], pos[1], lw=1, color=C)
            self.xs.append([pos[0]])
            self.ys.append([pos[1]])
            self.traces.append(line)

    def update(self, q):

        for i in range(self.n):
            A = self.arm.fk(q, index=i)
            A_next = self.arm.fk(q, index=i+1)

            if i != 0:
                self.joints[i].set_center(A[0:2, 3])

            self.links[i].set_xdata([A[0,3], A_next[0,3]])
            self.links[i].set_ydata([A[1,3], A_next[1,3]])
        self.joints[-1].set_center(A_next[0:2, 3])

        if self.do_trace:
            for i in range(self.n):
                pos = self.arm.fk(q, index=i+1)[0:2,3]
                self.xs[i].append(pos[0])
                self.ys[i].append(pos[1])
                self.traces[i].set_xdata(self.xs[i])
                self.traces[i].set_ydata(self.ys[i])

        plt.pause(0.02)

    def show(self):
        plt.show()

    def set_bounds(self, xbound=None, ybound=None):
        print('finish me')

    def play(self):
        # clear all the line plots
        if self.do_trace:
            self.do_trace = False
            for i in range(self.n):
                self.ax.lines.remove(self.traces[i])

        # move to the default position
        self.update(self.q0)

        # resize and create slider bars
        self.ax.set_position([0, 0, 0.75, 1])
        plt.axis('equal')
        plt.xlim([-self.arm.reach * 1.5, self.arm.reach * 1.5])
        plt.ylim([-self.arm.reach * 1.5, self.arm.reach * 1.5])

        max_h = 0.2
        min_h = 0.8 / self.n
        h = min(max_h, min_h)
        print(h)
        self.sliders = []
        self.gui_axes = []

        def get_text_from_A(A):
            pos = np.around(A[0:2, 3], decimals=2)
            theta = np.around(rot2theta(A[0:2, 0:2]), decimals=2)
            # print(str(pos[0]))

            s = 'Pos: [' + str(pos[0]) + ', ' + str(pos[1]) + ']\n'
            s += 'Angle: [' + str(theta) + ']\n'

            return s
        text_pos = [-self.reach * 1.35, self.reach * 1.15]
        self.text_box = self.ax.text(text_pos[0], text_pos[1], get_text_from_A(self.arm.fk(self.q0)))

        def slider_update(val):
            q = self.q0
            for i in range(self.n):
                q[i] = self.sliders[i].val
            self.update(q)
            self.text_box.set_text(get_text_from_A(self.arm.fk(q)))
            plt.draw()

        for i in range(self.n):
            self.gui_axes.append(self.fig.add_subplot())
            self.gui_axes[i].set_position([0.775, 0.8 - h * i, 0.15, h - 0.05])
            self.sliders.append(Slider(ax=self.gui_axes[i],
                                        label=str(i),
                                        valmin=-np.pi,
                                        valmax=np.pi,
                                        valinit=0,
                                        orientation='horizontal'))
            # self.gui_axes[i].text(-3, 0.517, f'Joint {i}: {self.q0[i]}')
            self.sliders[i].on_changed(slider_update)

        # set update function to be a loop
        plt.show()


class ArmViz:

    def __init__(self, arm, q0=None):
        self.arm = arm
        self.dh = arm.dh
        self.jt = arm.jt
        self.n = arm.n

        self.link_objects = []
        self.link_mesh_objects = []
        self.link_colors = []

        self.scatter_objects = []
        self.line_objects = []

        if q0 is None:
            q0 = [0] * self.n

        for i in range(self.n):
            self.link_objects.append(LinkMesh(self.dh[i], self.jt[i]))
            T = self.arm.fk(q0, i+1)
            R = T[0:3, 0:3]
            p = T[0:3, 3]
            mesh = self.link_objects[i].get_mesh(R, p)
            self.link_colors.append(self.link_objects[i].get_colors())
            self.link_mesh_objects.append(gl.GLMeshItem(vertexes=mesh,
                                                        vertexColors=self.link_colors[i],
                                                        drawEdges=True,
                                                        computeNormals=False,
                                                        edgeColor=np.array([0, 0, 0, 1])))
        self.app = pg.QtGui.QApplication([])
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Robot Visualization: Bask in the Glow')
        self.window.setGeometry(300, 300, 900, 600)
        self.grid = gl.GLGridItem()
        self.grid.scale(1, 1, 1)
        self.grid = gl.GLGridItem()
        self.grid.scale(1, 1, 1)
        self.window.addItem(self.grid)
        self.window.setCameraPosition(distance=5)
        self.window.setBackgroundColor('k')
        self.window.show()
        self.window.raise_()
        for i in range(self.n):
            self.window.addItem(self.link_mesh_objects[i])
        self.window.opts['center'] = pg.Vector(0, 0, 0)
        self.app.processEvents()

    def update(self, q=None):
        if q is None:
            q = self.q0

        for i in range(self.n):
            T = self.arm.fk(q, i + 1)
            R = T[0:3, 0:3]
            p = T[0:3, 3]
            mesh = self.link_objects[i].get_mesh(R, p)
            self.link_mesh_objects[i].setMeshData(vertexes=mesh, vertexColors=self.link_colors[i])

        self.app.processEvents()

    def addScatter(self, pos, color, size):
        self.scatter_objects.append(gl.GLScatterPlotItem(pos=pos, color=color, size=size))
        self.window.addItem(self.scatter_objects[-1])
        self.app.processEvents()


class LinkMesh:
    def __init__(self, dh, jt='r', link_width=0.1, joint_width=0.15, joint_height=0.25):

        lw = link_width

        d = dh[0]
        theta = dh[1]
        a = dh[2]
        alpha = dh[3]

        w = joint_width
        h = joint_height

        # points at the base
        self.link_points = np.array([[0, lw / 2, 0],
                          [0, 0, lw / 2],
                          [0, -lw / 2, 0],
                          [0, 0, -lw / 2],
                           [-a, sin(-alpha) * -d, cos(-alpha) * -d]])

        self.joint_points = np.array([[0.5 * w, -0.5 * w, -0.5 * h],
                                [-0.5 * w, -0.5 * w, -0.5 * h],
                                [-0.5 * w, -0.5 * w, 0.5 * h],
                                [0.5 * w, -0.5 * w, 0.5 * h],
                                [0.5 * w, 0.5 * w, -0.5 * h],
                                [-0.5 * w, 0.5 * w, -0.5 * h],
                                [-0.5 * w, 0.5 * w, 0.5 * h],
                                [0.5 * w, 0.5 * w, 0.5 * h]])

        self.joint_points = self.joint_points @ (rotx(-alpha) @ rotz(-theta)).T

        self.joint_points = self.joint_points + np.array([-a, sin(-alpha) * -d, cos(-alpha) * -d])


    def points_to_mesh(self, link_points, joint_points):
        link_mesh = np.array([[link_points[0], link_points[1], link_points[2]],  # j_i base 1
                         [link_points[0], link_points[3], link_points[2]],  # j_i base 2
                         [link_points[0], link_points[1], link_points[4]],
                         [link_points[1], link_points[2], link_points[4]],
                         [link_points[2], link_points[3], link_points[4]],
                         [link_points[3], link_points[0], link_points[4]]])

        joint_mesh = np.array([[joint_points[0], joint_points[1], joint_points[2]],
                         [joint_points[0], joint_points[2], joint_points[3]],
                         [joint_points[0], joint_points[3], joint_points[4]],
                         [joint_points[3], joint_points[4], joint_points[7]],
                         [joint_points[2], joint_points[3], joint_points[6]],
                         [joint_points[3], joint_points[6], joint_points[7]],
                         [joint_points[0], joint_points[1], joint_points[5]],
                         [joint_points[0], joint_points[4], joint_points[5]],
                         [joint_points[1], joint_points[2], joint_points[6]],
                         [joint_points[1], joint_points[5], joint_points[6]],
                         [joint_points[4], joint_points[5], joint_points[6]],
                         [joint_points[4], joint_points[6], joint_points[7]]])

        return np.vstack((link_mesh, joint_mesh))

    def get_mesh(self, R=np.eye(3), p=np.zeros((3,1))):
        link_points = self.link_points @ R.T + p
        joint_points = self.joint_points @ R.T + p
        mesh = self.points_to_mesh(link_points, joint_points)
        return mesh

    def get_colors(self):
        link_colors = np.zeros((6,3,4)) + np.array([0, 0, 0.8, 0.5])
        joint_colors = np.zeros((12,3,4)) + np.array([0.5, 0, 0, 0.5])
        return np.vstack((link_colors, joint_colors))


if __name__ == '__main__':
    from Kinematics import SerialArm

    dh = [[1, 0, 1, 0], [1, 0, 1, 0]]
    arm = SerialArm(dh)
    q0 = [0, 0]
    viz = ArmViz(arm)
    for i in range(2000):
        q = [2*np.pi*i/1000.0, -2*np.pi*i/1000.0]
        viz.update(q)

    viz.app.exec_()
