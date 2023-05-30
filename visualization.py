import time

import numpy as np
from numpy import sqrt, sin, cos
from numpy.linalg import norm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QPushButton,
                               QSizePolicy, QLabel)
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

from time import perf_counter, sleep

red = np.array([0.7, 0, 0, 1])
green = np.array([0, 0.7, 0, 1])
blue = np.array([0, 0, 0.7, 1])
dark_red = np.array([0.3, 0, 0, 1])
dark_green = np.array([0, 0.3, 0, 1])
dark_blue = np.array([0, 0, 0.3, 1])
white = np.array([1, 1, 1, 1])
grey = np.array([0.3, 0.3, 0.3, 1])


class VizScene:
    """The viz scene holds all the 3d objects to be plotted. This includes arms (which are GLMeshObjects), transforms
    (which are plots or quiver type things), scatter points, and lines."""
    def __init__(self):
        # self.arm = arm
        # self.arm_object = ArmMeshObject(arm, draw_frames=draw_frames)
        self.arms = []
        self.frames = []
        self.markers = []
        self.range = 5

        if QApplication.instance() is None:
            self.app = pg.QtGui.QApplication([])
        else:
            self.app = QApplication.instance()
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('RoboPy')
        self.window.setGeometry(200, 100, 1200, 900)
        self.grid = gl.GLGridItem()
        self.grid.scale(1, 1, 1)
        self.window.addItem(self.grid)
        self.window.setCameraPosition(distance=self.range)
        self.window.setBackgroundColor('k')
        self.window.show()
        self.window.raise_()
        self.window.opts['center'] = pg.Vector(0, 0, 0)

        self.app.processEvents()

    def add_arm(self, arm, draw_frames=False, joint_colors=None, link_colors=None, ee_color=None, label=None, q=None):
        self.arms.append(ArmMeshObject(arm, draw_frames=draw_frames, joint_colors=joint_colors, link_colors=link_colors, ee_color=ee_color))
        self.arms[-1].update(q)
        self.window.addItem(self.arms[-1].mesh_object)

        if 2 * arm.reach > self.range:
            self.range = 2 * arm.reach
            self.window.setCameraPosition(distance=self.range)

        if label is not None:
            text = GLTextItem(pos=arm.base[0:3, 3], text=label)
            self.window.addItem(text)
            text.setGLViewWidget(self.window)

        self.app.processEvents()

    def remove_arm(self, armIndex=None):
        if armIndex is None:
            for arm in self.arms:
                self.window.removeItem(arm.mesh_object)
            self.arms = []
        elif isinstance(armIndex, (int)):
            self.window.removeItem(self.arms[arm].mesh_object)
            self.arms.pop(arm)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()

    def add_frame(self, A, label=None):
        self.frames.append(FrameViz(label=label))
        self.frames[-1].update(A)
        self.window.addItem(self.frames[-1].mesh_object)
        if label is not None:
            self.window.addItem(self.frames[-1].label)
            self.frames[-1].label.setGLViewWidget(self.window)

        if 2 * norm(A[0:3, 3]) > self.range:
            self.range = 2 * norm(A[0:3, 3])
            self.window.setCameraPosition(distance=self.range)

        self.app.processEvents()

    def remove_frame(self, ind=None):
        if ind is None:
            for frame in self.frames:
                self.window.removeItem(frame.mesh_object)
            self.frames = []
        elif isinstance(ind, (int)):
            self.window.removeItem(self.frames[ind].mesh_object)
            self.frames.pop(ind)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()

    def add_marker(self, pos, color=green, size=10, pxMode=True):
        if not isinstance(pos, (np.ndarray)):
            pos = np.array(pos, dtype=float)
        if isinstance(color, (list)):
            color = np.array(color)
        if pos.ndim == 2:
            m = pos.shape[0]
            color = np.zeros((m, 4)) + color
        spot = gl.GLScatterPlotItem(pos=pos, color=color, size=size, pxMode=pxMode)
        self.markers.append(spot)
        self.window.addItem(spot)
        self.app.processEvents()

    def remove_marker(self, ind=None):
        if ind is None:
            for marker in self.markers:
                self.window.removeItem(marker)
            self.markers = []
        elif isinstance(ind, (int)):
            self.window.removeItem(self.markers[ind])
            self.markers.pop(ind)
        else:
            print("Warning: invalid index entered!")
            return None
        self.app.processEvents()

    def update(self, qs=None, As=None, poss=None):
        if qs is not None:
            if isinstance(qs[0], (list, tuple, np.ndarray)):
                for i in range(len(self.arms)):
                    self.arms[i].update(qs[i])
            else:
                self.arms[0].update(qs)

        if As is not None:
            if isinstance(As, (list, tuple)):
                for i in range(len(self.frames)):
                    self.frames[i].update(As[i])
            elif len(As.shape) == 3:
                for i in range(len(self.frames)):
                    self.frames[i].update(As[i])
            else:
                self.frames[0].update(As)

        if poss is not None:
            if isinstance(poss, (list, tuple, np.ndarray)):
                for i in range(len(self.markers)):
                    if not isinstance(poss[i], (np.ndarray)):
                        pos = np.array(poss[i])
                    else:
                        pos = poss[i]
                    self.markers[i].setData(pos=pos)
            else:
                if not isinstance(poss, (np.ndarray)):
                    pos = np.array(poss)
                else:
                    pos = poss
                self.markers[0].setData(pos=pos)

        self.app.processEvents()
        # sleep(0.00001)

    def hold(self, t=None):
        if t is None:
            while self.window.isVisible():
                self.app.processEvents()
        else:
            time_start = time.perf_counter()
            time_end = time_start + t
            while time.perf_counter() < time_end:
                self.app.processEvents()

    def wander(self, index=None, q0=None, speed=1e-1, duration=np.inf, accel=5e-4):
        if index is None:
            index = range(len(self.arms))

        tstart = perf_counter()
        t = tstart
        flag = True
        qs = []
        dqs = []

        while t < tstart + duration and self.window.isVisible():
            for i, ind in enumerate(index):
                n = self.arms[ind].n
                if flag:
                    if q0 is None:
                        qs.append(np.zeros((n,)))
                    else:
                        qs.append(q0[i])
                    # dqs.append(np.random.random_sample((n,)) * speed - speed / 2)
                    dqs.append(np.zeros((n,)))

                dqq = np.zeros((n,))
                for j in range(n):
                    s = dqs[i][j] / speed
                    dqq[j] = dqq[j] + np.random.random_sample((1,)) * accel - accel / 2 - accel * s**3
                dqs[i] = dqs[i] + dqq
                qs[i] = qs[i] + dqs[i]
                self.arms[ind].update(qs[i])

            if flag:
                flag = False
            t = perf_counter()
            self.app.processEvents()

    def quit(self):
        self.window.clear()
        self.window.close()
        self.app.exit()


class ArmPlayer:
    def __init__(self, arm):
        if QApplication.instance() is None:
            self.app = pg.QtGui.QApplication([])
        else:
            self.app = QApplication.instance()
        self.window = QMainWindow()
        self.window.setGeometry(200, 300, 1000, 700)
        self.window.setWindowTitle("Arm Play")

        self.main_layout = QHBoxLayout()
        w1 = gl.GLViewWidget()
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        w1.addItem(grid)
        self.arm = arm
        self.armObject = ArmMeshObject(arm)
        self.n = arm.n
        w1.addItem(self.armObject.mesh_object)
        w1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        w2 = QVBoxLayout()
        self.slider_list = []
        self.slider_label_list = []
        for i in range(arm.n):
            t = QLabel()
            if arm.jt[i] == 'r':
                t.setText(f"Joint {i + 1}: 0 degrees")
            else:
                t.setText(f"Joint {i + 1}: 0")
            t.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            s = QSlider(Qt.Horizontal)
            if arm.qlim is None:
                s.setMinimum(-360)
                s.setMaximum(360)
            else:
                if not arm.qlim[i, 0] == -np.inf:
                    s.setMinimum(arm.qlim[i, 0] * 180 / np.pi * 2)
                else:
                    s.setMinimum(-360)
                if not arm.qlim[i, 1] == np.inf:
                    s.setMaximum(arm.qlim[i, 1] * 180 / np.pi * 2)
                else:
                    s.setMaximum(360)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            s.sliderMoved.connect(self.update_sliders)
            self.slider_list.append(s)
            self.slider_label_list.append(t)
            w2.addWidget(t, stretch=1)
            w2.addWidget(s, stretch=3)
        button = QPushButton()
        button.setText("Randomize")
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.pressed.connect(self.button_pressed)
        self.random_button = button
        w2.addWidget(button)
        self.main_layout.addWidget(w1, stretch=3)
        self.main_layout.addLayout(w2, stretch=1)

        w = QWidget()
        w.setLayout(self.main_layout)
        self.window.setCentralWidget(w)
        self.window.show()
        self.window.raise_()

        self.app.processEvents()

        self.app.exec_()

    def update_sliders(self):

        qs = np.zeros((self.n,))

        for i, s in enumerate(self.slider_list):
            q = s.value()
            qs[i] = q / 2 * np.pi / 180
            if self.arm.jt[i] == 'r':
                self.slider_label_list[i].setText(f"Joint {i + 1}: {q / 2} degrees")
            else:
                self.slider_label_list[i].setText(f"Joint {i + 1}: {q / 2}")

        self.armObject.update(qs)

    def button_pressed(self):

        qs = np.random.random_sample((self.n,)) * 2 * np.pi - np.pi

        for i, z in enumerate(zip(self.slider_list, qs)):
            s, q1 = z[0], z[1]
            q = int(180 / np.pi * 2 * q1)
            s.setValue(q)
            if self.arm.jt[i] == 'r':
                self.slider_label_list[i].setText(f"Joint {i + 1}: {q / 2} degrees")
            else:
                self.slider_label_list[i].setText(f"Joint {i + 1}: {q / 2}")

        self.armObject.update(qs)


class SimViz(QMainWindow):
    def __init__(self, arm):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        super().__init__()
        self.setGeometry(50, 50, 1000, 800)
        self.n = arm.n

        holder = QWidget()
        self.setCentralWidget(holder)

        mainLayout = QHBoxLayout()
        holder.setLayout(mainLayout)
        self.w = gl.GLViewWidget()
        self.w.addItem(gl.GLGridItem())
        self.armVizObject = ArmMeshObject(arm)
        self.w.addItem(self.armVizObject.mesh_object)
        mainLayout.addWidget(self.w, stretch=1)
        self.w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        plotLayout = QVBoxLayout()
        mainLayout.addLayout(plotLayout, stretch=1)
        p1 = pg.PlotWidget()
        p2 = pg.PlotWidget()
        plotLayout.addWidget(p1)
        plotLayout.addWidget(p2)
        p1.setXRange(0, 5.0)
        p2.setXRange(0, 5.0)
        p1.setYRange(-np.pi, np.pi)
        p2.setYRange(-10, 10)
        p1.addLegend()
        p2.addLegend()
        self.qsPlots = []
        self.qdsPlots = []
        for i in range(self.n):
            pen = pg.mkPen(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            self.qsPlots.append(p1.plot(name=f'q{i + 1}', pen=pen))
            self.qdsPlots.append(p2.plot(name=f'qdot{i + 1}', pen=pen))

        self.show()

    def callback_function(self, sim, history):
        q = sim.q
        self.armVizObject.update(q)
        for i in range(self.n):
            self.qsPlots[i].setData(history.ts, history.qs[:, i])
            self.qdsPlots[i].setData(history.ts, history.qds[:, i])
        self.app.processEvents()


class ArmMeshObject:
    def __init__(self, arm, link_colors=None, joint_colors=None, ee_color=None, q0=None, draw_frames=False, frame_names=None):
        self.arm = arm
        self.n = arm.n
        self.dh = arm.dh
        self.draw_frames = draw_frames

        if q0 is None:
            q0 = np.zeros((self.n,))
        self.q0 = q0

        self.link_objects = []
        self.frame_objects = []

        # Make sure we have a list of numpy arrays for the colors
        if link_colors is None:
            link_colors = [dark_blue] * self.n
        elif not hasattr(link_colors[0], '__iter__'):
            link_colors = [link_colors] * self.n

        if joint_colors is None:
            joint_colors = [dark_red] * self.n
        elif not hasattr(joint_colors[0], '__iter__'):
            joint_colors = [joint_colors] * self.n

        self.frame_objects.append(FrameMeshObject())

        for i in range(self.n):
            self.link_objects.append(LinkMeshObject(self.dh[i],
                                     link_color=link_colors[i],
                                     joint_color=joint_colors[i]))
            self.frame_objects.append(FrameMeshObject())

        self.ee_object = EEMeshObject(color=ee_color)
        self.frame_objects.append(FrameMeshObject())

        self.mesh = np.zeros((0, 3, 3))
        self.colors = np.zeros((0, 3, 4))

        self.set_mesh(q0)

        self.mesh_object = gl.GLMeshItem(vertexes=self.mesh,
                                         vertexColors=self.colors,
                                         drawEdges=True,
                                         computeNormals=False,
                                         edgeColor=np.array([0, 0, 0, 1]))

    def update(self, q=None):
        if q is None:
            q = self.q0
        self.set_mesh(q)
        self.mesh_object.setMeshData(vertexes=self.mesh,
                                    vertexColors=self.colors)
    def set_mesh(self, q):
        meshes = []
        colors = []
        if self.draw_frames:
            A = self.arm.fk(q, 0, base=True, tip=False)
            R = A[0:3, 0:3]
            p = A[0:3, 3]
            meshes.append(self.frame_objects[0].get_mesh(R, p))
            colors.append(self.frame_objects[0].get_colors())
        for i in range(self.n):
            A = self.arm.fk(q, i+1, base=True, tip=False)
            R = A[0:3, 0:3]
            p = A[0:3, 3]
            meshes.append(self.link_objects[i].get_mesh(R, p))
            colors.append(self.link_objects[i].get_colors())
            if self.draw_frames:
                meshes.append(self.frame_objects[i+1].get_mesh(R, p))
                colors.append(self.frame_objects[i + 1].get_colors())
        A = self.arm.fk(q, i + 1, base=True, tip=True)
        R = A[0:3, 0:3]
        p = A[0:3, 3]
        if self.draw_frames:
            meshes.append(self.frame_objects[-1].get_mesh(R, p))
            colors.append(self.frame_objects[-1].get_colors())
        meshes.append(self.ee_object.get_mesh(R, p))
        colors.append(self.ee_object.get_colors())

        self.mesh = np.vstack(meshes)
        self.colors = np.vstack(colors)


class LinkMeshObject:
    def __init__(self, dh, jt='r',
                 link_width=0.1,
                 joint_width=0.15,
                 joint_height=0.25,
                 link_color=None,
                 joint_color=None):

        lw = link_width

        d = dh[0]
        theta = dh[1]
        a = dh[2]
        alpha = dh[3]

        w = joint_width
        h = joint_height

        le = sqrt(d**2 + a**2)

        self.link_points = np.array([[0, lw/2, 0],
                                     [0, 0, lw/2],
                                     [0, -lw/2, 0],
                                     [0, 0, -lw/2],
                                     [-le, lw / 2, 0],
                                     [-le, 0, lw / 2],
                                     [-le, -lw / 2, 0],
                                     [-le, 0, -lw / 2]])

        v1 = np.array([-le, 0, 0])
        v2 = np.array([-a, d * sin(-alpha), -d * cos(-alpha)])

        axis = np.cross(v1, v2)
        n_axis = np.linalg.norm(axis)
        if np.abs(n_axis) < 1e-12:
            R = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            ang = np.arccos(v1 @ v2 / (norm(v1) * norm(v2)))
            v = axis / norm(axis)
            V = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R = np.eye(3) + sin(ang) * V + (1 - cos(ang)) * V @ V
            # R = axis2R(ang, axis)

        self.link_points = self.link_points @ R.T

        self.joint_points = np.array([[0.5 * w, -0.5 * w, -0.5 * h],
                                [-0.5 * w, -0.5 * w, -0.5 * h],
                                [-0.5 * w, -0.5 * w, 0.5 * h],
                                [0.5 * w, -0.5 * w, 0.5 * h],
                                [0.5 * w, 0.5 * w, -0.5 * h],
                                [-0.5 * w, 0.5 * w, -0.5 * h],
                                [-0.5 * w, 0.5 * w, 0.5 * h],
                                [0.5 * w, 0.5 * w, 0.5 * h]])

        Rz = np.array([[cos(theta), -sin(theta), 0],
                       [sin(theta), cos(theta), 0],
                       [0, 0, 1]])

        Rx = np.array([[1, 0, 0],
                       [0, cos(alpha), -sin(alpha)],
                       [0, sin(alpha), cos(alpha)]])

        self.joint_points = self.joint_points @ Rz @ Rx + v2

        if link_color is None:
            link_color = np.array([0, 0, 0.35, 1])
        elif not isinstance(link_color, (np.ndarray)):
            link_color = np.array(link_color)
        self.link_colors = np.zeros((12, 3, 4)) + link_color

        if joint_color is None:
            joint_color = np.array([0.35, 0, 0., 1])
        elif not isinstance(joint_color, (np.ndarray)):
            joint_color = np.array(joint_color)
        self.joint_colors = np.zeros((12, 3, 4)) + joint_color
        self.joint_colors[6:8, :, :] = np.zeros((2, 3, 4)) + grey

    @staticmethod
    def points_to_mesh(link_points, joint_points):

        link_mesh = np.array([[link_points[0], link_points[4], link_points[5]],
                              [link_points[0], link_points[1], link_points[5]],
                              [link_points[1], link_points[5], link_points[6]],
                              [link_points[1], link_points[2], link_points[6]],
                              [link_points[2], link_points[6], link_points[7]],
                              [link_points[2], link_points[7], link_points[3]],
                              [link_points[3], link_points[7], link_points[4]],
                              [link_points[3], link_points[0], link_points[4]],
                              [link_points[0], link_points[2], link_points[3]],
                              [link_points[0], link_points[1], link_points[2]],
                              [link_points[5], link_points[4], link_points[7]],
                              [link_points[5], link_points[6], link_points[7]]])

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

    def get_colors(self):
        return np.vstack((self.link_colors, self.joint_colors))

    def get_mesh(self, R, p):
        lp = self.link_points @ R.T + p
        jp = self.joint_points @ R.T + p

        return self.points_to_mesh(lp, jp)


class FrameViz:
    def __init__(self, A=np.eye(4), scale=1, colors=[red, green, blue], label=None):
        self.frame_object = FrameMeshObject(scale, colors)
        self.mesh = self.frame_object.get_mesh(A[0:3, 0:3], A[0:3, 3])
        self.colors = self.frame_object.get_colors()
        self.mesh_object = gl.GLMeshItem(vertexes=self.mesh,
                                         vertexColors=self.colors,
                                         computeNormals=False,
                                         drawEdges=False)
        if label is not None:
            self.label = GLTextItem(A[0:3, 3], text=label)
        else:
            self.label = None

    def update(self, A):
        self.mesh = self.frame_object.get_mesh(A[0:3, 0:3], A[0:3, 3])
        self.colors = self.frame_object.get_colors()
        self.mesh_object.setMeshData(vertexes=self.mesh,
                                     vertexColors=self.colors)
        if self.label is not None:
            p = A[0:3, 3] + A[0:3, 0:3] @ np.array([0.35, 0, 0])
            self.label.setData(pos=p)


class FrameMeshObject:
    def __init__(self, scale=1, colors=[red, green, blue]):
        a = 0.1 * scale
        b = 0.35 * scale
        self.points = np.array([[0, 0, 0],
                                [a, 0, 0],
                                [0, a, 0],
                                [0, 0, a],
                                [b, 0, 0],
                                [0, b, 0],
                                [0, 0, b]])
        c = [np.zeros((4,3,4)) + red, np.zeros((4,3,4)) + green, np.zeros((4,3,4)) + blue]
        self.colors = np.vstack(c)

    @staticmethod
    def points_to_mesh(p):
        mesh = np.array([[p[0], p[2], p[3]],
                         [p[0], p[3], p[4]],
                         [p[3], p[2], p[4]],
                         [p[2], p[0], p[4]],
                         [p[0], p[1], p[3]],
                         [p[0], p[1], p[5]],
                         [p[1], p[3], p[5]],
                         [p[3], p[5], p[5]],
                         [p[0], p[1], p[2]],
                         [p[0], p[1], p[6]],
                         [p[1], p[2], p[6]],
                         [p[2], p[0], p[6]]
                         ])
        return mesh

    def get_mesh(self, R, p):
        points = self.points @ R.T + p
        mesh = self.points_to_mesh(points)
        return mesh

    def get_colors(self):
        return self.colors


class EEMeshObject:
    def __init__(self, scale=1, w=0.05, o1=0.05, o2=0.15, o3=0.3, o4=0.2, o5=0.1, color=None):
        if color is None:
            color = dark_red
        elif not isinstance(color, np.ndarray):
            color = np.asarray(color)

        w = w * scale
        o1 = o1 * scale
        o2 = o2 * scale
        o3 = o3 * scale
        o4 = o4 * scale
        o5 = o5 * scale

        self.points = np.array([[-o1, o2, w/2],
                                [-o1, -o2, w/2],
                                [o3, 0, w/2],
                                [-o1, o2, -w/2],
                                [-o1, -o2, -w/2],
                                [o3, 0, -w/2]
                                ])
        self.colors = np.zeros((8,3,4)) + color
        self.colors[1,:,:] = np.zeros((3,4)) + color

    @staticmethod
    def points_to_mesh(p):
        mesh = np.array([[p[0], p[1], p[2]],
                         [p[3], p[4], p[5]],
                         [p[0], p[1], p[3]],
                         [p[1], p[4], p[3]],
                         [p[1], p[2], p[5]],
                         [p[1], p[4], p[5]],
                         [p[0], p[2], p[5]],
                         [p[0], p[3], p[5]]
                         ])
        return mesh

    def get_mesh(self, R, p):
        points = self.points @ R.T + p
        mesh = self.points_to_mesh(points)
        return mesh

    def get_colors(self):
        return self.colors


class GLTextItem(GLGraphicsItem):
    """
    Class for plotting text on a GLWidget
    """

    def __init__(self, pos=None, text=None):
        GLGraphicsItem.__init__(self)
        self.setGLOptions('translucent')
        self.text = text
        self.pos = pos

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setPos(self, pos):
        self.pos = pos
        self.update()

    def setData(self, pos=None, text=None):
        if pos is not None:
            self.pos = pos

        if text is not None:
            self.text = text

        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(Qt.white)
        self.GLViewWidget.renderText(self.pos[0], self.pos[1], self.pos[2], self.text)
