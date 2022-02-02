import PySide2 as ps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

class PlanarMPL:

    def __init__(self, arm, q0=None, trace=False):
        self.arm = arm
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.n = arm.n
        self.reach = arm.reach
        plt.axis('equal')
        plt.xlim([-arm.reach * 1.3, arm.reach * 1.3])
        plt.ylim([-arm.reach * 1.3, arm.reach * 1.3])

        self.joints = []
        self.links = []
        self.joints.append(self.ax.add_patch(Circle([0, 0], 0.15, color=[0,0,0,1])))

        if q0 is None:
            q0 = [0] * self.n

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
            self.xs = []
            self.ys = []
            self.traces = []
            for i in range(self.n):
                pos = self.arm.fk(q0, index=i+1)[0:2,3]
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


