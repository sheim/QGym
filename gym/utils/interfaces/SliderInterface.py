import numpy as np
from tkinter import *
import torch


class SliderInterface:
    def __init__(self, env):
        self.master = Tk()
        self.w1 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w2 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w3 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.sliders = [self.w1, self.w2, self.w3]

        for s in self.sliders:
            s.set(0)
            s.pack()
        self.master.update()

    def update(self, env):
        mode = "joint"
        scalings = []
        for w in self.sliders:
            scalings.append(w.get())
        if mode == "one_leg":
            # 4th leg, all actuators

            eigenvectors = np.array(
                [
                    [-0.02939241, 0.81596737, 0.57735027],
                    [0.72134468, -0.38252912, 0.57735027],
                    [-0.69195227, -0.43343826, 0.57735027],
                ]
            )
            env.dof_pos_target[:, 9:12] = env.default_dof_pos[:, 9:12]
            for s in range(len(scalings)):
                eigenvector = eigenvectors[:, s].T
                # should be dof_pos_target
                env.dof_pos_target[:, 9:12] += (
                    torch.from_numpy(scalings[s] * eigenvector)
                    .to(torch.float)
                    .to(device="cuda")
                )

        elif mode == "joint":
            # 3rd actuator kfe, all legs
            eigenvectors = np.array(
                [
                    [0.30653829, -0.76561209, -0.26433388],
                    [-0.5543308, -0.12128448, 0.65422278],
                    [-0.40904421, 0.38943236, -0.65652515],
                    [0.65683672, 0.49746421, 0.26663625],
                ]
            )
            env.dof_pos_target[:, :] = env.default_dof_pos[:, :]
            for s in range(len(scalings)):
                eigenvector = eigenvectors[:, s].T
                fullvec = np.empty((1, 0))
                for i in range(0, 4):
                    fullvec = np.hstack((fullvec, np.array([[0, 0, eigenvector[i]]])))
                env.dof_pos_target[:, :] += (
                    torch.from_numpy(scalings[s] * fullvec)
                    .to(torch.float)
                    .to(device="cuda")
                )

        elif mode == "all":
            # all legs all actuators
            eigenvectors = np.array(
                [
                    [-0.03376138, -0.02442507, 0.03189318],
                    [0.33413213, -0.70889628, 0.18863108],
                    [-0.33155045, 0.23009204, -0.19336469],
                    [-0.00520263, -0.06861266, 0.14744584],
                    [0.36084328, 0.53007859, 0.30072622],
                    [-0.37715193, -0.04550793, -0.32017996],
                    [-0.03926625, 0.09404448, -0.00661434],
                    [0.37683584, 0.18909904, 0.1956597],
                    [-0.31592719, 0.07459572, 0.10245965],
                    [-0.02376299, 0.11192239, -0.02117587],
                    [0.38753615, -0.0902991, -0.74731018],
                    [-0.33272457, -0.29209121, 0.32182934],
                ]
            )
            env.dof_pos_target[:, :] = env.default_dof_pos[:, :]
            for s in range(len(scalings)):
                eigenvector = eigenvectors[:, s : s + 1].T
                env.dof_pos_target[:, :] += (
                    torch.from_numpy(scalings[s] * eigenvector)
                    .to(torch.float)
                    .to(device="cuda")
                )
                print(env.dof_pos_target)
        for s in self.sliders:
            s.pack()
        self.master.update()
