import os
import pygame  # noqa: E402
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
        self.w4 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w5 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w6 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )

        self.w7 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w8 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w9 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w10 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w11 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w12 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )

        self.w13 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w14 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w15 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w16 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w17 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )
        self.w18 = Scale(
            self.master,
            from_=-10,
            to=10,
            length=500,
            tickinterval=10,
            orient=HORIZONTAL,
        )

        self.sliders = [
            self.w1,
            self.w2,
            self.w3,
            self.w4,
            self.w5,
            self.w6,
            self.w7,
            self.w8,
            self.w9,
            self.w10,
            self.w11,
            self.w12,
            self.w13,
            self.w14,
            self.w15,
            self.w16,
            self.w17,
            self.w18,
        ]

        for s in self.sliders:
            s.set(0)
            s.pack()
        self.master.update()

    def update(self, env):
        for w in range(0, len(self.sliders)):
            env.pca_scalings[:, w : w + 1] = torch.full(
                (env.pca_scalings.shape[0], 1), self.sliders[w].get()
            )
        for s in self.sliders:
            s.pack()
        self.master.update()
