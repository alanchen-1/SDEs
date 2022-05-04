import numpy as np
import math

class OrnsteinUhlenbeck():
    def __init__(self, init_value : float, h : float, a : float = 1.0):
        self.init_value = init_value
        self.val = init_value
        self.vals = [init_value]
        self.h = h
        self.a = a

    def step(self):
        # \sqrt{h}N(0, 1) \implies variance is h
        kick = np.random.normal(0.0, math.sqrt(self.h))
        return self.val - (self.a*self.val*self.h) + kick

    def set_init_value(self, init_value : float):
        self.init_value = init_value

    def simulate(self, iters : int):
        # reset vars
        self.val = self.init_value
        self.vals = [self.init_value]

        # iterate
        for _ in range(iters):
            self.val = self.step()
            self.vals.append(self.val)
