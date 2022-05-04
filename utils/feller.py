import numpy as np
import math

class Feller():
    def __init__(self, init_value : float, h : float):
        self.init_value = init_value
        self.val = init_value
        self.vals = [init_value]
        self.h = h
    
    def step(self):
        kick = np.random.normal(0.0, math.sqrt(self.h))
        return self.val + np.sqrt(self.val) * kick

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
        
    def extinct_simulate(self):
        # reset vars
        self.val = self.init_value
        self.vals = [self.init_value]
        # count iters
        iters = 0
        while (self.val > 0):
            self.val = self.step()
            self.vals.append(self.val)
            iters += 1
        return iters

