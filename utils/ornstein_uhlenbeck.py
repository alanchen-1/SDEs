import numpy as np
import math

class OrnsteinUhlenbeck():
    """
    Class that simulates the Ornstein Uhlenbeck stochastic process.
    """
    def __init__(self, init_value : float, h : float, a : float = 1.0):
        """
        Constructor for Ornstein Uhlenbeck.
            Parameters:
                init_value (float) : initial value
                h (float) : step size
                a (float) : drift term factor (default = 1.0)
        """
        self.init_value = init_value
        self.val = init_value
        self.vals = [init_value]
        self.h = h
        self.a = a

    def step(self) -> float:
        """
        Does one step in the discrete approximation.
            Returns: 
                (float) : updated value
        """
        # \sqrt{h}N(0, 1) \implies variance is h
        kick = np.random.normal(0.0, math.sqrt(self.h))
        return self.val - (self.a*self.val*self.h) + kick

    def set_init_value(self, init_value : float) -> None:
        """
        Setter for the initial value.
        Useful if you want to rerun the process from a new initial value.
            Parameters:
                init_value (float) : new initial value
        """
        self.init_value = init_value

    def simulate(self, iters : int):
        """
        Simulates the process by calling self.step() [iters] times.
        Adds values to self.vals.
            Parameters:
                iters (int) : iterations to run for
        """
        # reset vars
        self.val = self.init_value
        self.vals = [self.init_value]

        # iterate
        for _ in range(iters):
            self.val = self.step()
            self.vals.append(self.val)
