import numpy as np
import math

class Feller():
    """
    Class to simulate the Feller diffusion process.
    """
    def __init__(self, init_value : float, h : float):
        """
        Constructor to make an instance of the Feller diffusion process.
            Parameters:
                init_value (float) : initial value
                h (float) : step size
        """
        self.init_value = init_value
        self.val = init_value
        self.vals = [init_value]
        self.h = h
    
    def step(self) -> float:
        """
        Does one iteration of the discrete approximation.
            Returns:
                (float) : updated value
        """
        kick = np.random.normal(0.0, math.sqrt(self.h))
        return self.val + np.sqrt(self.val) * kick

    def set_init_value(self, init_value : float) -> None:
        """
        Setter for initial value. Useful for rerunning the process.
            Parameters:
                init_value (float) : new initial value
        """
        self.init_value = init_value

    def simulate(self, iters : int) -> None:
        """
        Simulates the Feller diffusion process for some number of iterations.
        Warning: process may error out if iters is high enough and the process becomes negative.
            Parameters:
                iters (int) : number of iterations
        """
        # reset vars
        self.val = self.init_value
        self.vals = [self.init_value]
        # iterate
        for _ in range(iters):
            self.val = self.step()
            self.vals.append(self.val)
        
    def extinct_simulate(self):
        """
        Simulate until extinction time (while the value > 0) by calling step().
            Returns:
                iters (int) : number of iterations the process ran for
        """
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

