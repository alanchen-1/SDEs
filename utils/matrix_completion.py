import numpy as np
import scipy as sp
from .general import get_Pbeta

class MatrixCompletion():
    """
    Class to simulate matrix completion stochastic differential equation.
    """
    def __init__(self, init_value : np.array, h : float, beta : float):
        """
        Constructor for Matrix Completion object.
        Dimension of the simulation is autofilled based on len(init_value).
            Parameters:
                init_value (np.array) : vector representing initial value
                h (float) : step size
                beta (float) : beta parameter
        """
        self.val = init_value
        self.vals = [init_value]
        self.init_value = init_value
        self.n = len(init_value)
        self.h = h
        # error check
        if (beta != 0.0 and beta != float('inf')):
            raise NotImplementedError("P_beta not implemented for beta not 0 or infinity")
        else:
            self.beta = beta

    def calc_residual(self) -> np.array:
        """
        Calculates the residual vector based on self.val.
            Returns:
                (np.array) : computed residual vector
        """
        return 1 - (self.val ** 2)

    def get_noise(self, residual : np.array) -> np.array:
        """
        Gets the kick of n-dimensional white noise based on P_beta.
            Parameters:
                residual (np.array) : residual vector to be used in P_beta diagonal
            Returns:
                noise (np.array) : n dimensional noise
        """
        r = np.sqrt(residual)
        P_beta = get_Pbeta(self.beta, r)
        # use special covariance
        noise = np.random.multivariate_normal(np.zeros(self.n), P_beta * self.h)
        return noise
    
    def step(self) -> None:
        """
        Runs one iteration of the discrete approximation of the SDE.
        """
        residual = self.calc_residual()
        self.val = self.val + self.get_noise(residual)
        self.vals.append(self.val)

    def simulate(self, iters : int) -> None:
        """
        Simulates the process for a specified number of iterations.
        Warning: process may error out/become undefined if iters is large enough.
            Parameters:
                iters (int) : number of iterations to run
        """
        self.val = self.init_value
        self.vals = [self.init_value]
        for _ in range(iters):
            self.step()

    def extinct(self) -> bool:
        """
        Checks if the process is extinct based on self.val.
            Returns:
                (bool) : if process is extinct
        """
        for v in self.val:
            magnitude = abs(v)
            if (magnitude >= 1.0):
                return True

        return False

    def extinct_simulate(self) -> int:
        """
        Runs process until the stopping time is achieved by calling self.step() repeatedly.
            Returns:
                iters (int) : number of iterations the process ran for.
        """
        self.val = self.init_value
        self.vals = [self.init_value]
        iters = 0
        while(not self.extinct()):
            self.step()
            iters += 1
        return iters

