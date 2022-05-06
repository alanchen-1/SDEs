import numpy as np
import scipy as sp
from .general import get_Pbeta

class MatrixCompletion():
    def __init__(self, init_value : np.array, h : float, beta : float):
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

    def calc_residual(self):
        return 1 - (self.val ** 2)

    def get_noise(self, residual : np.array):
        r = np.sqrt(residual)
        P_beta = get_Pbeta(self.beta, r)
        # use special covariance
        noise = np.random.multivariate_normal(np.zeros(self.n), P_beta * self.h)
        return noise
    
    def step(self):
        residual = self.calc_residual()
        self.val = self.val + self.get_noise(residual)
        self.vals.append(self.val)

    def simulate(self, iters : int):
        self.val = self.init_value
        self.vals = [self.init_value]
        for _ in range(iters):
            self.step()

    def extinct(self) -> bool:
        for v in self.val:
            magnitude = abs(v)
            if (magnitude >= 1.0):
                return True

        return False

    def extinct_simulate(self):
        self.val = self.init_value
        self.vals = [self.init_value]
        iters = 0
        while(not self.extinct()):
            self.step()
            iters += 1
        return iters

