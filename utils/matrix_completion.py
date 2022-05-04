import numpy as np

def get_Pbeta(beta : float, r : np.array):
    if beta == 0.0:
        return np.diagonal(r ** 2)
    elif beta == 'inf':
        return np.outer(r, r)
    else:
        raise NotImplementedError("P_beta not implemented for beta not 0 or infinity")

class MatrixCompletion():
    def __init__(self, init_value : np.array, h : float, beta : float):
        self.val = init_value
        self.vals = [init_value]
        self.init_value = init_value
        self.n = len(init_value)
        self.h = h
        # error check
        if (beta != 0.0 and beta != 'inf'):
            raise NotImplementedError("P_beta not implemented for beta not 0 or infinity")
        else:
            self.beta = beta

    def calc_residual(self):
        return 1 - (self.val ** 2)

    def get_noise(self, residual : np.array):
        P_beta = get_Pbeta(self.beta, residual)
        # use special covariance
        noise = np.random.multivariate_normal(np.zeros(self.n), self.h * P_beta)
        return noise

    def simulate(self, iters : int):
        self.val = self.init_value
        self.vals = [self.init_value]
        for _ in range(iters):
            residual = self.calc_residual()
            self.val = self.val + self.get_noise(residual)
            self.vals.append(self.val)

    def extinct(self) -> bool:
        for v in self.val:
            magnitude = abs(v)
            if (magnitude >= 1.0):
                return True

        return False

    def extinct_simulate(self):
        self.val = self.init_value
        self.vals = [self.init_value]
        while(not self.extinct()):
            residual = self.calc_residual()
            self.val = self.val + self.get_noise(residual)
            self.vals.append(self.val)

