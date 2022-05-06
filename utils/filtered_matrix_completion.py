import numpy as np
from .general import get_Pbeta
import math

def random_matrix_wishart(n : int, p: int) -> np.array:
    """
    Generates a random matrix drawn from the Wishart Ensemble.
        Parameters:
            n (int) : dimension of matrix
            m (int) : number of columns in intermediate rectangular matrix used in Wishart generation
        Returns:
            P (np.array): n by n Wishart matrix
    """
    X = np.random.normal(0, 1, (n, p))
    P = X @ X.transpose()
    return P

def rescale(matrix : np.array, max_singular_val : float = 0.5):
    _, s, _ = np.linalg.svd(matrix)
    cur_max = np.amax(s)
    ratio = max_singular_val/cur_max
    return ratio * matrix

class FilteredMatrixProcess():
    def __init__(self, n : int, p : int, h : float, beta : float):
        # initialize L_0
        self.init_L = rescale(random_matrix_wishart(n, p))
        self.L = self.init_L
        self.n = n
        self.p = p
        # using L_0, initialize random v_0
        self.init_v = np.random.multivariate_normal(np.zeros(self.n), self.L)
        self.v = self.init_v
        self.vals = [self.init_v]
        # misc
        self.h = h
        self.beta = beta

    def calc_Lresidual(self):
        """
        Returns r^2 residual
        """
        return np.ones(self.n) - np.diagonal(self.L)
    
    def extinct(self) -> bool:
        """
        Determines if the process is extinct based on a condition on the diagonal
        elements of the current filter.
        """
        abs_diagonal = np.abs(np.diagonal(self.L))
        if np.amax(abs_diagonal) <= 1:
            return False
        return True

    def step_L(self, P_beta):
        self.L = self.L + self.h * (self.L @ P_beta @ self.L)

    def step_v(self, P_beta):
        r = np.sqrt(self.calc_Lresidual())
        P_beta = get_Pbeta(self.beta, r)
        noise = np.random.multivariate_normal(np.zeros(self.n), P_beta)
        self.v = self.v + (self.L @ noise) * math.sqrt(self.h)
        self.vals.append(self.v)

    def step(self):
        r = np.sqrt(self.calc_Lresidual())
        P_beta = get_Pbeta(self.beta, r)
        self.step_v(P_beta)
        self.step_L(P_beta)

    def extinct_simulate(self):
        # reset stuff
        self.L = rescale(random_matrix_wishart(self.n, self.p))
        self.v = np.random.multivariate_normal(np.zeros(self.n), self.L)
        self.vals = [self.v]
        # iterate until extinct
        iters = 0
        while(not self.extinct()):
            self.step()
            iters += 1
        return iters


        
