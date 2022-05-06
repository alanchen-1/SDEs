import numpy as np

def get_Pbeta(beta : float, r : np.array):
    if beta == 0.0:
        return np.diag(r ** 2)
    elif beta == float('inf'):
        return np.outer(r, r)
    else:
        raise NotImplementedError("P_beta not implemented for beta not 0 or infinity")

