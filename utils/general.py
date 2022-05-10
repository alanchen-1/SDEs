import numpy as np

def get_Pbeta(beta : float, r : np.array) -> np.array:
    """
    Returns P_beta for a given value of beta. Currently only 
    implemented for 0 and infinity.
    r^2 will appear on the diagonal of the returned matrix.
        Parameters:
            beta (float) : beta parameter
            r (np.array) : diagonal residual elements
        Returns:
            (np.array) : P_beta matrix with r^2 as the diagonal element

    """
    if beta == 0.0:
        return np.diag(r ** 2)
    elif beta == float('inf'):
        return np.outer(r, r)
    else:
        raise NotImplementedError("P_beta not implemented for beta not 0 or infinity")

