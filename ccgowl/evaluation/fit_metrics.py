from scipy import linalg
import numpy as np


def error_norm(theta_star, theta_hat, norm='frobenius', scaling=True, squared=True):
    """ sklearn Graphical LASSO """
    # compute the error
    error = theta_star - theta_hat
    # compute the error norm
    if norm == "frobenius":
        squared_norm = np.sum(error ** 2)
    elif norm == "spectral":
        squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
    else:
        raise NotImplementedError(
            "Only spectral and frobenius norms are implemented")
    # optionally scale the error norm
    if scaling:
        squared_norm = squared_norm / error.shape[0]
    # finally get either the squared norm or the norm
    if squared:
        result = squared_norm
    else:
        result = np.sqrt(squared_norm)

    return result
