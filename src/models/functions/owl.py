import numpy as np
from sklearn.isotonic import isotonic_regression

from src.models.functions.function import Function


def _get_off_diagonal_entries(x):
    lt_indices = np.tril_indices_from(x, -1)
    lt_indices = list(zip(*lt_indices))
    return lt_indices, np.array([x[i][j] for i, j in lt_indices])


class OWL(Function):

    def eval(self, beta, weights):
        v_abs = np.abs(beta)
        ix = np.argsort(v_abs)[::-1]
        v_abs = v_abs[ix]
        return weights.dot(v_abs)

    def gradient(self, beta, weights):
        raise NotImplementedError("The OWL function is a non-smooth function. \n"
                                  "Please call the prox function.")

    def gowl_penalty(self, x, weights):
        """ g(X) = sum_{i=1}^p rho_i * |x|_[i] """
        _, off_diagonal_entries = _get_off_diagonal_entries(x)
        return self.eval(off_diagonal_entries, weights)

    def prox(self, beta, weights):
        """
        X. Zeng, M. Figueiredo,
        The ordered weighted L1 norm: Atomic formulation, dual norm,
        and projections.
        eprint http://arxiv.org/abs/1409.4271
        """
        p = len(beta)
        abs_beta = np.abs(beta)
        # returns indices that would sort the array and then reverse the order to descending
        ix = np.argsort(abs_beta)[::-1]
        abs_beta = abs_beta[ix]
        iso_input = abs_beta - weights
        abs_beta = isotonic_regression(iso_input, y_min=0, increasing=False)

        idxs = np.zeros_like(ix)
        idxs[ix] = np.arange(p)
        abs_beta = abs_beta[idxs]

        beta = np.sign(beta) * abs_beta
        return beta

    def hessian(self):
        raise NotImplementedError("The OWL function is a non-smooth function. \n"
                                  "Please call the prox function.")
