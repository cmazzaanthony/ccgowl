import numpy as np

from src.models.functions.function import Function
from src.models.functions.owl import OWL


def _get_off_diagonal_entries(x):
    lt_indices = np.tril_indices_from(x, -1)
    lt_indices = list(zip(*lt_indices))
    return lt_indices, np.array([x[i][j] for i, j in lt_indices])


class GOWL(Function):

    def eval(self, x, weights):
        """
        g(X) = sum_{i=1}^p rho_i * |x|_[i]
        :param x: (p x p) matrix
        :param weights: weights for owl penalty
        """
        nsfunc = OWL()
        _, off_diagonal_entries = _get_off_diagonal_entries(x)
        return nsfunc.eval(off_diagonal_entries, weights)

    def gradient(self, beta, weights):
        raise NotImplementedError("The OWL function is a non-smooth function. \n"
                                  "Please call the prox function.")

    def prox(self, x, weights):
        """
        :param x: (p x p) matrix
        :param weights: weights for owl penalty
        """
        nsfunc = OWL()
        lt_indices, off_diagonal_entries = _get_off_diagonal_entries(x)

        prox_x = nsfunc.prox(off_diagonal_entries, weights)

        for i, pair in enumerate(lt_indices):
            x[pair] = prox_x[i]

        return np.tril(x, -1) + np.tril(x).T

    def hessian(self):
        raise NotImplementedError("The OWL function is a non-smooth function. \n"
                                  "Please call the prox function.")
