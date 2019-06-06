import numpy as np


def oscar_weights(lam1, lam2, p):
    """ w_i = lambda_1 + lambda_2 (n - i) """
    _lamdbas = np.arange(p - 1, -1, -1, dtype=np.double)
    _lamdbas *= lam2
    _lamdbas += lam1
    return _lamdbas
