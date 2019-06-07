import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
from statsmodels.stats.correlation_tools import cov_nearest

from src.models.functions.gowl import GOWL
from src.models.functions.logdet import LOGDET
from src.models.model import Model
from src.models.utils import oscar_weights


def _is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def _set_next_inital_step_size(theta_k1, theta_k):
    num = np.trace((theta_k1 - theta_k) @ (theta_k1 - theta_k))
    den = np.trace((theta_k1 - theta_k) @ (np.linalg.inv(theta_k) - np.linalg.inv(theta_k1)))
    return num / den


class GOWLModel(Model):

    def __init__(self,
                 x,
                 sample_cov,
                 theta0,
                 lam1,
                 lam2,
                 ss_type,
                 dual_gap=True,
                 max_iters=10,
                 epsilon=1e-05):
        super(GOWLModel, self).__init__('GOWL', x, None)
        _, p = sample_cov.shape
        self.S = sample_cov
        self.nsfunc = GOWL()
        self.sfunc = LOGDET()
        self.theta0 = theta0
        self._lambdas = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)
        self.dual_gap = dual_gap
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.theta_hat = np.zeros(self.S.shape)

        assert ss_type in ['constant', 'backtracking'], \
            "Choose a valid option: [constant, backtracking]"
        self.ss_type = ss_type

    def _pgm_objective(self, theta, S, rho):
        """ Composite objective min f(x) + g(x) """
        return self.sfunc.eval(theta, S) + self.nsfunc.eval(theta, rho)

    def _quad_approx(self, theta_k1, theta_k, S, t_k):
        return -np.log(np.linalg.det(theta_k)) + \
               np.trace(S @ theta_k) + \
               np.trace((theta_k1 - theta_k) @ (S - np.linalg.inv(theta_k))) + \
               (1 / (2 * t_k)) * np.linalg.norm((theta_k1 - theta_k), ord='fro') ** 2

    def _step_size(self, theta_k, S, _lambdas, t_init):
        c = 0.9
        j = 0
        t = t_init
        inequality_satisfied = True

        while inequality_satisfied:
            theta_k1 = self.nsfunc.prox(theta_k - t * self.sfunc.gradient(theta_k, S), _lambdas)
            if _is_pos_def(theta_k1) and self.sfunc.eval(theta_k1, S) <= self._quad_approx(theta_k1, theta_k, S, t):
                break
            t = np.power(c, j)
            j += 1

        return t

    def _duality_gap(self, p, theta_k, S, _lambdas):
        return np.trace(S @ theta_k) + self.nsfunc.eval(theta_k, _lambdas) - p

    def _gista(self,
               theta0,
               S,
               _lambdas,
               verbose=False):
        """
        G-ISTA algorithm

        https://papers.nips.cc/paper/4574-iterative-thresholding-algorithm-for-sparse-inverse-covariance-estimation.pdf
        """

        theta = theta0
        t = min(np.linalg.eigvals(theta0)) ** 2
        p = len(theta)

        if verbose:
            print(f'f(X,S) = {self.sfunc.eval(theta0, S)}')
            print(f'g(X,rho) = {self.nsfunc.eval(theta0, _lambdas)}')
            print(f'Initial Objective: {self._pgm_objective(theta0, S, _lambdas)}')

        if self._pgm_objective(self.theta0, S, _lambdas) > 10000:
            # Skip, bad starting point
            theta = make_sparse_spd_matrix(p, alpha=0.5, norm_diag=False, smallest_coef=-1.0, largest_coef=1.0)

        for i in range(self.max_iters):
            if not _is_pos_def(theta):
                print('Clipped Precision matrix')
                theta = cov_nearest(theta, method="clipped", threshold=0.1)

            if self.ss_type == 'backtracking':
                t = self._step_size(theta, S, _lambdas, t)

            delta = self._duality_gap(p, theta, S, _lambdas)

            if verbose:
                print(f'Duality Gap: {delta}.')

            if delta < self.epsilon and self.dual_gap:
                print(f'iterations: {i}')
                print(f'Duality Gap: {delta} < {self.epsilon}. Exiting.')
                break

            theta_k1 = self.nsfunc.prox(theta - t * self.sfunc.gradient(theta, S), _lambdas)

            if self.ss_type == 'backtracking':
                t = _set_next_inital_step_size(theta_k1, theta)

            theta = theta_k1

        return theta

    def fit(self):
        self.theta_hat = self._gista(self.theta0, self.S, self._lambdas)
