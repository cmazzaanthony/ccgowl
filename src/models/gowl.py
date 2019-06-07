from src.models.functions.logdet import LOGDET
from src.models.functions.gowl import GOWL
from src.models.model import Model
import numpy as np


def _is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def _set_next_inital_step_size(theta_k1, theta_k):
    num = np.trace((theta_k1 - theta_k) @ (theta_k1 - theta_k))
    den = np.trace((theta_k1 - theta_k) @ (np.linalg.inv(theta_k) - np.linalg.inv(theta_k1)))
    return num / den


class GOWLModel(Model):

    def __init__(self, x, y):
        super(GOWLModel, self).__init__('GOWL', x, y)
        self.nsfunc = GOWL()
        self.sfunc = LOGDET()

    def _pgm_objective(self, theta, S, rho):
        """ Composite objective min f(x) + g(x) """
        return self.sfunc.eval(theta, S) + self.nsfunc.eval(theta, rho)

    def _quad_approx(self, theta_k1, theta_k, S, t_k):
        return -np.log(np.linalg.det(theta_k)) + \
               np.trace(S @ theta_k) + \
               np.trace((theta_k1 - theta_k) @ (S - np.linalg.inv(theta_k))) + \
               (1 / (2 * t_k)) * np.linalg.norm((theta_k1 - theta_k), ord='fro') ** 2

    def _step_size(self, theta_k, S, f_func, gradf, prox, _lambdas, t_init):
        c = 0.9
        j = 0
        t = t_init
        inequality_satisfied = True

        while inequality_satisfied:
            theta_k1 = prox(theta_k - t * gradf(theta_k, S), _lambdas)
            if _is_pos_def(theta_k1) and f_func(theta_k1, S) <= self._quad_approx(theta_k1, theta_k, S, t):
                break
            t = np.power(c, j)
            j += 1

        return t

    def _duality_gap(self, p, theta_k, S, _lambdas):
        return np.trace(S @ theta_k) + self.nsfunc.eval(theta_k, _lambdas) - p

    def _proximal_descent(self,
                          theta_star,
                          theta0,
                          S,
                          _lambdas,
                          gradf,
                          prox,
                          max_iter=None,
                          eps=1e-3,
                          use_dual_gap=True,
                          step_size_type='constant',
                          verbose=False):
        """
        G-ISTA algorithm

        https://papers.nips.cc/paper/4574-iterative-thresholding-algorithm-for-sparse-inverse-covariance-estimation.pdf
        """

        assert step_size_type in ['constant', 'backtracking'], "Choose a valid option: [constant, backtracking]"

        theta = theta0
        t = min(np.linalg.eigvals(theta0)) ** 2
        p = len(theta)

        if verbose:
            print(f'Objective with True Theta: {self._pgm_objective(theta_star, S, _lambdas, f_func, g_func)}')
            print(f'f(X,S) = {self.sfunc.eval(theta0, S)}')
            print(f'g(X,rho) = {self.nsfunc.eval(theta0, _lambdas)}')
            print(f'Initial Objective: {self._pgm_objective(theta0, S, _lambdas, f_func, g_func)}')

        if pgm_objective(theta0, S, _lambdas, f_func, g_func) > 10000:
            # Skip, bad starting point
            theta = make_sparse_spd_matrix(p, alpha=0.5, norm_diag=False, smallest_coef=-1.0, largest_coef=1.0)

        for i in range(max_iter):
            if not is_pos_def(theta):
                print('Clipped Precision matrix')
                theta = cov_nearest(theta, method="clipped", threshold=0.1)

            if step_size_type == 'backtracking':
                t = step_size(theta, S, f_func, gradf, prox, _lambdas, t)

            delta = duality_gap(p, theta, S, _lambdas)

            if verbose:
                print(f'Duality Gap: {delta}.')

            if delta < eps and use_dual_gap:
                print(f'iterations: {i}')
                print(f'Duality Gap: {delta} < {eps}. Exiting.')
                break

            theta_k1 = prox(theta - t * gradf(theta, S), _lambdas)

            if step_size_type == 'backtracking':
                t = set_next_inital_step_size(theta_k1, theta)

            theta = theta_k1

        return theta

    def fit(self):
