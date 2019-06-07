from src.models.functions.mse import MSE
from src.models.functions.owl import OWL
from src.models.model import Model
import numpy as np

from src.models.utils import oscar_weights


class CCGOWL(Model):

    def __init__(self, x, y, lam1, lam2, max_iters=10, epsilon=1e-05):
        super(CCGOWL, self).__init__('CCGOWL', x, y)
        self.nsfunc = OWL()
        self.sfunc = MSE()
        self._lambdas = oscar_weights(lam1, lam2)
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.beta_0 = np.zeros(x.shape[1] - 1)
        self.beta = np.zeros(x.shape[1] - 1)

    def _backtracking_line_search(self, x, beta, y, weights, max_iter=20, epsilon=2.0):
        mse_val = self.sfunc.eval(x, beta, y)
        mse_grad_val = self.sfunc.gradient(x, beta, y)
        step = 1.0

        for ls in range(max_iter):
            beta_prox = self.nsfunc.prox(beta - mse_grad_val / step, weights / step)
            delta = (beta_prox - beta).flatten()

            orig_func = self.sfunc.eval(x, beta_prox, y)
            quad_func = mse_val + delta @ mse_grad_val.flatten() + 0.5 * step * (delta @ delta)

            if orig_func <= quad_func:
                break

            step *= epsilon

        return beta_prox

    def _proximal_grad_descent(self, design_matrix, response, lambdas):
        beta = self.beta_0
        for iterations in range(self.max_iters):

            next_beta = self._backtracking_line_search(design_matrix, beta, response, lambdas)

            if np.linalg.norm(next_beta - beta) <= self.epsilon:
                print(f'Threshold reached in {iterations}')
                break

            beta = next_beta

        return beta

    def fit(self):
        beta = self._proximal_grad_descent(self.X, self.Y, self._lambdas)
        self.beta = beta
