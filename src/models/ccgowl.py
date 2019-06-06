from src.models.functions.mse import MSE
from src.models.functions.owl import OWL
from src.models.model import Model
import numpy as np


class CCGOWL(Model):

    def __init__(self, x, y):
        super(CCGOWL, self).__init__('CCGOWL', x, y)
        self.nsfunc = OWL()
        self.sfunc = MSE()

    def backtracking_line_search(self, x, beta, y, weights, max_iter=20, epsilon=2.0):
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

    def proximal_grad_descent(self, beta_0, weights, X, Y, max_iter=10, epsilon=1e-05):
        beta = beta_0
        for iterations in range(max_iter):

            next_beta = self.backtracking_line_search(X, beta, Y, weights)

            if np.linalg.norm(next_beta - beta) <= epsilon:
                print(f'Threshold reached in {iterations}')
                break

            beta = next_beta

        return beta

    def fit(self):
        pass
