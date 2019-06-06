from src.models.functions.function import Function
import numpy as np


class LOGDET(Function):

    def eval(self, d_matrix, s_cov):
        """f(X) = -log det(X)+tr(SX)"""
        return -np.log(np.linalg.det(d_matrix)) + np.trace(s_cov.dot(d_matrix))

    def gradient(self, theta, s_cov):
        """df = S - inv(theta)"""
        return s_cov - np.linalg.inv(theta)

    def prox(self, *arg, **kwargs):
        pass

    def hessian(self, *arg, **kwargs):
        pass
