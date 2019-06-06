from src.models.functions.function import Function


class MSE(Function):

    def eval(self, x, beta, y):
        diff = x @ beta - y
        return (1 / len(y)) * 0.5 * diff.T @ diff

    def gradient(self, x, beta, y):
        return (1 / len(y)) * -x.T @ (y - x @ beta)

    def prox(self, *arg, **kwargs):
        pass

    def hessian(self):
        pass
