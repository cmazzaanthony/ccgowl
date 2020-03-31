from abc import ABC, abstractmethod


class Function(ABC):

    @abstractmethod
    def eval(self, *arg, **kwargs):
        pass

    @abstractmethod
    def gradient(self, *arg, **kwargs):
        pass

    @abstractmethod
    def prox(self, *arg, **kwargs):
        pass

    @abstractmethod
    def hessian(self, *arg, **kwargs):
        pass
