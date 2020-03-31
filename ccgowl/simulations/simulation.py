from abc import ABC, abstractmethod


class Simulation(ABC):

    @abstractmethod
    def run(self, *arg, **kwargs):
        pass

    @abstractmethod
    def plot_results(self, *arg, **kwargs):
        pass
