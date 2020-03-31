import pathlib
import pickle
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, name, x, y):
        self.name = name
        self.X = x
        self.Y = y

    @abstractmethod
    def fit(self, *arg, **kwargs):
        pass

    def save(self, fmt='pickle'):
        proj_root_path = pathlib.Path.cwd().parent.parent
        models_path = 'models/'
        if fmt == 'pickle':
            model_file_name = f"{self.name}_model.pkl"
            full_path = proj_root_path / models_path / model_file_name
            pickle.dump(self, open(full_path, 'wb'))

        raise NotImplementedError("Only pickle format is implemented")
