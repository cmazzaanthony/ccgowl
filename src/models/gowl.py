from src.models.model import Model
import numpy as np


class GOWL(Model):

    def __init__(self, x, y):
        super(GOWL, self).__init__('GOWL', x, y)

