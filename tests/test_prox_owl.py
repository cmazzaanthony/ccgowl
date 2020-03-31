import unittest

import numpy as np

from ccgowl.models.functions.gowl import GOWL


class TestProxOWL(unittest.TestCase):

    def test_prox_graph_owl_pos(self):
        _lambdas = np.array([0.3, 0.2, 0.1])
        x = np.arange(0, 9).reshape(3, 3).astype(float)

        nsfunc = GOWL()
        x = nsfunc.prox(x, _lambdas)

        expected_x = np.array([[0.0, 2.9, 5.8],
                               [2.9, 4.0, 6.7],
                               [5.8, 6.7, 8.0]])

        np.testing.assert_array_equal(expected_x, x)

    def test_prox_graph_owl_neg(self):
        _lambdas = np.array([0.3, 0.2, 0.1])
        x = - np.arange(0, 9).reshape(3, 3).astype(float)

        nsfunc = GOWL()
        x = nsfunc.prox(x, _lambdas)

        expected_x = - np.array([[0.0, 2.9, 5.8],
                                 [2.9, 4.0, 6.7],
                                 [5.8, 6.7, 8.0]])

        np.testing.assert_array_equal(expected_x, x)

    def test_gowl_penalty(self):
        _lambdas = np.array([0.3, 0.2, 0.1])
        x = np.arange(0, 9).reshape(3, 3).astype(float)

        nsfunc = GOWL()
        gowl_val = nsfunc.eval(x, _lambdas)

        expected = _lambdas.dot([7, 6, 3])
        self.assertEqual(expected, gowl_val)
