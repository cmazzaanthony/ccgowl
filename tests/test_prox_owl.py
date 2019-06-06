import unittest

import numpy as np

from gowl.prox.prox_owl import prox_graph_owl, gowl_penalty


class TestProxOWL(unittest.TestCase):

    def test_prox_graph_owl_pos(self):
        _lambdas = np.array([0.3, 0.2, 0.1])
        X = np.arange(0, 9).reshape(3, 3).astype(float)

        X = prox_graph_owl(X, _lambdas)

        expected_X = np.array([[0.0, 2.9, 5.8],
                               [2.9, 4.0, 6.7],
                               [5.8, 6.7, 8.0]])

        np.testing.assert_array_equal(expected_X, X)

    def test_prox_graph_owl_neg(self):
        _lambdas = np.array([0.3, 0.2, 0.1])
        X = - np.arange(0, 9).reshape(3, 3).astype(float)

        X = prox_graph_owl(X, _lambdas)

        expected_X = - np.array([[0.0, 2.9, 5.8],
                                 [2.9, 4.0, 6.7],
                                 [5.8, 6.7, 8.0]])

        np.testing.assert_array_equal(expected_X, X)

    def test_gowl_penalty(self):
        _lambdas = np.array([0.3, 0.2, 0.1])
        X = np.arange(0, 9).reshape(3, 3).astype(float)

        owl_val = gowl_penalty(X, _lambdas)

        expected = _lambdas.dot([7, 6, 3])
        self.assertEqual(expected, owl_val)
