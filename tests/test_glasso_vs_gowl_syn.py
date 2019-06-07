import unittest

import numpy as np
from sklearn.covariance import GraphicalLasso
from sklearn.metrics import f1_score

from src.data.make_synthetic_data import generate_theta_star_gowl, standardize, Block
from src.evaluation.cluster_metrics import spectral_clustering
from src.evaluation.fit_metrics import error_norm
from src.models.gowl import GOWLModel
from src.models.utils import oscar_weights
from src.visualization.visualize import plot_multiple_theta_matrices_2d


def _fit_evaluations(true_theta, theta_hat, clusters, estimator_name):
    print(f"""# clusters: {clusters}, MSE: {error_norm(theta_star=true_theta,
                                                       theta_hat=theta_hat)} for {estimator_name}""")

    print(f"""# clusters: {clusters}, Absolute Error: {error_norm(theta_star=true_theta,
                                                                  theta_hat=theta_hat,
                                                                  squared=False)} for {estimator_name}""")


def _cluster_evaluations(y_true, y_hat, estimator_name):
    print(f"{estimator_name} F1 Score: {f1_score(y_true, y_hat, average='macro')}")


class TestSyntheticData(unittest.TestCase):

    def test_gowl_vs_glasso_1(self):
        np.random.seed(680)
        p = 10
        n_blocks = 1
        theta_star, blocks, theta_blocks = generate_theta_star_gowl(p=p,
                                                                    alpha=0.5,
                                                                    noise=0.1,
                                                                    n_blocks=n_blocks,
                                                                    block_min_size=2,
                                                                    block_max_size=6)

        lam1 = 0.001  # controls sparsity
        lam2 = 0.01  # encourages equality of coefficients
        rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)

        theta_star = theta_star[0]
        sigma = np.linalg.inv(theta_star)
        n = 100
        X = np.random.multivariate_normal(np.zeros(p), sigma, n)
        X = standardize(X)
        S = np.cov(X.T)

        theta_0 = np.linalg.inv(S)

        model = GOWLModel(X, S, theta_0, lam1, lam2, 'backtracking', max_iters=100000)
        model.fit()
        theta_gowl = model.theta_hat

        gl = GraphicalLasso()
        gl.fit(S)
        theta_glasso = gl.get_precision()

        print('Non zero entries in precision matrix {}'.format(np.count_nonzero(theta_gowl)))
        plot_multiple_theta_matrices_2d([theta_blocks, theta_star, theta_glasso, theta_gowl],
                                        [f"Blocks: {len(blocks)}", 'True Theta', 'GLASSO', 'GOWL'])
        _fit_evaluations(theta_star, theta_glasso, 1, 'GLASSO')
        _fit_evaluations(theta_star, theta_gowl, 1, 'GOWL')
        y_hat_gowl = spectral_clustering(theta=theta_gowl, K=2)
        y_hat_glasso = spectral_clustering(theta=theta_glasso, K=2)
        y_true = spectral_clustering(theta=theta_blocks, K=2).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'GOWL')
        _cluster_evaluations(y_true, y_hat_glasso, 'GLASSO')

    def test_gowl_vs_glasso_2(self):
        np.random.seed(680)
        p = 10
        blocks = [
            Block(dim=p,
                  idx=0,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=0.9),
            Block(dim=p,
                  idx=1,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=-0.9),
        ]
        theta_star, blocks, theta_blocks = generate_theta_star_gowl(p=p,
                                                                    alpha=0.5,
                                                                    noise=0.1,
                                                                    blocks=blocks)

        lam1 = 0.01  # controls sparsity
        lam2 = 0.01  # encourages equality of coefficients
        rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)

        theta_star = theta_star[0]
        sigma = np.linalg.inv(theta_star)
        n = 100
        X = np.random.multivariate_normal(np.zeros(p), sigma, n)
        X = standardize(X)
        S = np.cov(X.T)

        theta_0 = np.linalg.inv(S)
        model = GOWLModel(X, S, theta_0, lam1, lam2, 'backtracking', max_iters=100000)
        model.fit()
        theta_gowl = model.theta_hat

        gl = GraphicalLasso()
        gl.fit(S)
        theta_glasso = gl.get_precision()

        print('Non zero entries in precision matrix {}'.format(np.count_nonzero(theta_gowl)))
        plot_multiple_theta_matrices_2d([theta_blocks, theta_star, theta_glasso, theta_gowl],
                                        [f"Blocks: {len(blocks)}", 'True Theta', 'GLASSO', 'GOWL'])

        _fit_evaluations(theta_star, theta_glasso, 2, 'GLASSO')
        _fit_evaluations(theta_star, theta_gowl, 2, 'GOWL')

        y_hat_gowl = spectral_clustering(theta=theta_gowl, K=3)
        y_hat_glasso = spectral_clustering(theta=theta_glasso, K=3)
        y_true = spectral_clustering(theta=theta_blocks, K=3).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'GOWL')
        _cluster_evaluations(y_true, y_hat_glasso, 'GLASSO')

    def test_gowl_vs_glasso_3(self):
        np.random.seed(680)
        p = 10
        blocks = [
            Block(dim=p,
                  idx=0,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=0.9),
            Block(dim=p,
                  idx=1,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=-0.9),
            Block(dim=p,
                  idx=3,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=-0.5),
        ]
        theta_star, blocks, theta_blocks = generate_theta_star_gowl(p=p,
                                                                    alpha=0.5,
                                                                    noise=0.1,
                                                                    blocks=blocks)
        lam1 = 0.01  # controls sparsity
        lam2 = 0.01  # encourages equality of coefficients
        rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)

        theta_star = theta_star[0]
        sigma = np.linalg.inv(theta_star)
        n = 100
        X = np.random.multivariate_normal(np.zeros(p), sigma, n)
        X = standardize(X)
        S = np.cov(X.T)

        theta_0 = np.linalg.inv(S)
        model = GOWLModel(X, S, theta_0, lam1, lam2, 'backtracking', max_iters=100000)
        model.fit()
        theta_gowl = model.theta_hat

        gl = GraphicalLasso(max_iter=200)
        gl.fit(S)
        theta_glasso = gl.get_precision()

        print('Non zero entries in precision matrix {}'.format(np.count_nonzero(theta_gowl)))
        plot_multiple_theta_matrices_2d([theta_blocks, theta_star, theta_glasso, theta_gowl],
                                        [f"Blocks: {len(blocks)}", 'True Theta', 'GLASSO', 'GOWL'])

        _fit_evaluations(theta_star, theta_glasso, 3, 'GLASSO')
        _fit_evaluations(theta_star, theta_gowl, 3, 'GOWL')

        y_hat_gowl = spectral_clustering(theta=theta_gowl, K=4)
        y_hat_glasso = spectral_clustering(theta=theta_glasso, K=4)
        y_true = spectral_clustering(theta=theta_blocks, K=4).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'GOWL')
        _cluster_evaluations(y_true, y_hat_glasso, 'GLASSO')

    def test_gowl_vs_glasso_duality_gap_3(self):
        """
        Duality Gap goes negative in this case. Should that happen?
        """
        np.random.seed(680)
        p = 10
        blocks = [
            Block(dim=p,
                  idx=0,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=0.9),
            Block(dim=p,
                  idx=1,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=-0.9),
            Block(dim=p,
                  idx=3,
                  block_min_size=2,
                  block_max_size=6,
                  block_value=-0.5),
        ]
        theta_star, blocks, theta_blocks = generate_theta_star_gowl(p=p,
                                                                    alpha=0.5,
                                                                    noise=0.1,
                                                                    blocks=blocks)
        lam1 = 0.001  # controls sparsity
        lam2 = 0.01  # encourages equality of coefficients
        rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)

        theta_star = theta_star[0]
        sigma = np.linalg.inv(theta_star)
        n = 100
        X = np.random.multivariate_normal(np.zeros(p), sigma, n)
        X = standardize(X)
        S = np.cov(X.T)

        theta_0 = np.linalg.inv(S)
        model = GOWLModel(X, S, theta_0, lam1, lam2, 'backtracking', max_iters=100000)
        model.fit()
        theta_gowl = model.theta_hat

        gl = GraphicalLasso(max_iter=200)
        gl.fit(S)
        theta_glasso = gl.get_precision()

        print('Non zero entries in precision matrix {}'.format(np.count_nonzero(theta_gowl)))
        plot_multiple_theta_matrices_2d([theta_blocks, theta_star, theta_glasso, theta_gowl],
                                        [f"Blocks: {len(blocks)}", 'True Theta', 'GLASSO', 'GOWL'])

        _fit_evaluations(theta_star, theta_glasso, 3, 'GLASSO')
        _fit_evaluations(theta_star, theta_gowl, 3, 'GOWL')

        y_hat_gowl = spectral_clustering(theta=theta_gowl, K=4)
        y_hat_glasso = spectral_clustering(theta=theta_glasso, K=4)
        y_true = spectral_clustering(theta=theta_blocks, K=4).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'GOWL')
        _cluster_evaluations(y_true, y_hat_glasso, 'GLASSO')


if __name__ == "__main__":
    unittest.main()
