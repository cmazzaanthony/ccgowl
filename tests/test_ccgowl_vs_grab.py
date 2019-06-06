import unittest

import numpy as np
from sklearn.metrics import f1_score

import grab.GRAB as grab
from gowl.ccgowl.descent import proximal_grad_descent, mse
from gowl.data.real_data import standardize
from gowl.data.synthetic_data import generate_theta_star_gowl
from gowl.evaluation.cluster_metrics import spectral_clustering
from gowl.evaluation.fit_metrics import error_norm
from gowl.prox.prox_owl import oscar_weights
from gowl.simulations.synthetic_results_grid_search import generate_blocks, generate_synthetic_data, run_ccgowl
from gowl.visualization.graph_visualization import plot_multiple_theta_matrices_2d


def _fit_evaluations(true_theta, theta_hat, clusters, estimator_name):
    print(f"""# clusters: {clusters}, MSE: {error_norm(theta_star=true_theta,
                                                       theta_hat=theta_hat)} for {estimator_name}""")

    print(f"""# clusters: {clusters}, Absolute Error: {error_norm(theta_star=true_theta,
                                                                  theta_hat=theta_hat,
                                                                  squared=False)} for {estimator_name}""")


def _cluster_evaluations(y_true, y_hat, estimator_name):
    print(f"{estimator_name} F1 Score: {f1_score(y_true, y_hat, average='macro')}")


class TestCCGOWLvsGRABEstimator(unittest.TestCase):

    def test_ccgowl_vs_grab_1(self):
        np.random.seed(680)
        p = 10
        n_blocks = 1
        theta_star, blocks, theta_blocks = generate_theta_star_gowl(p=p,
                                                                    alpha=0.5,
                                                                    noise=0.1,
                                                                    n_blocks=n_blocks,
                                                                    block_min_size=2,
                                                                    block_max_size=6)
        theta_star = theta_star[0]
        sigma = np.linalg.inv(theta_star)
        n = 100
        X = np.random.multivariate_normal(np.zeros(p), sigma, n)
        X = standardize(X)
        S = np.cov(X.T)

        lam1 = 0.05263158
        lam2 = 0.05263158
        beta_0 = np.zeros(p - 1)
        weights = oscar_weights(lam1, lam2, p - 1)

        column_idxs = np.arange(p)
        theta_owl = np.zeros((p, p))
        for j in range(p - 1):
            y_j = X[:, j]
            idx_j = np.delete(column_idxs, j)
            X_j = X[:, idx_j]
            beta_j = proximal_grad_descent(beta_0,
                                           weights,
                                           X_j,
                                           y_j)

            # sigma_j_sq = np.divide(mse(X_j, beta_j, y_j), n)  # sigma is 1 since standardized
            sigma_j_sq = ((np.linalg.norm(X_j @ beta_j - y_j)) ** 2) / n
            theta_j = - (1 / sigma_j_sq) * beta_j
            theta_j = np.insert(theta_j, j, 1.0)  # insert diagonal entry
            theta_owl[:, j] = theta_j

        lmbda = .2
        K = 10
        o_size = .3  # The size of overlap, as an input parameter
        max_iter = 20
        tol = 1e-4
        dual_max_iter = 600
        dual_tol = 1e-4
        theta_grab, blocks = grab.BCD(S,
                                      lmbda=lmbda,
                                      K=K,
                                      o_size=o_size,
                                      max_iter=max_iter,
                                      tol=tol,
                                      dual_max_iter=dual_max_iter,
                                      dual_tol=dual_tol)

        theta_grab = np.asarray(theta_grab)

        print('Non zero entries in precision matrix {}'.format(np.count_nonzero(theta_owl)))
        plot_multiple_theta_matrices_2d([S, theta_blocks, theta_star, theta_grab, theta_owl],
                                        ['Sample Covariance', f"Blocks: {len(blocks)}", 'True Theta', 'GRAB', 'CCGOWL'])

        _fit_evaluations(theta_star, theta_grab, 1, 'GRAB')
        _fit_evaluations(theta_star, theta_owl, 1, 'GOWL')

        y_hat_gowl = spectral_clustering(theta=theta_owl, K=2)
        y_hat_grab = spectral_clustering(theta=theta_grab, K=2)
        y_true = spectral_clustering(theta=theta_blocks, K=2).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'CCGOWL')
        _cluster_evaluations(y_true, y_hat_grab, 'GRAB')

    def test_ccgowl_vs_grab_2(self):
        np.random.seed(680)
        p = 50
        block_percentage = 0.1

        alpha = 0.5
        noise = 0.1
        K = int(p * block_percentage)
        block_min_size = p * 0.1
        block_max_size = p * 0.4
        # ccl_1 = 0.011 # p = 100
        # ccl_2 = 0.002 # p = 100
        ccl_1 = 0.1  # p = 50
        ccl_2 = 0.004 # p = 50

        thetas_with_noise, theta_blocks, scov_matrices, X_matrices = generate_synthetic_data(K,
                                                                                             p,
                                                                                             block_min_size,
                                                                                             block_max_size,
                                                                                             alpha,
                                                                                             noise,
                                                                                             1)

        S = scov_matrices[0]
        theta_ccgowl = run_ccgowl(X_matrices[0], ccl_1, ccl_2)

        lmbda = .2
        K = 10
        o_size = .3  # The size of overlap, as an input parameter
        max_iter = 20
        tol = 1e-4
        dual_max_iter = 600
        dual_tol = 1e-4
        theta_grab, blocks = grab.BCD(S,
                                      lmbda=lmbda,
                                      K=K,
                                      o_size=o_size,
                                      max_iter=max_iter,
                                      tol=tol,
                                      dual_max_iter=dual_max_iter,
                                      dual_tol=dual_tol)

        theta_grab = np.asarray(theta_grab)

        plot_multiple_theta_matrices_2d([S, theta_blocks, thetas_with_noise[0], theta_grab, theta_ccgowl],
                                        ['Sample Covariance', f"Blocks: {len(blocks)}", 'True Theta', 'GRAB', 'CCGOWL'])

        _fit_evaluations(theta_blocks, theta_grab, 1, 'GRAB')
        _fit_evaluations(theta_blocks, theta_ccgowl, 1, 'CCGOWL')

        y_hat_gowl = spectral_clustering(theta=theta_ccgowl, K=2)
        y_hat_grab = spectral_clustering(theta=theta_grab, K=2)
        y_true = spectral_clustering(theta=theta_blocks, K=2).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'CCGOWL')
        _cluster_evaluations(y_true, y_hat_grab, 'GRAB')

