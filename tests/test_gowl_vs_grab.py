import unittest

import numpy as np
from sklearn.metrics import f1_score

import grab.GRAB as grab
from gowl.algorithms.proximal_descent import proximal_descent, pgm_objective
from gowl.data.real_data import standardize
from gowl.data.synthetic_data import generate_theta_star_gowl, Block
from gowl.evaluation.cluster_metrics import spectral_clustering
from gowl.evaluation.fit_metrics import error_norm
from gowl.loss.loss_functions import grad_log_det_loss, log_det_loss
from gowl.prox.prox_owl import prox_graph_owl, oscar_weights, gowl_penalty
from gowl.simulations.synthetic_results_grid_search import generate_synthetic_data, run_gowl
from gowl.visualization.graph_visualization import plot_multiple_theta_matrices_2d


def _fit_evaluations(true_theta, theta_hat, clusters, estimator_name):
    print(f"""# clusters: {clusters}, MSE: {error_norm(theta_star=true_theta,
                                                       theta_hat=theta_hat)} for {estimator_name}""")

    print(f"""# clusters: {clusters}, Absolute Error: {error_norm(theta_star=true_theta,
                                                                  theta_hat=theta_hat,
                                                                  squared=False)} for {estimator_name}""")


def _cluster_evaluations(y_true, y_hat, estimator_name):
    print(f"{estimator_name} F1 Score: {f1_score(y_true, y_hat, average='macro')}")


class TestGRABEstimator(unittest.TestCase):

    def test_gowl_vs_grab_1(self):
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

        lam1 = 0.001  # controls sparsity
        lam2 = 0.01  # encourages equality of coefficients
        rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)
        print(f'True Theta Objective: {pgm_objective(theta_star, S, rho, log_det_loss, gowl_penalty)}')

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

        theta_0 = np.linalg.inv(S)
        theta_owl = proximal_descent(theta_star,
                                     theta_0,
                                     S,
                                     rho,
                                     gradf=grad_log_det_loss,
                                     prox=prox_graph_owl,
                                     f_func=log_det_loss,
                                     g_func=gowl_penalty,
                                     max_iter=100000,
                                     step_size_type='backtracking',
                                     use_dual_gap=True,
                                     verbose=True)

        print('Non zero entries in precision matrix {}'.format(np.count_nonzero(theta_owl)))
        plot_multiple_theta_matrices_2d([S, theta_blocks, theta_star, theta_grab, theta_owl],
                                        ['Sample Covariance', f"Blocks: {len(blocks)}", 'True Theta', 'GRAB', 'GOWL'])

        _fit_evaluations(theta_star, theta_grab, 1, 'GRAB')
        _fit_evaluations(theta_star, theta_owl, 1, 'GOWL')

        y_hat_gowl = spectral_clustering(theta=theta_owl, K=2)
        y_hat_grab = spectral_clustering(theta=theta_grab, K=2)
        y_true = spectral_clustering(theta=theta_blocks, K=2).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'GOWL')
        _cluster_evaluations(y_true, y_hat_grab, 'GRAB')

    def test_gowl_vs_grab_2(self):
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
        theta_star = theta_star[0]
        sigma = np.linalg.inv(theta_star)
        n = 100
        X = np.random.multivariate_normal(np.zeros(p), sigma, n)
        X = standardize(X)
        S = np.cov(X.T)

        lam1 = 0.01  # controls sparsity
        lam2 = 0.01  # encourages equality of coefficients
        rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)
        print(f'True Theta Objective: {pgm_objective(theta_star, S, rho, log_det_loss, gowl_penalty)}')

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

        theta_0 = np.linalg.inv(S)
        theta_owl = proximal_descent(theta_star,
                                     theta_0,
                                     S,
                                     rho,
                                     gradf=grad_log_det_loss,
                                     prox=prox_graph_owl,
                                     f_func=log_det_loss,
                                     g_func=gowl_penalty,
                                     max_iter=100000,
                                     step_size_type='backtracking',
                                     use_dual_gap=True,
                                     verbose=True)

        print('Non zero entries in precision matrix {}'.format(np.count_nonzero(theta_owl)))
        plot_multiple_theta_matrices_2d([S, theta_blocks, theta_star, theta_grab, theta_owl],
                                        ['Sample Covariance', f"Blocks: {len(blocks)}", 'True Theta', 'GRAB', 'GOWL'])

        _fit_evaluations(theta_star, theta_grab, 1, 'GRAB')
        _fit_evaluations(theta_star, theta_owl, 1, 'GOWL')

        y_hat_gowl = spectral_clustering(theta=theta_owl, K=3)
        y_hat_grab = spectral_clustering(theta=theta_grab, K=3)
        y_true = spectral_clustering(theta=theta_blocks, K=3).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'GOWL')
        _cluster_evaluations(y_true, y_hat_grab, 'GRAB')

    def test_gowl_vs_grab_3(self):
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
        theta_owl = proximal_descent(theta_star,
                                     theta_0,
                                     S,
                                     rho,
                                     gradf=grad_log_det_loss,
                                     prox=prox_graph_owl,
                                     f_func=log_det_loss,
                                     g_func=gowl_penalty,
                                     max_iter=100000,
                                     step_size_type='backtracking',
                                     use_dual_gap=True,
                                     verbose=True)

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
                                        ['Sample Covariance', f"Blocks: {len(blocks)}", 'True Theta', 'GRAB', 'GOWL'])

        _fit_evaluations(theta_star, theta_grab, 4, 'GRAB')
        _fit_evaluations(theta_star, theta_owl, 4, 'GOWL')

        y_hat_gowl = spectral_clustering(theta=theta_owl, K=4)
        y_hat_grab = spectral_clustering(theta=theta_grab, K=4)
        y_true = spectral_clustering(theta=theta_blocks, K=4).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'GOWL')
        _cluster_evaluations(y_true, y_hat_grab, 'GRAB')

    def test_gowl_vs_grab_n(self):
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
        ccl_1 = 0.01  # p = 50
        ccl_2 = 0.0001  # p = 50

        thetas_with_noise, theta_blocks, scov_matrices, X_matrices = generate_synthetic_data(K,
                                                                                             p,
                                                                                             block_min_size,
                                                                                             block_max_size,
                                                                                             alpha,
                                                                                             noise,
                                                                                             1)
        theta_star = thetas_with_noise[0]
        S = scov_matrices[0]
        theta_gowl = run_gowl(theta_star, S, ccl_1, ccl_2)

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

        plot_multiple_theta_matrices_2d([S, theta_blocks, thetas_with_noise[0], theta_grab, theta_gowl],
                                        ['Sample Covariance', f"Blocks: {len(blocks)}", 'True Theta', 'GRAB', 'CCGOWL'])

        _fit_evaluations(theta_blocks, theta_grab, 1, 'GRAB')
        _fit_evaluations(theta_blocks, theta_gowl, 1, 'CCGOWL')

        y_hat_gowl = spectral_clustering(theta=theta_gowl, K=2)
        y_hat_grab = spectral_clustering(theta=theta_grab, K=2)
        y_true = spectral_clustering(theta=theta_blocks, K=2).flatten()
        _cluster_evaluations(y_true, y_hat_gowl, 'CCGOWL')
        _cluster_evaluations(y_true, y_hat_grab, 'GRAB')
