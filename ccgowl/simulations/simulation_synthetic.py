import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.model_selection import KFold

import ccgowl.models.grab.GRAB as grab
from ccgowl.data.make_synthetic_data import generate_theta_star_gowl, standardize, Block
from ccgowl.evaluation.cluster_metrics import spectral_clustering
from ccgowl.evaluation.fit_metrics import error_norm
from ccgowl.models.ccgowl import CCGOWLModel
from ccgowl.models.gowl import GOWLModel
import itertools
from ccgowl.visualization.visualize import plot_multiple_theta_matrices_2d


def _fit_evaluations(true_theta, theta_hat):
    return {
        'MSE': error_norm(theta_star=true_theta, theta_hat=theta_hat),
        'AE': error_norm(theta_star=true_theta, theta_hat=theta_hat, squared=False)
    }


def _cluster_evaluations(y_true, y_hat):
    cm = confusion_matrix(y_true, y_hat)
    if len(cm) == 1:
        return 1,1,1
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    
    return f1_score(y_true, y_hat, average='macro'), sensitivity, specificity


def generate_blocks(n_blocks, p, min_size, max_size):
    block_list = list()
    for i in range(n_blocks):
        if i % 2 == 0:
            block_value = round(np.random.uniform(0.7, 1), 2)
        else:
            block_value = round(np.random.uniform(-0.7, -1), 2)
        b = Block(dim=p,
                  idx=i,
                  block_min_size=min_size,
                  block_max_size=max_size,
                  block_value=block_value)

        block_list.append(b)

    return block_list


def run(n, p, kappa, method):
    alpha = 0.5
    noise = 0.1
    K = int(p * kappa)
    block_min_size = p * 0.1
    block_max_size = p * 0.4
    blocks = generate_blocks(K, p, block_min_size, block_max_size)
    theta_star, blocks, theta_blocks = generate_theta_star_gowl(p=p,
                                                                alpha=alpha,
                                                                noise=noise,
                                                                blocks=blocks)
    theta_star = theta_star[0]
    sigma = np.linalg.inv(theta_star)
    kf = KFold(n_splits=2)
    lambda_1 = np.linspace(0, 0.1, 10)
    lambda_2 = np.linspace(0, 0.1, 10)
    X = np.random.multivariate_normal(np.zeros(p), sigma, n)
    X = standardize(X)

    if method == 'grab':
        result = run_experiment(
            K,
            0.0,
            0.0,
            X,
            theta_star,
            theta_blocks,
            method
        )
        print(result)

    df_list = list()
    for lam1, lam2 in list(itertools.product(lambda_1, lambda_2)):
        results = list()
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            result = run_experiment(
                K,
                lam1,
                lam2,
                X_train,
                theta_star,
                theta_blocks,
                method
            )
            results.append(result['F1'])

        df_list.append([lam1, lam2, np.average(results)])

    df = pd.DataFrame(df_list, columns=['lam1', 'lam2', 'error'])
    df.sort_values(by='error', ascending=False, inplace=True)
    selected_params = df.head(1).to_dict(orient="records")[0]
    result = run_experiment(
        K,
        selected_params['lam1'],
        selected_params['lam2'],
        X,
        theta_star,
        theta_blocks,
        method
    )
    print(result)


def run_experiment(K, lam1, lam2, X, theta_star, theta_blocks, method):
    S = np.cov(X.T)

    if method == 'gowl':
        model = GOWLModel(X, S, lam1, lam2, 'backtracking', max_iters=100000)
        model.fit()
        theta_hat = model.theta_hat
    elif method == 'ccgowl':
        model = CCGOWLModel(X, lam1, lam2)
        model.fit()
        theta_hat = model.theta_hat
    else:
        lmbda = .2
        o_size = .3  # The size of overlap, as an input parameter
        max_iter = 20
        tol = 1e-4
        dual_max_iter = 600
        dual_tol = 1e-4
        theta_hat, blocks = grab.BCD(S,
                                     lmbda=lmbda,
                                     K=K,
                                     o_size=o_size,
                                     max_iter=max_iter,
                                     tol=tol,
                                     dual_max_iter=dual_max_iter,
                                     dual_tol=dual_tol)
        theta_hat = np.asarray(theta_hat)

    # plot_multiple_theta_matrices_2d([theta_star, theta_grab, theta_gowl, theta_ccgowl],
    #                                 ['True Theta', 'GRAB', 'GOWL', 'CCGOWL'])

    theta_fit = _fit_evaluations(theta_star, theta_hat)
    y_hat_theta = spectral_clustering(theta=theta_hat, K=K)
    y_true = spectral_clustering(theta=theta_blocks, K=K).flatten()
    f1, sensitivity, specificity = _cluster_evaluations(y_true, y_hat_theta)

    return {
        'Fit': theta_fit,
        'F1': f1,
        'sensitivity':sensitivity,
        'specificity': specificity
    }


if __name__ == '__main__':
    # p = 25
    # kappa = 0.1
    # n = 2000

    df = []
    for method in ['grab']:
        for p in [15,25]:
            for kappa in [0.1,0.2]:
                for n in [1000,2000]:
                    print('here')
                    d = run(n, p, kappa, method)
                    df.append( { 'p':p, '\kappa':kappa, 'n':n, 'method':method, uppercase(method)+'$/F_1$':d['F1'],uppercase(method)+'/sensitivity':d['sensitivity'], uppercase(method)+'specificity':d['specificity']} )

    df = pd.DataFrame(df)
    print(df)