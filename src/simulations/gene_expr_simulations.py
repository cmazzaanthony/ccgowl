import operator
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.covariance import GraphicalLasso
from sklearn.datasets import make_sparse_spd_matrix
from statsmodels.stats.correlation_tools import cov_nearest

import grab.GRAB as grab
from gowl.algorithms.proximal_descent import proximal_descent
from gowl.data.real_data import load_gene_subset
from gowl.evaluation.cluster_metrics import spectral_clustering
from gowl.loss.loss_functions import grad_log_det_loss, log_det_loss
from gowl.prox.prox_owl import prox_graph_owl, oscar_weights, gowl_penalty
from gowl.simulations.stock_data_simulations import plot_with_annotations, compute_true_group
from gowl.simulations.synthetic_results_grid_search import run_ccgowl
from gowl.visualization.graph_visualization import plot_theta_matrix_2d


def convert_to_df_with_labels(labels, theta):
    df = pd.DataFrame(theta)
    df.index = labels
    df.columns = labels

    return df


def normalize_dfs(true_df, p, labels):
    """
    Assigns lowest label value to lowest number of entries in each cluster.
    :param true_df:
    :param p:
    :param labels:
    :return:
    """
    lowtg = true_df.values[np.tril_indices(p, -1)].tolist()
    dict_counts = dict(Counter(lowtg))
    sorted_x = dict(sorted(dict_counts.items(), key=operator.itemgetter(1)))
    lowtg_index = dict()
    for idx, val in enumerate(sorted_x.keys()):
        lowtg_index[idx] = [i for i, e in enumerate(lowtg) if e == val]

    for i, val in lowtg_index.items():
        for j in val:
            lowtg[j] = i

    mat_clusters = np.zeros((p, p))
    mat_clusters[np.tril_indices(p, -1)] = lowtg
    df = pd.DataFrame(mat_clusters)
    df.index = labels
    df.columns = labels
    return df


def test_gene_expr_dataset():
    np.random.seed(680)
    X, S, y, gene_decr_map, labels = load_gene_subset()

    S = cov_nearest(S, method="clipped", threshold=0.1)

    s_df = pd.DataFrame(S)
    s_df.columns = list(labels.values())
    s_df.index = list(labels.values())

    # plot_with_annotations(s_df, 'Sample covariance')

    # S, y, labels = load_gene_data_reduced()
    p = 50
    # S = S[:p, :p]

    # lam1 = 0.006  # controls sparsity
    # lam2 = 0.00009  # encourages equality of coefficients
    # rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)

    # theta_0 = make_sparse_spd_matrix(p, alpha=0.5, norm_diag=False, smallest_coef=-1.0, largest_coef=1.0)
    theta_0 = np.linalg.inv(S)

    # GOWL estimator
    # theta_gowl = proximal_descent(theta_0,
    #                               theta_0,
    #                               S,
    #                               rho,
    #                               gradf=grad_log_det_loss,
    #                               prox=prox_graph_owl,
    #                               f_func=log_det_loss,
    #                               g_func=gowl_penalty,
    #                               max_iter=10000,
    #                               use_dual_gap=True,
    #                               eps=0.001,
    #                               step_size_type='backtracking',
    #                               verbose=True)

    # ccGOWL estimator
    theta_owl = run_ccgowl(X.values, 0.3, 0.00612821)
    np.fill_diagonal(theta_owl, 1.0)

    lmbda = .6
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

    grab_df = pd.DataFrame(theta_grab)
    grab_df.index = list(labels.values())
    grab_df.columns = list(labels.values())

    gl = GraphicalLasso()
    gl.fit(S)
    theta_glasso = gl.get_precision()

    owl_df = pd.DataFrame(theta_owl)
    owl_df.index = list(labels.values())
    owl_df.columns = list(labels.values())

    glasso_df = pd.DataFrame(theta_glasso)
    glasso_df.index = list(labels.values())
    glasso_df.columns = list(labels.values())

    y_true_clusters_df = compute_true_group(list(labels.values()), theta_owl)
    K = len(np.unique(y_true_clusters_df.values[np.tril_indices(p, -1)].tolist()))
    y_true_pair_per_cluster = pairs_in_clusters(y_true_clusters_df, K + 1, p)
    # y_true_pair_per_cluster.to_csv('true_gene_groups.csv')

    owl_clusters = spectral_clustering(theta=theta_owl, K=K)
    owl_clusters = [int(cluster) for cluster in owl_clusters]
    owl_mat_clusters = np.zeros((p, p))
    owl_mat_clusters[np.tril_indices(p, -1)] = owl_clusters
    owl_clusters_df = pd.DataFrame(owl_mat_clusters)
    owl_clusters_df.index = list(labels.values())
    owl_clusters_df.columns = list(labels.values())

    glasso_clusters = spectral_clustering(theta=theta_glasso, K=K)
    glasso_clusters = [int(cluster) for cluster in glasso_clusters]
    glasso_mat_clusters = np.zeros((p, p))
    glasso_mat_clusters[np.tril_indices(p, -1)] = glasso_clusters
    glasso_clusters_df = pd.DataFrame(glasso_mat_clusters)
    glasso_clusters_df.index = list(labels.values())
    glasso_clusters_df.columns = list(labels.values())

    grab_clusters = spectral_clustering(theta=theta_grab, K=K)
    grab_clusters = [int(cluster) for cluster in grab_clusters]
    grab_mat_clusters = np.zeros((p, p))
    grab_mat_clusters[np.tril_indices(p, -1)] = grab_clusters
    grab_clusters_df = pd.DataFrame(grab_mat_clusters)
    grab_clusters_df.index = list(labels.values())
    grab_clusters_df.columns = list(labels.values())

    owl_clusters_df = normalize_dfs(owl_clusters_df, p, list(labels.values()))
    glasso_clusters_df = normalize_dfs(glasso_clusters_df, p, list(labels.values()))
    grab_clusters_df = normalize_dfs(grab_clusters_df, p, list(labels.values()))

    # plot_with_annotations(glasso_clusters_df, 'GLASSO Groups')
    # plot_with_annotations(owl_clusters_df, 'GOWL Groups')
    # plot_with_annotations(grab_clusters_df, 'GRAB Groups')
    #
    # plot_theta_matrix_2d(glasso_df, 'GLASSO')
    # plot_theta_matrix_2d(owl_df, 'GOWL')
    # plot_theta_matrix_2d(grab_df, 'GRAB')

    # plot_multiple_theta_matrices_2d([theta_blocks, theta_star, theta_grab, theta_owl],
    #                                 [f"Blocks: {len(blocks)}", 'True Theta', 'GRAB', 'GOWL'])

    # owl_pair_per_cluster = pairs_in_clusters(owl_clusters_df,
    #                                          K,
    #                                          p)
    # owl_pair_per_cluster.to_csv('ccgowl_golub_et_al_groups.csv')

    # glasso_pair_per_cluster = pairs_in_clusters(glasso_clusters_df,
    #                                             K,
    #                                             p)
    # glasso_pair_per_cluster.to_csv('glasso_golub_et_al_groups.csv')

    grab_pair_per_cluster = pairs_in_clusters(grab_clusters_df,
                                                K,
                                                p)
    grab_pair_per_cluster.to_csv('grab_golub_et_al_groups_May21.csv')


def pairs_in_clusters(clusters_df, K, p):
    cluster_values = np.arange(K - 1, dtype=float)
    rows = list()
    for val in cluster_values:
        pairs = np.where(clusters_df == val)
        for i, j in zip(pairs[0], pairs[1]):
            if j >= i:
                continue

            # if (clusters_df.index[i].split('/')[1] == 'Uncategorized'
            #     or
            #     clusters_df.index[j].split('/')[1] == 'Uncategorized'):
            #     continue

            rows.append([
                val,
                clusters_df.index[i],
                clusters_df.columns[j],
            ])

    df = pd.DataFrame(rows)
    df.columns = ['Group', 'I', 'J']
    return df


def compute_true_group(labels, theta):
    df = convert_to_df_with_labels(labels, theta.copy())
    gics = [label.split('/')[1] for label in np.unique(df.columns)]
    le = preprocessing.LabelEncoder()
    le.fit(gics)
    gic_to_label = dict(zip(gics, le.transform(gics)))
    gic_to_label['no_cluster'] = len(gic_to_label)

    i = 0
    for index, row in df.iterrows():
        j = 0
        for k in row.keys():
            if j >= i:
                continue

            if k.split('/')[1] == index.split('/')[1]:
                row[k] = gic_to_label[k.split('/')[1]]
            else:
                row[k] = np.nan

            j += 1

        i += 1

    df = df.fillna(gic_to_label['no_cluster'])
    return df


if __name__ == '__main__':
    test_gene_expr_dataset()
