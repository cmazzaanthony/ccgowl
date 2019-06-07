import operator
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.covariance import GraphicalLasso
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.metrics import f1_score
import grab.GRAB as grab
from statsmodels.stats.correlation_tools import cov_nearest

from gowl.algorithms.proximal_descent import proximal_descent
from gowl.data.real_data import load_huge_stock_data
from gowl.evaluation.cluster_metrics import spectral_clustering
from gowl.loss.loss_functions import grad_log_det_loss, log_det_loss
from gowl.prox.prox_owl import prox_graph_owl, oscar_weights, gowl_penalty

from gowl.simulations.synthetic_results_grid_search import run_ccgowl
from gowl.visualization.graph_visualization import plot_theta_matrix_2d


def convert_to_df_with_labels(labels, theta):
    # labels = [f"{label.split('/')[0][:4]}/{label.split('/')[1][:2]}" for label in labels]
    df = pd.DataFrame(theta)
    df.index = labels
    df.columns = labels

    return df


def plot_with_annotations(df, title):
    mask = np.zeros_like(df.values, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(title)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df, cmap=cmap, mask=mask, ax=ax, vmax=df.values.max(), center=0)
    plt.show()


def test_stocks_returns_dataset():
    np.random.seed(680)
    data, S, labels = load_huge_stock_data()
    p = S.shape[0]

    s_df = pd.DataFrame(S)
    s_df.columns = labels
    s_df.index = labels

    lam1 = 0.000001  # controls sparsity
    lam2 = 0.000001  # encourages equality of coefficients
    rho = oscar_weights(lam1, lam2, (p ** 2 - p) / 2)

    theta_0 = np.linalg.inv(S)
    # theta_0 = cov_nearest(theta_0, method="clipped", threshold=0.1)
    # theta_0 = make_sparse_spd_matrix(p, alpha=0.5, norm_diag=False, smallest_coef=-1.0, largest_coef=1.0)

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
    theta_owl = run_ccgowl(data.values, 0.4, 0.0001)

    np.fill_diagonal(theta_owl, 1.0)
    owl_df = pd.DataFrame(theta_owl)
    owl_df.index = labels
    owl_df.columns = labels

    gl = GraphicalLasso()
    gl.fit(S)
    theta_glasso = gl.get_precision()
    glasso_df = pd.DataFrame(theta_glasso)
    glasso_df.index = labels
    glasso_df.columns = labels

    # lmbda = .4
    # K = 10
    # o_size = .3  # The size of overlap, as an input parameter
    # max_iter = 20
    # tol = 1e-4
    # dual_max_iter = 600
    # dual_tol = 1e-4
    # theta_grab, blocks = grab.BCD(S,
    #                               lmbda=lmbda,
    #                               K=K,
    #                               o_size=o_size,
    #                               max_iter=max_iter,
    #                               tol=tol,
    #                               dual_max_iter=dual_max_iter,
    #                               dual_tol=dual_tol)

    theta_grab = np.zeros((p, p))

    theta_grab = np.asarray(theta_grab)

    y_true_clusters_df = compute_true_group(labels, theta_owl)
    K = len(np.unique(y_true_clusters_df.values[np.tril_indices(p, -1)].tolist()))

    owl_clusters = spectral_clustering(theta=theta_owl, K=K)
    owl_clusters = [int(cluster) for cluster in owl_clusters]
    owl_mat_clusters = np.zeros((p, p))
    owl_mat_clusters[np.tril_indices(p, -1)] = owl_clusters
    owl_clusters_df = convert_to_df_with_labels(labels, owl_mat_clusters.copy())

    grab_clusters = spectral_clustering(theta=theta_grab, K=K)
    grab_clusters = [int(cluster) for cluster in grab_clusters]
    grab_mat_clusters = np.zeros((p, p))
    grab_mat_clusters[np.tril_indices(p, -1)] = grab_clusters
    grab_clusters_df = convert_to_df_with_labels(labels, grab_mat_clusters.copy())

    glasso_clusters = spectral_clustering(theta=theta_glasso, K=K)
    glasso_clusters = [int(cluster) for cluster in glasso_clusters]
    glasso_mat_clusters = np.zeros((p, p))
    glasso_mat_clusters[np.tril_indices(p, -1)] = glasso_clusters
    glasso_clusters_df = convert_to_df_with_labels(labels, glasso_mat_clusters.copy())

    y_true_clusters_df = normalize_dfs(y_true_clusters_df, p, labels)
    owl_clusters_df = normalize_dfs(owl_clusters_df, p, labels)
    glasso_clusters_df = normalize_dfs(glasso_clusters_df, p, labels)
    grab_clusters_df = normalize_dfs(grab_clusters_df, p, labels)

    # plot_with_annotations(glasso_clusters_df, 'GLASSO')
    # plot_with_annotations(owl_clusters_df, 'GOWL')
    # plot_with_annotations(y_true_clusters_df, 'THETA STAR')

    # print(f"OWL F1: {compute_f1(y_true_clusters_df, owl_clusters_df, p)}")
    # print(f"GLASSO F1: {compute_f1(y_true_clusters_df, glasso_clusters_df, p)}")

    # plot_theta_matrix_2d(glasso_df, 'GLASSO')
    # plot_theta_matrix_2d(owl_df, 'GOWL')

    # y_true_pair_per_cluster = pairs_in_clusters(y_true_clusters_df, K, p)
    # y_true_pair_per_cluster.to_csv('true_stock_groups.csv')

    owl_pair_per_cluster = pairs_in_clusters(owl_clusters_df, K, p)
    owl_pair_per_cluster.to_csv('owl_stock_groups_May21.csv')

    # glasso_pair_per_cluster = pairs_in_clusters(glasso_clusters_df, K, p)
    # glasso_pair_per_cluster.to_csv('glasso_stock_groups.csv')
    #
    # #
    # owl_pair_per_cluster = pairs_in_clusters(grab_clusters_df, K, p)
    # owl_pair_per_cluster.to_csv('grab_stock_groups_May21.csv')


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


def pairs_in_clusters(clusters_df, K, p):
    cluster_values = np.arange(K, dtype=float)
    rows = list()
    for val in cluster_values:
        pairs = np.where(clusters_df == val)
        for i, j in zip(pairs[0], pairs[1]):
            if j >= i:
                continue

            rows.append([
                val,
                clusters_df.index[i],
                clusters_df.columns[j],
            ])

    df = pd.DataFrame(rows)
    df.columns = ['Group', 'I', 'J']
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
    return convert_to_df_with_labels(labels, mat_clusters.copy())


def compute_f1(y_true_df, y_hat_df, p):
    y_true = y_true_df.values[np.tril_indices(p, -1)].tolist()
    y_hat = y_hat_df.values[np.tril_indices(p, -1)].tolist()

    y_true.sort()
    y_hat.sort()

    return f1_score(y_true, y_hat, average='macro')


if __name__ == '__main__':
    test_stocks_returns_dataset()
