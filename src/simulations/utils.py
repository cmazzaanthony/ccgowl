import operator
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import preprocessing


def convert_to_df_with_labels(labels, theta):
    df = pd.DataFrame(theta)
    df.index = labels
    df.columns = labels

    return df


def pairs_in_clusters(clusters_df, K):
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


def compute_true_group(theta, labels):
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


def normalize_dfs(true_df, labels, p):
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
