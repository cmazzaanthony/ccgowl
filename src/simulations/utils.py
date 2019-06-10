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
