import operator
from collections import Counter

import numpy as np
from sklearn import mixture


def spectral_clustering(theta, K, method='GMM'):
    p = theta.shape[0]
    v = theta[np.tril_indices(p, -1)].reshape(-1, 1)

    if method == 'GMM':
        model = mixture.GaussianMixture(n_components=K, covariance_type='full')
    else:
        raise NotImplementedError(
            "Only GMM clustering method is implemented")

    model.fit(v)
    prob = model.predict_proba(v)
    clusters = (prob > 0.1) * 1
    cluster_indices = {k: np.where(clusters[:, k] == 1)[0] for k in range(K)}
    for i, indices in enumerate(list(cluster_indices.values())):
        v[indices] = i
    return normalize(v.flatten())


def normalize(y_grouped):
    """
    Normalizes the cluster to ensure that the smallest group has label 0 and
    the second smallest has label 1 etc.
    """
    dict_counts = dict(Counter(y_grouped))
    sorted_x = dict(sorted(dict_counts.items(), key=operator.itemgetter(1)))
    lowtg_index = dict()
    for idx, val in enumerate(sorted_x.keys()):
        lowtg_index[idx] = [i for i, e in enumerate(y_grouped) if e == val]

    for i, val in lowtg_index.items():
        for j in val:
            y_grouped[j] = i

    return y_grouped
