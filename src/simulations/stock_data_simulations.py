import operator
import pathlib
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from fa2 import ForceAtlas2
from matplotlib.collections import LineCollection
from sklearn import preprocessing

from src.data.make_synthetic_data import standardize
from src.evaluation.cluster_metrics import spectral_clustering
from src.models.ccgowl import CCGOWLModel
from src.simulations.simulation import Simulation
from src.visualization.curved_edges import curved_edges


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


def read_stock_data():
    proj_root_path = pathlib.Path.cwd().parent.parent
    data_path = 'data/raw/s_&_p/data.csv'
    sector_path = 'data/raw/s_&_p/info.csv'
    data_full_path = proj_root_path / data_path
    sectors_full_path = proj_root_path / sector_path
    data = pd.read_csv(data_full_path, index_col=0)
    info = pd.read_csv(sectors_full_path)
    data = standardize(data)
    stock_names = info['V1']
    sectors = info['V2']
    data.columns = [f'{comp}/{gic}' for comp, gic in zip(stock_names, sectors)]

    return data, data.columns.to_list()


class StockData(Simulation):

    def __init__(self, model, model_params):
        self.data, self.info = read_stock_data()
        _, self.p = self.data.shape
        self.sample_cov = np.cov(self.data.T)
        self.model = model(self.data.values, *model_params)
        self.true_groups = pd.DataFrame()
        self.predicted_groups = pd.DataFrame()

    def run(self):
        np.random.seed(680)

        self.model.fit()
        theta_hat = self.model.theta_hat

        y_true_clusters_df = self._compute_true_group(theta_hat)
        K = len(np.unique(y_true_clusters_df.values[np.tril_indices(self.p, -1)].tolist()))

        theta_clusters = spectral_clustering(theta=theta_hat, K=K)
        theta_clusters = [int(cluster) for cluster in theta_clusters]
        theta_mat_clusters = np.zeros((self.p, self.p))
        theta_mat_clusters[np.tril_indices(self.p, -1)] = theta_clusters
        clusters_df = convert_to_df_with_labels(self.info, theta_mat_clusters.copy())

        y_true_clusters_df = self._normalize_dfs(y_true_clusters_df)
        clusters_df = self._normalize_dfs(clusters_df)

        self.true_groups = pairs_in_clusters(y_true_clusters_df, K)
        self.predicted_groups = pairs_in_clusters(clusters_df, K)

    def plot_results(self):
        df_true = self.true_groups
        df_pred = self.predicted_groups

        carac_dict = dict(zip(['Consumer Discretionary', 'Consumer Staples', 'Energy',
                               'Financials', 'Health Care', 'Industrials',
                               'Information Technology', 'Materials',
                               'Telecommunications Services', 'Utilities', 'No Group'],
                              ['red', 'blue', 'gray', 'yellow', 'black',
                               'skyblue', 'orange', 'yellowgreen', 'pink', 'cyan', 'purple']))

        df_filtered = df_pred.loc[(df_pred['I'] != df_pred['J']) & (df_pred['Group'] != 10)]
        df_filtered.drop_duplicates(['I', 'J'], inplace=True)

        ####### Colors
        df_true = df_pred.loc[(df_true['I'] != df_true['J']) & (df_true['Group'] != 10)]
        df_true.drop_duplicates(['I', 'J'], inplace=True)

        ######## Graph using estimates
        G = nx.from_pandas_edgelist(df_filtered, 'I', 'J', create_using=nx.Graph())

        unique_sectors_in_cluster = list(np.unique(list(df_true['I']) + list(df_true['J'])))
        carac = pd.DataFrame({
            'ID': unique_sectors_in_cluster,
            'myvalue': [carac_dict[entry.split('/')[1]] for entry in unique_sectors_in_cluster],
        })

        carac = carac.set_index('ID')
        carac = carac.reindex(G.nodes())
        j = 0
        for i, row in carac.iterrows():
            if pd.isna(row['myvalue']):
                carac.iloc[j, 0] = carac_dict[i.split('/')[1]]
            j += 1

        forceatlas2 = ForceAtlas2(  # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=1.0,

            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,

            # Log
            verbose=True)
        pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=50)
        curves = curved_edges(G, pos)
        lc = LineCollection(curves, color=(0.0, 0.0, 0.0), alpha=0.1, linewidths=5.0, linestyles='solid')

        plt.figure(figsize=(20, 20))
        plt.gca().add_collection(lc)
        nx.draw_networkx_nodes(G, pos, node_size=800, alpha=0.9, node_color=list(carac['myvalue']))
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.show()

    def _compute_true_group(self, theta):
        df = convert_to_df_with_labels(self.info, theta.copy())
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

    def _normalize_dfs(self, true_df):
        """
        Assigns lowest label value to lowest number of entries in each cluster.
        :param true_df:
        :param p:
        :param labels:
        :return:
        """
        lowtg = true_df.values[np.tril_indices(self.p, -1)].tolist()
        dict_counts = dict(Counter(lowtg))
        sorted_x = dict(sorted(dict_counts.items(), key=operator.itemgetter(1)))
        lowtg_index = dict()
        for idx, val in enumerate(sorted_x.keys()):
            lowtg_index[idx] = [i for i, e in enumerate(lowtg) if e == val]

        for i, val in lowtg_index.items():
            for j in val:
                lowtg[j] = i

        mat_clusters = np.zeros((self.p, self.p))
        mat_clusters[np.tril_indices(self.p, -1)] = lowtg
        return convert_to_df_with_labels(self.info, mat_clusters.copy())


if __name__ == '__main__':
    stock_sim = StockData(CCGOWLModel, [0.4, 0.0001])
    stock_sim.run()
    stock_sim.plot_results()
