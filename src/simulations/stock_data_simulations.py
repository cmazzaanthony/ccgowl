import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from fa2 import ForceAtlas2
from matplotlib.collections import LineCollection

from src.data.make_synthetic_data import standardize
from src.evaluation.cluster_metrics import spectral_clustering
from src.models.ccgowl import CCGOWLModel
from src.simulations.simulation import Simulation
from src.simulations.utils import convert_to_df_with_labels, pairs_in_clusters, compute_true_group, normalize_dfs
from src.visualization.curved_edges import curved_edges


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

        y_true_clusters_df = compute_true_group(theta_hat, self.info)
        K = len(np.unique(y_true_clusters_df.values[np.tril_indices(self.p, -1)].tolist()))

        theta_clusters = spectral_clustering(theta=theta_hat, K=K)
        theta_clusters = [int(cluster) for cluster in theta_clusters]
        theta_mat_clusters = np.zeros((self.p, self.p))
        theta_mat_clusters[np.tril_indices(self.p, -1)] = theta_clusters
        clusters_df = convert_to_df_with_labels(self.info, theta_mat_clusters.copy())

        y_true_clusters_df = normalize_dfs(y_true_clusters_df, self.info, self.p)
        clusters_df = normalize_dfs(clusters_df, self.info, self.p)

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

        df_true = df_pred.loc[(df_true['I'] != df_true['J']) & (df_true['Group'] != 10)]
        df_true.drop_duplicates(['I', 'J'], inplace=True)

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


if __name__ == '__main__':
    stock_sim = StockData(CCGOWLModel, [0.4, 0.0001])
    stock_sim.run()
    stock_sim.plot_results()
