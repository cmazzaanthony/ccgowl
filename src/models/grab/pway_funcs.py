import numpy as np
from sklearn import cluster
from grab import GRAB


def my_spectral_clustering(S, K):
    S2 = np.abs(S)

    spectral = cluster.SpectralClustering(n_clusters=K,
                                          eigen_solver='arpack',
                                          affinity="precomputed", assign_labels='discretize', eigen_tol=.001,
                                          random_state=48)

    labels = spectral.fit_predict(S2)

    clusters = get_clusters_from_labels(labels, K)
    return clusters


def standardize(X):
    ret = (X - X.mean(axis=0)) / np.std(X, axis=0, ddof=1)
    return ret;


def get_likelihood(X, T):
    X = np.array(X)
    logdet = np.linalg.slogdet(T)[1]
    print(logdet)
    trace = np.trace(np.matrix(X) * np.matrix(T))
    print(trace)

    return logdet - trace
    # X = np.array(X)
    # return np.linalg.slogdet(T)[1] - np.sum(X*T)


def get_obj(S, T, W, lmbda):
    T = np.matrix(T)
    S = np.array(S)
    logdet = np.linalg.slogdet(T)[1]
    print(logdet)
    trace = np.trace(np.matrix(S) * T)
    print(trace)
    return logdet - trace - lmbda * np.sum(np.abs(T)) + lmbda * np.trace(W * np.abs(T))


def norm(T, p):
    T = np.matrix(T)
    if (p == 1):
        return np.sum(np.abs(T))
    elif (p == 2):
        return np.linalg.norm(T, 'fro')
    else:
        return -1


def find_highest_variance_genes(data, num_var, add_important_genes=0, gene_names=None):
    P = data.shape[0];
    stds = np.zeros(P)

    for i in range(P):
        var = np.var(data[i, :])
        stds[i] = var

    best_genes = np.argsort(-stds)
    best_genes = best_genes[0:num_var]
    if (add_important_genes == 1):

        imp_genes_names = ["FLT3", "NPM1", "CEBPA", "KIT", "NRAS", "MLL", "WT1", "IDH1", "IDH2", "TET2", "DNMT3A",
                           "ASXL1"]
        imp_genes = []
        for gene in imp_genes_names:
            imp_genes.append(gene_names.index(gene))
        new_imp_genes = []
        best_genes_set = set(best_genes)
        for gene in imp_genes:
            if not gene in best_genes_set:
                new_imp_genes.append(gene)
        num_important_genes = len(new_imp_genes)
        print("num imp genes: ", num_important_genes)
        best_genes = best_genes[0:num_var - num_important_genes]
        best_genes = list(best_genes)
        for gene in new_imp_genes:
            best_genes.append(gene)
        best_genes = np.array(best_genes)
        best_genes = best_genes[np.random.permutation(num_var)]
        best_genes = list(best_genes)

    return best_genes


# def compute_functional_enrichment(my_pwyas, cluster_variables, num_vars_AML, K):
#     if (cluster_variables == 1):
#         idxes_gene_names = scipy.io.loadmat("data/genes_C_R" + str(num_vars_AML) + ".mat")['genes']
#     else:
#         idxes_gene_names = scipy.io.loadmat("data/genes_" + str(num_vars_AML) + ".mat")['genes']
#
#     idxes_gene_names = [s.encode('ascii', 'ignore').strip() for s in idxes_gene_names]
#     idxes_gene_names = np.array(idxes_gene_names)
#
#     print "selected genes are: ", idxes_gene_names
#     for (i, g) in enumerate(idxes_gene_names):
#         print i, ": ", idxes_gene_names[i]
#     print idxes_gene_names[0]
#
#     AML1 = pd.read_csv("AML1_cancer.csv", index_col=0)
#
#     gene_names = list(AML1.index)
#
#     gene_sets = load_pathways()
#
#     # if (num_pathways!=-1):
#     # pathways = pathways[0:num_pathways]
#     # print pathways
#     all_genes = []
#     map(all_genes.extend, gene_sets)
#     print "mapping done"
#     all_genes = list(np.sort(list(set(all_genes))))
#     print (all_genes)
#     print (len(all_genes))
#
#     union_genes = list(set(all_genes).union(set(gene_names)))
#
#     #Testing what happens if we limit our scope to 500 genes
#     #union_genes = set([])
#     #for k in range(len(my_pwyas)):
#     #    union_genes = union_genes.union(set(idxes_gene_names[my_pwyas[k]]))
#     #union_genes = list(union_genes)
#     ##############
#
#     N = len(union_genes)
#
#     print "union size: ", N
#     num_match = 0
#
#     # gene_sets = gene_sets[0:100]
#     num_hypothesis = K * len(gene_sets)
#
#     all_ps = list()
#     ii = 0
#     for gene_set in gene_sets:
#         ii = ii + 1
#         best_p = 1e9
#         for pway_idx in my_pwyas:
#             pway =[]
#             if len(pway_idx)>0:
#                 pway = idxes_gene_names[pway_idx]
#             S1 = set(gene_set)
#             #Testing what happens if we limit our scope to 500 genes
#             #S1 = S1.intersection(union_genes)
#             #gene_set=list(S1)
#             #########
#
#
#             S2 = set(pway)
#             # print "S1: ", gene_set
#             # print "S2: ", pway
#             intersect_size = len(S1.intersection(S2))
#             only_pway = len(pway) - intersect_size
#             only_gs = len(gene_set) - intersect_size
#
#             none = N - len(S1.union(S2))
#
#             print "aaa", intersect_size, only_pway, only_gs, none
#
#             oddsratio, pvalue = stats.fisher_exact([[intersect_size, only_gs], [only_pway, none]])
#             all_ps.append(pvalue)
#             # if (pvalue<1e-10):
#             #print "matched ", pvalue, " ", k, " ", ii
#             if (pvalue < .05 / num_hypothesis):
#                 # print "pvalue: ", pvalue
#                 num_match = num_match + 1
#
#             if (pvalue < best_p):
#                 best_p = pvalue
#                 # if (best_p<.05/num_hypothesis):
#                 # print "best p: ",best_p
#                 # num_match = num_match + 1
#     print "bonferroni number of matched genes:", num_match
#     all_ps = np.array(all_ps)
#     all_ps = np.sort(all_ps)
#     all_ps = quic.adjust_pvals_fdr(all_ps)
#
#     for p in all_ps:
#         if (p < .05):
#             print "pvalue: ", p
#
#     pos = sum(1 for p in all_ps if p < .05)
#     print "fdr number of matched genes:", pos
#
#     return pos


# def find_hubs(my_pwyas, cluster_variables, num_vars_AML):
#     if (cluster_variables == 1):
#         idxes_gene_names = scipy.io.loadmat("data/genes_C_R" + str(num_vars_AML) + ".mat")['genes']
#     else:
#         idxes_gene_names = scipy.io.loadmat("data/genes_" + str(num_vars_AML) + ".mat")['genes']
#
#     gene_names = np.array(idxes_gene_names)
#
#     num_pways = []
#
#     pway_sets = [set(pway) for pway in my_pwyas]
#     for i in range(num_vars_AML):
#         num_pway = 0
#         for pway in pway_sets:
#             if i in pway:
#                 num_pway = num_pway + 1
#         if num_pway > 0:
#             print "hub ", num_pway, " ", gene_names[i]
#         num_pways.append(num_pway)
#     return num_pways

def get_clusters_from_labels(labels, K=-1):
    P = len(labels)
    if (K == -1):
        K = np.int(np.max(labels)) + 1
    clusters = [[] for k in range(K)]
    for i in range(P):
        k = np.int(labels[i])
        clusters[k].append(i)
    return clusters


# Scores from U, then...
def get_pathways_from_U_kmeans(U, Max, capacity_increase, K, P, o_size):
    scores = make_Scores_from_U(U)
    myMax = np.int(Max * capacity_increase)
    Maxes = np.ones((K)) * myMax
    C = (1 + o_size) * P
    # pathways_hat = pw_learn.assign_pways2(scores, C, P, K, Maxes)
    # pathways_hat = pw_learn.assign_pways3(scores, C, P, K, Maxes)
    pathways_hat = GRAB.assign_pways2(U, scores, C, P, K, Maxes)
    return pathways_hat


def make_Scores_from_U(U):
    # U = np.abs(U)
    scores = list()
    K = U.shape[0]
    P = U.shape[1]
    for k in range(K):
        l = list()
        for i in range(P):
            l.append(U[k, i])
        scores.append(l)
    return scores
