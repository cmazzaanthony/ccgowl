import numpy as np
import heapq
import copy
import grab.pway_funcs as fn2
from grab.quic import QUIC_lmat
from sklearn.cluster import KMeans


matsession = None


class Edge:  # Used for Max heap: The heap package is originally min heap!
    def __init__(self, i, k, score):
        self.i = i
        self.k = k
        self.score = score

    def __lt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.score == other.score


inf = 100000


def pway_post_process(pathways, P):
    for k in range(len(pathways)):
        pathways[k] = list(set(pathways[k]))
        pathways[k] = np.sort(pathways[k])

    p_new = list()
    for i in range(len(pathways)):
        p1 = pathways[i]
        shouldAdd = 1
        for j in range(len(pathways)):
            if (i == j):
                continue
            p2 = pathways[j]
            if (set(p1).issubset(set(p2))):
                if (i > j or len(p1) != len(p2)):
                    shouldAdd = 0
                    break
        if (shouldAdd == 1):
            p_new.append(p1)

    K = len(p_new)

    Z_new = np.zeros(shape=(P, K))

    for k in range(len(p_new)):
        p = set(p_new[k])
        for j in p:
            Z_new[j, k] = 1

    return (p_new, Z_new)


def init_pathways(S, P, C, K, Max, clusters=None):
    if clusters == None:
        clusters = fn2.my_spectral_clustering(S, K)

    # print "res of spec: ", clusters

    scores = list()
    Maxes = list()
    pways = list()
    must_add = []
    for k in range(K):
        scores.append([])
        if len(clusters[k]) > Max:
            npop = len(clusters[k]) - Max
            for i in range(npop):
                j = clusters[k].pop()
                must_add.append(j)
        C = C - len(clusters[k])
        Maxes.append(Max - len(clusters[k]))
        for i in range(P):
            if i in clusters[k]:
                scores[k].append(-1)
                continue
            s = 0
            for j in clusters[k]:
                s = s + np.abs(S[i, j])

            if (len(clusters[k]) != 0):
                s = s / len(clusters[k])
            else:
                s = -inf
            scores[k].append(s)

        pways.append([])
        for j in clusters[k]:
            pways[k].append(j)

    scoresA = np.zeros(shape=(P, K))
    for k in range(K):
        for i in range(P):
            scoresA[i, k] = scores[k][i]

    p_new = assign_pways(scoresA, C, P, K, Maxes, must_add, call_from_init=True)
    z_new = get_z_from_p(p_new, P, K)

    sum = 0
    for k in range(len(z_new)):
        sum = sum + len(z_new[k])
    add_Z_to_pway(z_new, pways, 0, P)

    (p, z) = pway_post_process(pways, P)

    return (p, z)


def kmeansScores(scores, P, K):
    X = np.ndarray(shape=(P, K))
    for i in range(P):
        for k in range(K):
            X[i, k] = scores[k][i]
    labels = KMeans(n_clusters=K).fit_predict(X)
    return labels


def assign_pways(scores, C, P, K, Maxes, must_add=None, call_from_init=False):
    scores = copy.copy(scores)
    scores = scores.T

    p_new = list();
    for k in range(K):
        tmp = []
        p_new.append(tmp)

    assigned_count = [0] * P
    assigned = 0;

    all_allowed = np.int(C)

    var_list = range(P)
    if (must_add != None):
        second_priority = set(var_list).difference(must_add)
        var_list = list(np.sort(must_add))
        var_list.extend(np.sort(list(second_priority)))
    # 1) Max assignment for each var
    if call_from_init:
        var_list = must_add  # If call_from_init, we only add the ones that don't have any pathways

    for i in var_list:
        if ((all_allowed - assigned) <= 0):
            break
        scp = list()
        for k in range(K):
            scp.append(scores[k][i])
        tmp = heapq.nlargest(K, range(len(scp)), scp.__getitem__)

        best_k = tmp[0]
        idx = 0
        while (len(p_new[best_k]) >= Maxes[best_k]):
            idx = idx + 1
            if (idx == K):
                best_k = -1
                break
            best_k = tmp[idx]

        if (best_k == -1 or scores[best_k][i] < 0):
            continue
        assigned = assigned + 1
        scores[best_k][i] = -inf
        assigned_count[i] = assigned_count[i] + 1
        p_new[best_k].append(i)

    scorelist = list()
    for i in range(P):
        for k in range(K):
            e = Edge(i, k, scores[k][i])
            heapq.heappush(scorelist, e)

    all_allowed = all_allowed - assigned
    idx = 0
    count = 0

    num_inf = 0
    all_len = P * K
    while count < all_allowed and idx < all_len:
        edge = heapq.heappop(scorelist)
        idx = idx + 1

        k = edge.k
        var_idx = edge.i
        if edge.score == -inf:
            num_inf = num_inf + 1

        if (len(p_new[k]) >= Maxes[k] or edge.score == -inf):
            continue

        assigned_count[var_idx] = assigned_count[var_idx] + 1
        p_new[k].append(var_idx)

        count = count + 1

    for k, p in enumerate(p_new):
        p_new[k] = np.sort(list(set(p)))

    return p_new


# This is very similar to assign_pways2, but just first does kmeans on U.
# For now, assume Max=P
def assign_pways2(U, scores, C, P, K, Maxes, must_add=None):
    p_new = list();
    for k in range(K):
        tmp = []
        p_new.append(tmp)

    assigned_count = [0] * P
    assigned = 0;

    k = 0
    all_allowed = np.int(C)

    var_list = range(P)
    if (must_add != None):
        second_priority = set(var_list).difference(must_add)
        var_list = list(np.sort(must_add))
        var_list.extend(np.sort(list(second_priority)))

    # 1) Max assignment for each var
    # What happens at the end of this?: each var is added to exactly one cluster

    labels = kmeansScores(scores, P, K)

    for i in var_list:
        best_k = labels[i]
        assigned = assigned + 1
        scores[best_k][i] = -inf
        assigned_count[i] = assigned_count[i] + 1
        p_new[best_k].append(i)

    # recompute score based on the clusters...
    for i in range(P):
        for k in range(K):
            if scores[k][i] == -inf:
                continue
            s = 0
            for j in range(P):
                if i != j and labels[j] == k:
                    s += np.dot(U[:, i].T, U[:, j])
            scores[k][i] = s
            if s != 0:
                scores[k][i] /= len(p_new[k])

    # 3) assign from max weight, just don't assign more than capacity

    scorelist = list()
    for i in range(P):
        for k in range(K):
            # print (scores[k][i], " ")
            e = Edge(i, k, scores[k][i])
            heapq.heappush(scorelist, e)

    # heapq.heapify(scorelist)

    all_allowed = all_allowed - assigned
    idx = 0
    count = 0

    num_inf = 0
    all_len = P * K
    while count < all_allowed and idx < all_len:
        edge = heapq.heappop(scorelist)
        # print edge.score, edge.k, edge.i
        idx = idx + 1

        k = edge.k
        var_idx = edge.i
        if edge.score == -inf:
            num_inf = num_inf + 1

        # print ("len is ",len(p_new[k]),M)
        if (len(p_new[k]) >= Maxes[k] or edge.score == -inf):
            continue
        # print ("len is ",len(p_new[k]),M)

        assigned_count[var_idx] = assigned_count[var_idx] + 1
        # print "adding ", var_idx, " to ", k
        p_new[k].append(var_idx)

        count = count + 1

    for k, p in enumerate(p_new):
        p_new[k] = np.sort(list(set(p)))
        # print "here: ", p_new[k]

        # for i in range(P):
        # if (assigned_count[i] == 0):
        # print (i, " is ", 0)

        # print "len is assign_pways2:"
        # for k in range(K):
        # print len(p_new[k])

    return p_new


def get_z_from_p(p_new, P, K):
    z_new = list()
    for i in range(P):
        z_new.append([])
    for k in range(K):
        for t in range(len(p_new[k])):
            idx = p_new[k][t]
            z_new[idx].append(k)
    return z_new


def make_lmbda_matrix(Z, P, lmbda, W=None):
    if W is not None:
        ZZT = W
    else:
        ZZT = np.matrix(Z) * np.matrix(Z.T);
    ret = np.ones(shape=(P, P)) * lmbda;
    ret -= lmbda * ZZT
    ret = np.maximum(ret, 0)
    return ret;


def BCD(S, lmbda=.15, K=5, o_size=.25, max_iter=20, tol=1e-4, dual_max_iter=600, dual_tol=1e-4):
    P = S.shape[1];
    Max = (int)((1 + o_size) * ((float)(P) / K))
    capacity_increase = 1.3

    C = P * ((K * Max) / (float)(P))

    Max = np.int(Max * capacity_increase)

    Max = np.min([Max, P])

    Z = None
    W = None
    Theta = None

    prev_ll = 0
    train_lls = []
    test_lls = []
    objectives = []

    partial_accs = []

    for i in range(max_iter):
        print("###############")
        print("BCD ", i)

        (W, z) = zstep(S, W, Theta, P, K, C, lmbda, Max, dual_tol, dual_max_iter)

        if i == 0:
            Theta0 = None
        else:
            Theta0 = Theta

        lmbdas = make_lmbda_matrix(Z, P, lmbda, W)
        Theta = QUIC_lmat(S, lmbdas, Theta0)

        train_ll = fn2.get_likelihood(S, Theta)

        obj = fn2.get_obj(S, Theta, W, lmbda)

        print("Train, likelihood:", train_ll)
        print("TrainR, objective:", obj)

        if (np.abs(train_ll - prev_ll) <= tol):
            break;
        prev_ll = train_ll

    pathways = get_pways_from_W(W, P, K, Max, capacity_increase, o_size)

    return (Theta, pathways)


def get_pways_from_W(W, P, K, Max, capacity_increase, o_size):
    (w, v) = np.linalg.eigh(W)
    w = np.maximum(w, 0)
    w = w[P - K:P]
    w = np.diag(np.sqrt(w))
    v = v[:, P - K:P]
    Z = np.matrix(v) * np.matrix(w)
    pathways = fn2.get_pathways_from_U_kmeans(Z.T, Max, capacity_increase, K, P, o_size)
    return pathways


def map_Z(Z, lmbda, beta, tau, K, P):
    Mrow = 1
    Mcol = np.sqrt(tau)
    C2 = beta

    prevZ = Z
    K = Z.shape[1]
    for piter in range(10):

        norm2s = []
        for i in range(P):
            norm2 = fn2.norm(Z[i, :], 2)
            norm2s.append(fn2.norm(Z[i, :], 1))

            if norm2 > Mrow:
                Z[i, :] /= (norm2 / Mrow)
                norm2 = fn2.norm(Z[i, :], 2)

        norm2 = fn2.norm(Z, 2)

        if (norm2 > C2):
            Z /= (norm2 / C2)

        for k in range(K):
            norm2 = fn2.norm(Z[:, k], 2)
            if norm2 > Mcol:
                Z[:, k] /= (norm2 / Mcol)

        if (prevZ == Z).all():
            break
        prevZ = Z
    return Z


def solve_z(T, P, alpha2, beta2, tau, z, etha=1, tol=1e-4, maxIter=600):
    if z is None:
        z = np.zeros(P)

    alpha2 /= tau
    prevf = 1e9

    for k in range(maxIter):
        M = T - np.diag(z)
        (w, v) = np.linalg.eigh(M)

        f = 0

        for i in range(P - beta2, P):
            if w[i] > 0:
                f += w[i]

        f += alpha2 * np.sum(z)
        # print "now f: ", f
        if (prevf - f) < tol:
            break
        prevf = f

        for i in range(len(w)):
            if i < P - beta2:
                w[i] = 0
                continue
            if w[i] >= 0:
                w[i] = 1
            else:
                w[i] = 0

        Mp = np.matrix(v) * np.matrix(np.diag(w)) * np.matrix(v.T)

        g = -np.diag(Mp) + alpha2

        if (k < maxIter / 2):
            z -= etha * g
        else:
            z -= etha * (1 / np.sqrt(k - maxIter / 2 + 1)) * g

        z = np.maximum(z, 0)

    # Now, let's construct W
    M = T - np.diag(z)
    (w, v) = np.linalg.eigh(M)

    for i in range(len(w)):
        if i < P - beta2:
            w[i] = 0
            continue
        if w[i] >= 0:
            w[i] = 1
        else:
            w[i] = 0

    W = np.matrix(v) * np.matrix(np.diag(w)) * np.matrix(v.T)
    W *= tau

    return (W, z)


def zstep(S, W, Theta, P, K, C, lmbda, Max, dual_tol, dual_max_iter):
    coef = 1

    beta = np.sqrt(P * lmbda / (2 * lmbda))
    beta /= coef

    tau = beta ** 2 / K

    # Here, we will do hard initialization
    if (W is None):
        (pathways, Z) = init_pathways(S, P, C, K, Max)

        Z = map_Z(Z, lmbda, beta, tau, K, P)

        W = np.matrix(Z) * np.matrix(Z.T)

        get_pways_from_W(W, P, K, Max, 1.3, .25)

        return (W, None)

    Theta = np.abs(Theta)

    beta2 = K
    (W, z) = solve_z(Theta, P, 1.0, beta2, tau, None, tol=dual_tol, maxIter=dual_max_iter)

    return (W, z)


def add_Z_to_pway(Z, pways, start, end):
    for i in range(start, end):
        for t in range(len(Z[i - start])):
            k = Z[i - start][t]
            pways[k].append(i)


def get_likelihood(X, T):
    X = np.array(X)
    return np.linalg.slogdet(T)[1] - np.sum(X * T)
