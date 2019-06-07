import numpy as np
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from statsmodels.stats.correlation_tools import cov_nearest


class Block:

    def __init__(self, dim, idx, block_min_size=None, block_max_size=None, block_value=None):
        self.idx = idx
        self.block_size = np.random.randint(low=block_min_size, high=block_max_size)
        self.block_value = np.random.uniform(0.9, 1) if block_value is None else block_value
        self.indices = self._generate_indices(dim, self.block_size)

    @staticmethod
    def _generate_indices(dim, block_size):
        return [
            np.random.randint(low=0, high=dim, size=2)
            for _ in range(block_size)
        ]

    def apply_block_precision_matrix(self, theta, value=None):
        block_value = self.block_value if value is None else 1.0
        for idx in self.indices:
            # insert at i,j and j,i since precision matrix
            # is symmetric
            theta[idx[0]][idx[1]] = block_value
            theta[idx[1]][idx[0]] = block_value

        return theta

    def __str__(self):
        return (f"Block Index: {self.idx} \n"
                f"Block Size: {self.block_size} \n"
                f"Block Value: {round(self.block_value, 5)} \n")


def standardize(X):
    ret = (X - X.mean(axis=0)) / np.std(X, axis=0, ddof=1)
    return ret


def get_sample_cov(X):
    return np.cov(X.T)


def generate_sigma(p):
    return make_spd_matrix(p)


def generate_random_data(n, sigma):
    p = sigma.shape[0]
    return np.random.multivariate_normal(np.zeros(p), sigma, n)


def add_noise(theta, p, alpha, threshold=0.1):
    noise_mat = make_sparse_spd_matrix(dim=p,
                                       alpha=alpha,
                                       norm_diag=False,
                                       smallest_coef=-threshold,
                                       largest_coef=threshold)
    np.fill_diagonal(theta, 0.0)
    theta_star = cov_nearest(noise_mat + theta, method="clipped", threshold=0.1)
    return theta_star


def create_blocks(p, n_blocks, block_min_size, block_max_size):
    blocks = [
        Block(p, i, block_min_size, block_max_size)
        for i in range(n_blocks)
    ]

    return blocks


def generate_theta_star_gowl(p,
                             alpha,
                             noise,
                             n_blocks=None,
                             block_min_size=None,
                             block_max_size=None,
                             blocks=None,
                             trials=1):
    """
    Generate \Theta^* for synthetic data
    :param p: dimension of precision matrix
    :param alpha: entries of theta set to zero with probability alpha
    :param noise: level of noise add to theta star
    :param n_blocks: K blocks
    :param block_min_size: smallest block size
    :param block_max_size: largest block size
    :param blocks: custom blocks
    :param trials: number thetas to generate with noise added
    """
    if all([n_blocks, block_min_size, block_max_size]):
        blocks = create_blocks(p, n_blocks, block_min_size, block_max_size)

    theta_with_blocks_verbose = np.zeros((p, p))
    np.fill_diagonal(theta_with_blocks_verbose, 1.0)
    for block in blocks:
        theta_with_blocks_verbose = block.apply_block_precision_matrix(theta_with_blocks_verbose)

    if len(theta_with_blocks_verbose[np.where(theta_with_blocks_verbose != 0)]) < (p + 2 * len(blocks)):
        print('Problem')

    # Add Gaussian noise 10 times
    thetas_with_noise = list()
    for _ in range(trials):
        theta_blocks_with_noise = add_noise(theta_with_blocks_verbose,
                                            p,
                                            alpha,
                                            noise)
        thetas_with_noise.append(theta_blocks_with_noise)

    return thetas_with_noise, blocks, theta_with_blocks_verbose


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


def generate_synthetic_data(n_blocks,
                            p,
                            block_min_size,
                            block_max_size,
                            alpha,
                            noise,
                            noise_trails=10):
    blocks = generate_blocks(n_blocks, p, block_min_size, block_max_size)
    thetas_with_noise, _, theta_blocks = generate_theta_star_gowl(p=p,
                                                                  alpha=alpha,
                                                                  noise=noise,
                                                                  blocks=blocks,
                                                                  trials=noise_trails)

    scov_matrices = list()
    X_matrices = list()
    for theta_noise in thetas_with_noise:
        sigma = np.linalg.inv(theta_noise)
        p = sigma.shape[0]
        n = 100
        X = np.random.multivariate_normal(np.zeros(p), sigma, n)
        X = standardize(X)
        S = np.cov(X.T)
        scov_matrices.append(S)
        X_matrices.append(X)

    return thetas_with_noise, theta_blocks, scov_matrices, X_matrices
