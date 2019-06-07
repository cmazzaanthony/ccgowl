import numpy as np
import src.models.grab.pway_funcs as fn2
import scipy.io
import src.models.grab.GRAB as grab

lmbda = .2
K = 10
o_size = .3  # The size of overlap, as an input parameter
max_iter = 20
tol = 1e-4
dual_max_iter = 600
dual_tol = 1e-4

train = scipy.io.loadmat("path/genes.mat")['train']
train = fn2.standardize(train)
data = train
data = data.T

S = np.cov(data)

(Theta, blocks) = grab.BCD(S, lmbda=lmbda, K=K, o_size=o_size, max_iter=max_iter, tol=tol,
                           dual_max_iter=dual_max_iter, dual_tol=dual_tol)

print("Theta: ", Theta)
print("Overlappign Blocks: ", blocks)
