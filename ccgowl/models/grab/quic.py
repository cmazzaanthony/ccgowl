import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from pandas import *
import numpy as np

rpy2.robjects.numpy2ri.activate()

ro.r('source(\'imports.R\')')


#####################################
# ALL CODE TAKEN FROM HERE:
# https://github.com/mjhosseini/grab
#####################################


def is_psd(x):
    return np.all(np.linalg.eigvals(x) > 0)


def QUIC(S, lmbda):
    num_vars = S.shape[0]
    tmp = ro.r.matrix(S, nrow=num_vars, ncol=num_vars)

    ro.globalenv['S'] = tmp

    ro.r('res = QUIC(S,rho = ' + str(lmbda) + ')')
    print("QUIC iter: ", ro.r('res$iter'))
    print("QUIC regloglik: ", ro.r('res$regloglik'))
    Theta = ro.r('res$X')
    return np.matrix(Theta)


def QUIC_lmat(S, lmbdas, Theta0=None):
    num_vars = S.shape[0]

    tmp = ro.r.matrix(S, nrow=num_vars, ncol=num_vars)

    ro.globalenv['S'] = tmp

    tmp = ro.r.matrix(lmbdas, nrow=num_vars, ncol=num_vars)

    ro.globalenv['lmbdas'] = tmp

    if Theta0 is None:
        ro.r('res = QUIC(S,rho = lmbdas,tol=1e-4,maxIter=100)')
    else:
        Theta0 = np.array(Theta0)
        XX = np.array(np.linalg.inv(Theta0))
        tmp2 = ro.r.matrix(XX, nrow=num_vars, ncol=num_vars)
        ro.globalenv['T0'] = tmp2
        ro.r('res = QUIC(S,rho = lmbdas,tol=1e-4,maxIter=100,W.init=T0)')

    print("QUIC iter: ", ro.r('res$iter'))
    print("QUIC regloglik: ", ro.r('res$regloglik'))
    Theta = ro.r('res$X')
    return np.matrix(Theta)
