"""k-Jl methods
"""

# 1. standard libraries
import copy
import multiprocessing
import warnings

# 2. third-party packages
from collections import Counter

import numpy as np

# 3. your own package
import sklearn
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import pairwise_distances
from datetime import datetime

# set some stuffs
from sklearn.utils import resample

from config import DEBUG
from kjl.utils.data import data_info
from kjl.utils.tool import mprint

warnings.simplefilter('always', FitFailedWarning)
multiprocessing.set_start_method('spawn', True)

np.random.seed(100)


# __all__= ['_grow_tree'] # allow private functions (start with _) can be imported by using "import *"

def getGaussianGram(Xrow, Xcol, sigma, goFast=1):
    """ get kernel (Gaussian) gram matrix
    The Gram matrix K is deﬁned as $K_ij = K(X_i , X_j) over a (sub) sample X = {X _i}, i=1,...,,n
    Parameters
    ----------
    Xrow
    Xcol
    sigma
    goFast

    Returns
    -------

    """
    if goFast == 1:
        A1 = np.expand_dims(np.power(np.linalg.norm(Xrow, axis=1), 2), axis=1)
        A2 = -2 * np.matmul(Xrow, np.transpose(Xcol))
        B = np.power(np.linalg.norm(Xcol, axis=1), 2)
        K = np.add(np.add(A1, A2), np.transpose(B))
        K = np.exp(-K * 1 / sigma ** 2)

    else:
        # Dist = np.linalg.norm(Xrow - Xcol)  # it's wrong!
        # Dist= cdist(Xrow, Xcol, metric='euclidean')
        Dist = pairwise_distances(Xrow, Y=Xcol, metric='euclidean')
        K = np.exp(-np.power(Dist, 2) * 1 / sigma ** 2)

    return K



#
# def kernelJLInitialize(X, sigma, d, m, n, centering=0, independent_row_col=1, random_state=100):
#     """Project data to d-dimension spaces
#
#     Parameters
#     ----------
#     X
#     sigma
#     d
#     m
#     n
#     centering
#     independent_row_col
#
#     Returns
#     -------
#
#     """
#     print(f'random_state: {random_state}')
#     np.random.seed(random_state)  # don't remove
#     N, D = X.shape
#
#     if independent_row_col:
#         # # indRow and indCol are independent each other
#         # indRow = np.random.randint(N, size=n)
#         # indCol = np.random.randint(N, size=m)
#
#         indRow = resample(range(N), n_samples=n, random_state=random_state,
#                           replace=False)
#         indCol = resample(range(N), n_samples=m, random_state=random_state,
#                           replace=False)
#
#     else:
#         # Random select max(n,m) rows
#         # indices = np.random.randint(N, size=max(n, m))
#         indices = resample(range(N), n_samples=max(n, m), random_state=random_state,
#                            replace=False)
#         # In indRow and indCol, one includes another
#         indRow = indices[0:n]
#         indCol = indices[0:m]
#
#     Xrow = X[indRow, :]  # nxD
#     Xcol = X[indCol, :]  # mXD
#
#     # compute Gaussian kernel gram matrix A (i.e., K generated from a subset of X)
#     # print(Xrow, Xcol)
#     A = getGaussianGram(Xrow, Xcol, sigma)  # nXm
#     # print(A)
#     if centering:
#         # subtract the mean of col from each element in a col
#         A = A - np.mean(A, axis=0)
#
#     # Projection matrix: ZK (nXd = nXm * mXd) # matrix product : Gaussian sketching
#     # U = np.dot(A, np.random.multivariate_normal([0] * d, np.diag([1] * d), m))
#     random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)
#     U = np.matmul(A, random_matrix)  # preferred for matrix multiplication
#     print("Finished getting the projection matrix")
#
#     # Obtain gram between full data and Xrow (Nxn)
#     K = getGaussianGram(X, Xrow, sigma)
#
#     # projected data (Nxd = NXn * nXd)
#     KU = np.matmul(K, U)  # preferred for matrix multiplication
#     print("Projected data")
#
#     return KU, U, Xrow, random_matrix, A


class KJL():
    def __init__(self, params):
        self.params = params
        self.random_state = params.random_state
        self.verbose = params.verbose
        self.i = 0
        self.t = 1
        self.fixed_U_size = True

    def fit(self, X_train, y_train=None):
        """Get KJL related data, such as, U, X_row, random matrix, and A

        Parameters
        ----------
        X_train

        Returns
        -------

        """

        #####################################################################################################
        # Step 1. Get sigma according to q_kjl
        d = self.params.d_kjl
        n = self.params.n_kjl
        q = self.params.q_kjl

        self.n_samples, _ = X_train.shape

        if hasattr(self, 'sigma') and self.sigma:
            sigma = self.sigma
        else:
            # compute sigma
            dists = pairwise_distances(X_train)
            sigma = np.quantile(dists, q)
            if sigma == 0:
                print(f'sigma:{sigma}, and use 1e-7 for the latter experiment.')
                sigma = 1e-7
        mprint(f'sigma: {sigma}, q_kjl: {q}, n_kjl: {n}, d_kjl: {d}, random_state: {self.random_state}',
               self.verbose, DEBUG)

        #####################################################################################################
        # Step 2. Get Xrow according to n
        n = n or max([200, int(round(X_train.shape[0] / 100, 0))])  # n_v: rows; m_v: cols. 200, 100?
        m = n
        np.random.seed(self.random_state)  # to get the fixed random_matrix (R)
        N, D = X_train.shape
        independent_row_col = 0
        if independent_row_col:
            # # indRow and indCol are independent each other
            # indRow = np.random.randint(N, size=n)
            # indCol = np.random.randint(N, size=m)
            indRow = resample(range(N), n_samples=n, random_state=self.random_state, stratify=y_train,
            replace = False)
            indCol = resample(range(N), n_samples=m, random_state=self.random_state, stratify=y_train,
                              replace=False)
        else:
            # Random select max(n,m) rows
            # indices = np.random.randint(N, size=max(n, m))
            indices = resample(range(N), n_samples=max(n, m), random_state=self.random_state, stratify=y_train,
                               replace=False)
            # In indRow and indCol, one includes another
            indRow = indices[0:n]
            indCol = indices[0:m]
        mprint(f"y_train: {Counter(y_train)}, y_row: {Counter(y_train[indRow])}",
               self.verbose, DEBUG)
        Xrow = X_train[indRow, :]  # nxD
        Xcol = X_train[indCol, :]  # mXD

        #####################################################################################################
        # Step 3. Get U according to Xrow
        # compute Gaussian kernel gram matrix A (i.e., K generated from a subset of X)
        A = getGaussianGram(Xrow, Xcol, sigma)  # nXm
        self.uncenter_A = A

        centering = 1
        if centering:
            # subtract the mean of col from each element in a col
            A = A - np.mean(A, axis=0)

        # Projection matrix: ZK (nXd = nXm * mXd) # matrix product : Gaussian sketching
        # U = np.dot(A, np.random.multivariate_normal([0] * d, np.diag([1] * d), m))
        self.random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)
        self.U = np.matmul(A, self.random_matrix)  # preferred for matrix multiplication
        # print("Finished getting the projection matrix")
        self.A = A
        self.Xrow = Xrow
        self.sigma_kjl = sigma

        if self.verbose >= DEBUG: data_info(X_train, name='after KJL, X_train')

        return self

    def transform(self, X):  # transform
        """Project X onto a lower space using kjl (i.e. kernel JL met)

        Parameters
        ----------
        X: array with shape (n_samples, n_feats)

        Returns
        -------
            X: array with shape (n_samples, d)
            "d" is the lower space dimension

        """
        K = getGaussianGram(X, self.Xrow, self.sigma_kjl)  # get kernel matrix using rbf
        X = np.matmul(K, self.U)  # K@U

        return X

    def update(self, X):

        self.fixed_U_size = False
        if self.fixed_U_size:  # U: nxn
            # Get t first
            n_new, _ = X.shape
            ratio = n_new / (self.n_samples + n_new)
            t = int(round(ratio * self.params.n_kjl, 0))
            X = sklearn.utils.shuffle(X, random_state=self.random_state)
            X = X[:t, :]  # random select t rows
            self.t = t
            mprint(f't: {self.t}, ratio: {ratio}, n: {self.params.n_kjl}, n_samples: {self.n_samples}', self.verbose, DEBUG)

            if self.i + t > self.Xrow.shape[0]:
                d = self.Xrow.shape[0] - self.i
                self.Xrow[self.i: d, :] = X[:d, :]
                self.i = t - d
                self.Xrow[0:self.i, :] = X[d:, :]
            else:
                self.Xrow[self.i: self.i + t, :] = X
                self.i += t
            self.A = getGaussianGram(self.Xrow, self.Xrow, self.sigma_kjl)

            centering = True
            if centering:
                # subtract the mean of col from each element in a col
                self.A = self.A - np.mean(self.A, axis=0)

            self.U = np.matmul(self.A, self.random_matrix)  # preferred for matrix multiplication

        else:  # increased U : n <- n+10
            # Get t first
            n_new, _ = X.shape
            t = int(round(self.params.ratio_kjl * n_new, 0))
            X = sklearn.utils.shuffle(X, random_state=self.random_state)
            X = X[:t, :]  # random select t rows
            self.t = t
            mprint(f't: {self.t}', self.verbose, DEBUG)

            # # only one column and one row will change comparing with the previous one, so we need to optimize it.
            A1 = getGaussianGram(self.Xrow, X, self.sigma_kjl)  # n x t
            A = np.concatenate([self.uncenter_A, A1], axis=1)  # A: nxn, A1 nxt => nx(n+t)
            A2 = getGaussianGram(X, X, self.sigma_kjl)  # kernel(x, x) = t x t  # A2: txt
            A1 = np.concatenate([A1.T, A2], axis=1)  # A1: tx(t+n)
            self.A = np.concatenate([A, A1], axis=0)  # (n+t)x(n+t)
            self.uncenter_A = self.A

            centering = True
            if centering:
                # subtract the mean of col from each element in a col
                self.A = self.A - np.mean(self.A, axis=0)

            self.Xrow = np.concatenate([self.Xrow, X], axis=0)
            d = self.params.d_kjl
            # update random_matrix
            # np.random.seed(self.random_state)  # there is no need to fixed the new_random_matrix (R)
            M1 = np.random.multivariate_normal([0] * d, np.diag([1] * d), t)  # means, cov, size : nxt
            # print(f'M1: {M1}')
            self.new_random_matrix = np.concatenate([self.random_matrix, M1], axis=0)  # (n+t)xd
            self.random_matrix = self.new_random_matrix
            self.U = np.matmul(self.A, self.random_matrix)
