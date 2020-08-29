"""k-Jl methods
"""

# 1. standard libraries
import multiprocessing
import warnings

# 2. third-party packages
import numpy as np

# 3. your own package
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import pairwise_distances
from datetime import datetime

# set some stuffs
from sklearn.utils import resample

from kjl.utils.data import data_info

warnings.simplefilter('always', FitFailedWarning)
multiprocessing.set_start_method('spawn', True)


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


def kernelJLInitialize(X, sigma, d, m, n, centering=0, independent_row_col=1, random_state=100):
    """Project data to d-dimension spaces

    Parameters
    ----------
    X
    sigma
    d
    m
    n
    centering
    independent_row_col

    Returns
    -------

    """
    print(f'random_state: {random_state}')
    np.random.seed(random_state)  # don't remove
    N, D = X.shape

    if independent_row_col:
        # # indRow and indCol are independent each other
        # indRow = np.random.randint(N, size=n)
        # indCol = np.random.randint(N, size=m)

        indRow = resample(range(N), n_samples=n, random_state=random_state,
                          replace=False)
        indCol = resample(range(N), n_samples=m, random_state=random_state,
                          replace=False)

    else:
        # Random select max(n,m) rows
        # indices = np.random.randint(N, size=max(n, m))
        indices = resample(range(N), n_samples=max(n, m), random_state=random_state,
                           replace=False)
        # In indRow and indCol, one includes another
        indRow = indices[0:n]
        indCol = indices[0:m]

    Xrow = X[indRow, :]  # nxD
    Xcol = X[indCol, :]  # mXD

    # compute Gaussian kernel gram matrix A (i.e., K generated from a subset of X)
    # print(Xrow, Xcol)
    A = getGaussianGram(Xrow, Xcol, sigma)  # nXm
    # print(A)
    if centering:
        # subtract the mean of col from each element in a col
        A = A - np.mean(A, axis=0)

    # Projection matrix: ZK (nXd = nXm * mXd) # matrix product : Gaussian sketching
    # U = np.dot(A, np.random.multivariate_normal([0] * d, np.diag([1] * d), m))
    random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)
    U = np.matmul(A, random_matrix)  # preferred for matrix multiplication
    print("Finished getting the projection matrix")

    # Obtain gram between full data and Xrow (Nxn)
    K = getGaussianGram(X, Xrow, sigma)

    # projected data (Nxd = NXn * nXd)
    KU = np.matmul(K, U)  # preferred for matrix multiplication
    print("Projected data")

    return KU, U, Xrow, random_matrix, A


# def merge_parameters(tuned_parameters):
#     if len(tuned_parameters.keys()) <=1:
#         return tuned_parameters
#     for i,
#
#     return


class KJL():
    def __init__(self, kjl_params, debug=False):
        self.kjl_params = kjl_params
        self.debug = debug
        self.random_state = self.kjl_params['random_state']

    def fit(self, X_train):
        """Get KJL related data, such as, U, X_row, random matrix, and A

        Parameters
        ----------
        X_train

        Returns
        -------

        """
        if self.kjl_params['kjl']:
            d = self.kjl_params['kjl_d']
            n = self.kjl_params['kjl_n']
            q = self.kjl_params['kjl_q']

            start = datetime.now()
            n = n or max([200, int(np.floor(X_train.shape[0] / 100))])  # n_v: rows; m_v: cols. 200, 100?
            m = n
            if hasattr(self, 'sigma') and self.sigma:
                sigma = self.sigma
            else:
                # compute sigma
                dists = pairwise_distances(X_train)
                if self.debug:
                    # for debug
                    _qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                    _sigmas = np.quantile(dists, _qs)  # it will cost time
                    print(f'train set\' sigmas with qs: {list(zip(_sigmas, _qs))}')
                sigma = np.quantile(dists, q)
                if sigma == 0:
                    print(f'sigma:{sigma}, and use 1e-7 for the latter experiment.')
                    sigma = 1e-7
            # self.sigma_kjl = sigma
            print("sigma: {}".format(sigma))

            # project train data
            # if debug: data_info(X_train, name='before KJL, X_train')
            # X_train, self.U_kjl, self.Xrow_kjl, self.random_matrix, self.A = kernelJLInitialize(X_train, self.sigma_kjl,
            #                                                                                     d, m, n, centering=1,
            #                                                                                     independent_row_col=0,
            #                                                                                     random_state=self.random_state)

            print(f'random_state: {self.random_state}')
            np.random.seed(self.random_state)  # don't remove
            N, D = X_train.shape

            independent_row_col = 0
            if independent_row_col:
                # # indRow and indCol are independent each other
                # indRow = np.random.randint(N, size=n)
                # indCol = np.random.randint(N, size=m)

                indRow = resample(range(N), n_samples=n, random_state=self.random_state,
                                  replace=False)
                indCol = resample(range(N), n_samples=m, random_state=self.random_state,
                                  replace=False)

            else:
                # Random select max(n,m) rows
                # indices = np.random.randint(N, size=max(n, m))
                indices = resample(range(N), n_samples=max(n, m), random_state=self.random_state,
                                   replace=False)
                # In indRow and indCol, one includes another
                indRow = indices[0:n]
                indCol = indices[0:m]

            Xrow = X_train[indRow, :]  # nxD
            Xcol = X_train[indCol, :]  # mXD

            # compute Gaussian kernel gram matrix A (i.e., K generated from a subset of X)
            # print(Xrow, Xcol)
            A = getGaussianGram(Xrow, Xcol, sigma)  # nXm
            # print(A)
            centering = 1
            if centering:
                # subtract the mean of col from each element in a col
                A = A - np.mean(A, axis=0)

            # Projection matrix: ZK (nXd = nXm * mXd) # matrix product : Gaussian sketching
            # U = np.dot(A, np.random.multivariate_normal([0] * d, np.diag([1] * d), m))
            random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)
            U = np.matmul(A, random_matrix)  # preferred for matrix multiplication
            print("Finished getting the projection matrix")

            # Obtain gram between full data and Xrow (Nxn)
            K = getGaussianGram(X_train, Xrow, sigma)

            # projected data (Nxd = NXn * nXd)
            KU = np.matmul(K, U)  # preferred for matrix multiplication
            print("Projected data")

            X_train = KU
            self.A = A
            self.U = U
            self.K = K
            self.Xrow = Xrow
            self.random_matrix = random_matrix
            self.sigma_kjl = sigma

            if self.debug: data_info(X_train, name='after KJL, X_train')

            end = datetime.now()
            kjl_train_time = (end - start).total_seconds()
            print("kjl on train set took {} seconds".format(kjl_train_time))

        else:
            kjl_train_time = 0

        self.kjl_train_time = kjl_train_time


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
        # if self.kjl_params['kjl']:
        #     # # for debug
        #     if self.debug:
        #         data_info(X, name='X_test_std')
        #         _qs = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
        #         _sigmas = np.quantile(pairwise_distances(X), _qs)
        #         print(f'test set\' sigmas with qs: {list(zip(_sigmas, _qs))}')
        #
        #     start = datetime.now()
        #     print("Projecting test data")
        #     K = getGaussianGram(X, self.Xrow, self.sigma_kjl)  # get kernel matrix using rbf
        #     X = np.matmul(K, self.U)
        #     if self.verbose > 5: print(K, K.shape, self.U, self.U.shape, X)
        #     if self.debug: data_info(X, name='after KJL, X')
        #     end = datetime.now()
        #     kjl_test_time = (end - start).total_seconds()
        #     print("kjl on test set took {} seconds".format(kjl_test_time))
        #
        # else:
        #     kjl_test_time = 0
        #
        # self.kjl_test_time = kjl_test_time

        K = getGaussianGram(X, self.Xrow, self.sigma_kjl)  # get kernel matrix using rbf
        X = np.matmul(K, self.U)

        return X

    def update(self, x):
        fix_U = True
        if fix_U:  # U: nxn
            # (what about self.sigma_kjl? (should we update it? ))
            self.Xrow[-1] = x
            # # only one column and one row will change comparing with the previous one, so we need to optimize it.
            # A = getGaussianGram(self.Xrow, self.Xrow, self.sigma_kjl)
            # centering = True
            # if centering:
            #     # subtract the mean of col from each element in a col
            #     A = A - np.mean(A, axis=0)

            A1 = getGaussianGram(self.Xrow[:-1, :], x, self.sigma_kjl)
            A1 = A1.reshape(-1, 1)
            _v = np.asarray([1.0])  # kernel(A1, A1) = 1
            A1 = np.concatenate([A1, _v.reshape(-1, 1)], axis=0).reshape(-1, )
            self.A[-1, :] = A1
            self.A[:, -1] = A1.transpose()
            centering = True
            if centering:
                # subtract the mean of col from each element in a col
                self.A = self.A - np.mean(self.A, axis=0)

            self.U_kjl = np.matmul(self.A, self.random_matrix)  # preferred for matrix multiplication

        else:  # the size of U : n <- n+1
            # self.Xrow = np.concatenate([self.Xrow, x_copy], axis=0)
            # # only one column and one row will change comparing with the previous one, so we need to optimize it.
            # # To be modified?
            # A = getGaussianGram(self.Xrow, self.Xrow, self.sigma_kjl)

            A1 = getGaussianGram(self.Xrow, x, self.sigma_kjl)
            self.A = np.concatenate([self.A, A1], axis=1)
            _v = np.asarray([1.0])  # kernel(A1, A1) = 1
            A1 = np.concatenate([A1, _v.reshape(-1, 1)], axis=0).reshape(1, -1)
            self.A = np.concatenate([self.A, A1], axis=0)

            centering = True
            if centering:
                # subtract the mean of col from each element in a col
                self.A = self.A - np.mean(self.A, axis=0)

            self.Xrow = np.concatenate([self.Xrow, x], axis=0)
            d = self.params['kjl_d']
            n = self.Xrow.shape[0]  # n <= n+1
            self.random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), n)

            self.U_kjl = np.matmul(self.A, self.random_matrix)  # preferred for matrix multiplication
