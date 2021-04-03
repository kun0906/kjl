"""k-Jl methods
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import resample

np.random.seed(42)

# __all__= ['_grow_tree'] # allow private functions (start with _) can be imported by using "import *"

"""
function [ K ] = getGaussianGram( Xrow, Xcol, sigma, gofast)
% Given two data sets, compute the cross gram between the two 
% Xrow = n by D, where D is the data dimension 
% Xcol = m by D, where D is the data dimension 
% goFast = 0 or 1, default value is 1. 
% K = n by m, gram matrix 

if nargin < 4
    goFast = 1;
end

if goFast == 1  
    A1 = sum(Xrow .* Xrow, 2);
    A2 = -2 * Xrow * Xcol';
    B = sum(Xcol .* Xcol, 2);
    K = bsxfun(@plus, A1, A2);
    K = bsxfun(@plus, K, B');
    K = exp(-K/sigma^2);
else 
    Dist = pdist2(Xrow, Xcol); 
    K = Dist.^2; 
    K = exp(-K/sigma^2);   
end 


end

"""


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
        A1 = np.expand_dims(np.power(np.linalg.norm(Xrow, axis=1), 2), axis=1)  # change vector to matrix 100x1
        # A1 = np.power(np.linalg.norm(Xrow, axis=1), 2)    # vector: shape(n, ), matrix: (n, 1)
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


class KJL:
    def __init__(self, params = None, verbose = 1):
        self.params = params
        self.random_state = 42  # params['random_state']
        self.verbose = verbose

    def fit(self, X_train, y_train=None, X_train_raw=None, y_train_raw=None):
        """Get KJL related data, such as, U, X_row, random matrix, and A

        Parameters
        ----------
        X_train

        Returns
        -------

        """

        #####################################################################################################
        # 1. Get sigma according to q_kjl
        d = self.params['kjl_d']
        n = self.params['kjl_n']
        q = self.params['kjl_q']
        N, D = X_train.shape

        if hasattr(self.params, 'sigma'):
            self.sigma = self.params.sigma
        else:
            # compute sigma
            dists = pairwise_distances(X_train)
            self.sigma = np.quantile(dists, q)
            if self.sigma == 0:
                print(f'sigma:{self.sigma}, and use 1e-7 for the latter experiment.')
                self.sigma = 1e-7
        # print(f'sigma: {self.sigma}, q_kjl: {q}, n_kjl: {n}, d_kjl: {d}, random_state: {self.random_state}',
        #       self.verbose)

        #####################################################################################################
        # 2. Get Xrow according to n
        # n = 100  # n or max([200, int(round(X_train.shape[0] / 100, 0))])  # n_v: rows; m_v: cols. 200, 100?
        m = n
        # To get the fixed random_matrix (R)
        np.random.seed(self.random_state)
        independent_row_col = 0
        if independent_row_col:
            # # indRow and indCol are independent each other
            # indRow = np.random.randint(N, size=n)
            # indCol = np.random.randint(N, size=m)
            indRow = resample(range(N), n_samples=n, random_state=self.random_state, stratify=y_train,
                              replace=False)
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
        # mprint(f"y_train: {Counter(y_train)}, y_row: {Counter(y_train[indRow])}",
        #        self.verbose, DEBUG)
        Xrow = X_train[indRow, :]  # nxD
        Xcol = X_train[indCol, :]  # mXD

        #####################################################################################################
        # 3. Get U according to Xrow
        # compute Gaussian kernel gram matrix A (i.e., K generated from a subset of X)
        A = getGaussianGram(Xrow, Xcol, self.sigma)  # nXm
        self.uncenter_A = A

        centering = 1
        if centering:
            # subtract the mean of col from each element in a col
            A = A - np.mean(A, axis=0)

        #####################################################################################################
        # 4. Projection matrix: ZK (nXd = nXm * mXd) # matrix product : Gaussian sketching
        # To get the fixed random_matrix (R). without this line, the results might be changed with time goes by
        np.random.seed(self.random_state)
        self.random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)  # mxd
        self.U = np.matmul(A, self.random_matrix)  # preferred for matrix multiplication
        self.Xrow = Xrow

        return self

    def transform(self, X):
        """Project X onto a lower space using kjl (i.e. kernel JL met)

        Parameters
        ----------
        X: array with shape (n_samples, n_feats)

        Returns
        -------
            X: array with shape (n_samples, d)
            "d" is the lower space dimension

        """
        K = getGaussianGram(X, self.Xrow, self.sigma)  # get kernel matrix
        X = np.matmul(K, self.U)  # K@U

        return X

#####################################################################################################
# Online KJL version
# class KJL():
#     def __init__(self, params):
#         self.params = params
#         self.random_state = params.random_state
#         self.verbose = params.verbose
#         self.i = 0
#         self.t = 1
#         self.fixed_U_size = True
#
#     def fit(self, X_train, y_train=None, X_train_raw=None, y_train_raw=None):
#         """Get KJL related data, such as, U, X_row, random matrix, and A
#
#         Parameters
#         ----------
#         X_train
#
#         Returns
#         -------
#
#         """
#
#         #####################################################################################################
#         # Step 1. Get sigma according to q_kjl
#         d = self.params.d_kjl
#         n = self.params.n_kjl
#         q = self.params.q_kjl
#
#         self.n_samples, _ = X_train.shape
#
#         if hasattr(self, 'sigma_kjl') and self.sigma_kjl:
#             sigma_kjl = self.sigma_kjl
#         elif hasattr(self.params, 'sigma_kjl'):
#             sigma_kjl = self.params.sigma_kjl
#         else:
#             # compute sigma
#             dists = pairwise_distances(X_train)
#             sigma_kjl = np.quantile(dists, q)
#             if sigma_kjl == 0:
#                 print(f'sigma_kjl:{sigma_kjl}, and use 1e-7 for the latter experiment.')
#                 sigma = 1e-7
#         mprint(f'sigma_kjl: {sigma_kjl}, q_kjl: {q}, n_kjl: {n}, d_kjl: {d}, random_state: {self.random_state}',
#                self.verbose, DEBUG)
#
#         #####################################################################################################
#         # Step 2. Get Xrow according to n
#         n = n or max([200, int(round(X_train.shape[0] / 100, 0))])  # n_v: rows; m_v: cols. 200, 100?
#         m = n
#         np.random.seed(self.random_state)  # to get the fixed random_matrix (R)
#         N, D = X_train.shape
#         independent_row_col = 0
#         if independent_row_col:
#             # # indRow and indCol are independent each other
#             # indRow = np.random.randint(N, size=n)
#             # indCol = np.random.randint(N, size=m)
#             indRow = resample(range(N), n_samples=n, random_state=self.random_state, stratify=y_train,
#             replace = False)
#             indCol = resample(range(N), n_samples=m, random_state=self.random_state, stratify=y_train,
#                               replace=False)
#         else:
#             # Random select max(n,m) rows
#             # indices = np.random.randint(N, size=max(n, m))
#             indices = resample(range(N), n_samples=max(n, m), random_state=self.random_state, stratify=y_train,
#                                replace=False)
#             # print(f'indx: {indices}, self.random_state: {self.random_state}, self.sigma_kjl: {sigma_kjl}')
#             # In indRow and indCol, one includes another
#             indRow = indices[0:n]
#             indCol = indices[0:m]
#         mprint(f"y_train: {Counter(y_train)}, y_row: {Counter(y_train[indRow])}",
#                self.verbose, DEBUG)
#         Xrow = X_train[indRow, :]  # nxD
#         Xcol = X_train[indCol, :]  # mXD
#
#         #####################################################################################################
#         # Step 3. Get U according to Xrow
#         # compute Gaussian kernel gram matrix A (i.e., K generated from a subset of X)
#         A = getGaussianGram(Xrow, Xcol, sigma_kjl)  # nXm
#         self.uncenter_A = A
#
#         centering =  self.params.centering_kjl
#         if centering:
#             # subtract the mean of col from each element in a col
#             A = A - np.mean(A, axis=0)
#
#         # Projection matrix: ZK (nXd = nXm * mXd) # matrix product : Gaussian sketching
#         # U = np.dot(A, np.random.multivariate_normal([0] * d, np.diag([1] * d), m))
#         self.random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)
#         self.U = np.matmul(A, self.random_matrix)  # preferred for matrix multiplication
#         # print("Finished getting the projection matrix")
#         self.A = A
#         self.Xrow = Xrow
#         self.sigma_kjl = sigma_kjl
#         self.Xrow_raw = X_train_raw[indRow]
#         self.yrow_raw = y_train_raw[indRow]
#         self.Xrow_accum = X_train_raw
#
#         return self
#
#     def transform(self, X):  # transform
#         """Project X onto a lower space using kjl (i.e. kernel JL met)
#
#         Parameters
#         ----------
#         X: array with shape (n_samples, n_feats)
#
#         Returns
#         -------
#             X: array with shape (n_samples, d)
#             "d" is the lower space dimension
#
#         """
#         K = getGaussianGram(X, self.Xrow, self.sigma_kjl)  # get kernel matrix using rbf
#         X = np.matmul(K, self.U)  # K@U
#
#         return X
#
#     # def update(self, X, y, X_raw, y_raw, std_inst):
#     #     """
#     #
#     #     Parameters
#     #     ----------
#     #     X_raw: unstd
#     #     std_inst
#     #
#     #     Returns
#     #     -------
#     #
#     #     """
#     #     # return 0
#     #     self.fixed_U_size = True
#     #     # X_raw =X
#     #     if self.fixed_U_size:  # U: nxn
#     #         # Get t first
#     #         n_new, _ = X.shape
#     #         ratio = n_new / (self.n_samples + n_new)
#     #         # t =5
#     #         t = int(round(ratio * self.params.n_kjl, 0))
#     #         t = 1 if t <1 else t
#     #         # X, y = sklearn.utils.shuffle(X, y, random_state=self.random_state)
#     #         # X = X[:t, :]  # random select t rows
#     #         # y = y[:t]
#     #         X, y = sklearn.utils.resample(X, y, n_samples=t, random_state=self.random_state, replace=False)
#     #         self.t = t
#     #         mprint(f't: {self.t}, ratio: {ratio}, n: {self.params.n_kjl}, n_samples: {self.n_samples}, y: {Counter(y)}', self.verbose, INFO)
#     #         mprint(f"before updating, y: {Counter(self.yrow_raw)}", self.verbose, INFO)
#     #         # if self.i + t >= self.Xrow_raw.shape[0]:
#     #         #     d = self.Xrow_raw.shape[0] - self.i
#     #         #     self.Xrow_raw[self.i: self.params.n_kjl, :] = X_raw[:d, :]
#     #         #     self.yrow_raw[self.i: self.params.n_kjl] = y_raw[:d]
#     #         #     self.i = t - d
#     #         #     self.Xrow_raw[0:self.i, :] = X_raw[d:, :]
#     #         #     self.yrow_raw[0:self.i] = y_raw[d:]
#     #         # else:
#     #         #     self.Xrow_raw[self.i: self.i + t, :] = X_raw
#     #         #     self.yrow_raw[self.i: self.i + t] = y_raw
#     #         #     self.i += t
#     #         mprint(f'self.kjl.i: {self.i}', self.verbose, INFO)
#     #
#     #         idxs = sklearn.utils.resample(range(self.params.n_kjl), n_samples= t, random_state=self.random_state*self.i, replace=False)
#     #         # print(f'idxs: {idxs}, X: {X}, self.Xrow: {self.Xrow}, y: {y}')
#     #         self.i = (self.i+t) %  self.params.n_kjl
#     #
#     #         is_fast = False
#     #         if not is_fast:
#     #             for _i, _x, _y in zip(idxs, X, y):
#     #                 self.Xrow[_i] = _x
#     #                 self.yrow_raw[_i] = _y
#     #
#     #             dists = pairwise_distances(self.Xrow)
#     #             self.sigma_kjl = np.quantile(dists, q=self.params.q_kjl)
#     #
#     #             # idx1 = sklearn.utils.resample(range(X_raw.shape[0]), n_samples=self.params.n_kjl,stratify=y_raw,
#     #             #                                             random_state=self.random_state, replace=False)
#     #             # # idx2 = resample(range(X_raw.shape[0]), n_samples=100, random_state=self.random_state, stratify=y_raw,
#     #             # #                    replace=False)
#     #             # # print(f'idx1-idx2: {[v1-v2 for v1, v2 in zip(idx1, idx2)]}, idx1: {idx1}, self.random_state: {self.random_state}')
#     #             # # idx1 = idx1[0:100]
#     #             # self.Xrow = X_raw[idx1, :]
#     #             # print(f'indx: {idx1}, self.sigma_kjl: {self.sigma_kjl}')
#     #             # # dists = pairwise_distances(self.Xrow)
#     #             # # self.sigma_kjl = np.quantile(dists, q=0.3)
#     #             self.A = getGaussianGram(self.Xrow, self.Xrow, self.sigma_kjl)
#     #         else:
#     #             self.Xrow[idxs] = X
#     #             self.yrow_raw[idxs] = y
#     #             #
#     #             # if std_inst is None:
#     #             #     self.Xrow = self.Xrow_raw
#     #             # else:
#     #             #     self.Xrow = std_inst.transform(self.Xrow_raw)
#     #             #
#     #
#     #             A = getGaussianGram(self.Xrow, X, self.sigma_kjl)  # nxt
#     #             self.uncenter_A[:, idxs] = A
#     #             self.uncenter_A[idxs, :] = A.transpose()
#     #             self.A = self.uncenter_A
#     #
#     #             # self.A = getGaussianGram(self.Xrow, self.Xrow, self.sigma_kjl)
#     #             # # self.A[:, -t:] = A1
#     #             # # A2 = getGaussianGram(X, X, self.sigma_kjl)  # kernel(x, x) = txt
#     #             # # A1[-t:, -t:] = A2  # t*n
#     #             # # self.A[-t:] = A1.T  # n*n
#     #
#     #         # print(f'is_fast: {is_fast}, A: {list(self.A)}, self.Xrow: {self.Xrow}')
#     #
#     #         # self.Xrow = sklearn.utils.shuffle(self.Xrow, random_state=self.random_state)
#     #
#     #         centering = self.params.centering_kjl
#     #         if centering:
#     #             # subtract the mean of col from each element in a col
#     #             self.A = self.A - np.mean(self.A, axis=0)
#     #
#     #         self.U = np.matmul(self.A, self.random_matrix)  # preferred for matrix multiplication
#     #         mprint(f"after updating, y_row: {Counter(self.yrow_raw)}", self.verbose, INFO)
#     #     else:  # increased U : n <- n+10
#     #
#     #         # self.Xrow_accum = np.concatenate([self.Xrow_accum, X], axis=0)
#     #         # dists = pairwise_distances(self.Xrow_accum)
#     #         # self.sigma_kjl = np.quantile(dists, q=0.3)
#     #         # mprint(f"Xrow_accum: {self.Xrow_accum.shape}", self.verbose, INFO)
#     #         # self.Xrow = sklearn.utils.resample(self.Xrow_accum, n_samples=100,
#     #         #                               random_state=self.random_state * self.i, replace=False)
#     #         # self.A = getGaussianGram(self.Xrow, self.Xrow, self.sigma_kjl)
#     #         #
#     #         # centering = self.params.centering_kjl
#     #         # if centering:
#     #         #     # subtract the mean of col from each element in a col
#     #         #     self.A = self.A - np.mean(self.A, axis=0)
#     #         # # self.random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)
#     #         # self.U = np.matmul(self.A, self.random_matrix)  # preferred for matrix multiplication
#     #         # mprint(f"after updating, y_row: {Counter(self.yrow_raw)}", self.verbose, INFO)
#     #
#     #         # Get t first
#     #         n_new, _ = X.shape
#     #         t = int(round(self.params.ratio_kjl * n_new, 0))
#     #         X_raw = sklearn.utils.shuffle(X, random_state=self.random_state)
#     #         X_raw = X[:t, :]  # random select t rows
#     #         self.t = t
#     #         mprint(f't: {self.t}', self.verbose, DEBUG)
#     #
#     #         # # only one column and one row will change comparing with the previous one, so we need to optimize it.
#     #         if std_inst is None:
#     #             self.Xrow = self.Xrow_raw
#     #             X= X_raw
#     #         else:
#     #             self.Xrow = std_inst.transform(self.Xrow_raw)
#     #             X = std_inst.transform(X_raw)
#     #         A1 = getGaussianGram(self.Xrow, X, self.sigma_kjl)  # n x t
#     #         A = np.concatenate([self.uncenter_A, A1], axis=1)  # A: nxn, A1 nxt => nx(n+t)
#     #         A2 = getGaussianGram(X, X, self.sigma_kjl)  # kernel(x, x) = t x t  # A2: txt
#     #         A1 = np.concatenate([A1.T, A2], axis=1)  # A1: tx(t+n)
#     #         self.A = np.concatenate([A, A1], axis=0)  # (n+t)x(n+t)
#     #         self.uncenter_A = self.A
#     #
#     #         centering = True
#     #         if centering:
#     #             # subtract the mean of col from each element in a col
#     #             self.A = self.A - np.mean(self.A, axis=0)
#     #
#     #         # self.Xrow = np.concatenate([self.Xrow, X], axis=0)
#     #         self.Xrow_raw = np.concatenate([self.Xrow_raw, X_raw], axis=0)
#     #         d = self.params.d_kjl
#     #         # update random_matrix
#     #         # np.random.seed(self.random_state)  # there is no need to fixed the new_random_matrix (R)
#     #         M1 = np.random.multivariate_normal([0] * d, np.diag([1] * d), t)  # means, cov, size : nxt
#     #         # print(f'M1: {M1}')
#     #         self.new_random_matrix = np.concatenate([self.random_matrix, M1], axis=0)  # (n+t)xd
#     #         self.random_matrix = self.new_random_matrix
#     #         self.U = np.matmul(self.A, self.random_matrix)
#     #
#     #     return  self
#     #
#     # def update2(self, X, y, X_raw, y_raw, std_inst):
#     #     """
#     #
#     #     Parameters
#     #     ----------
#     #     X_raw: unstd
#     #     std_inst
#     #
#     #     Returns
#     #     -------
#     #
#     #     """
#     #     # return 0
#     #     self.fixed_U_size = True
#     #     # X_raw =X
#     #     if self.fixed_U_size:  # U: nxn
#     #         # Get t first
#     #         n_new, _ = X_raw.shape
#     #         ratio = n_new / (self.n_samples + n_new)
#     #         t = int(round(ratio * self.params.n_kjl, 0))
#     #         t = 1 if t <1 else t
#     #         X_raw, y_raw = sklearn.utils.shuffle(X_raw, y_raw, random_state=self.random_state)
#     #         X_raw = X_raw[:t, :]  # random select t rows
#     #         y_raw = y_raw[:t]
#     #         self.t = t
#     #         mprint(f't: {self.t}, ratio: {ratio}, n: {self.params.n_kjl}, n_samples: {self.n_samples}, y_raw: {Counter(y_raw)}', self.verbose, INFO)
#     #         mprint(f"before updating, y_row: {Counter(self.yrow_raw)}", self.verbose, INFO)
#     #         # if self.i + t >= self.Xrow_raw.shape[0]:
#     #         #     d = self.Xrow_raw.shape[0] - self.i
#     #         #     self.Xrow_raw[self.i: self.params.n_kjl, :] = X_raw[:d, :]
#     #         #     self.yrow_raw[self.i: self.params.n_kjl] = y_raw[:d]
#     #         #     self.i = t - d
#     #         #     self.Xrow_raw[0:self.i, :] = X_raw[d:, :]
#     #         #     self.yrow_raw[0:self.i] = y_raw[d:]
#     #         # else:
#     #         #     self.Xrow_raw[self.i: self.i + t, :] = X_raw
#     #         #     self.yrow_raw[self.i: self.i + t] = y_raw
#     #         #     self.i += t
#     #         print(f'self.kjl.i: {self.i}')
#     #
#     #         idxs = sklearn.utils.resample(range(self.params.n_kjl), n_samples= t, random_state=self.random_state*self.i)
#     #         print(f'idxs: {idxs}')
#     #         self.i = (self.i+t) %  self.params.n_kjl
#     #
#     #         for _i, _x, _y in zip(idxs, X_raw, y_raw):
#     #             self.Xrow_raw[_i] = _x
#     #             self.yrow_raw[_i] = _y
#     #
#     #         if std_inst is None:
#     #             self.Xrow = self.Xrow_raw
#     #         else:
#     #             self.Xrow = std_inst.transform(self.Xrow_raw)
#     #
#     #         self.A = getGaussianGram(self.Xrow, self.Xrow, self.sigma_kjl)
#     #
#     #         # A1 = getGaussianGram(self.Xrow, x, self.sigma_kjl)  # nxt
#     #         # self.A[:, -t:] = A1
#     #         # A2 = getGaussianGram(x, x, self.sigma_kjl)  # kernel(x, x) = txt
#     #         # A1[-t:, -t:] = A2  # t*n
#     #         # self.A[-t:] = A1.T  # n*n
#     #
#     #         # self.Xrow = sklearn.utils.shuffle(self.Xrow, random_state=self.random_state)
#     #
#     #
#     #         centering = True
#     #         if centering:
#     #             # subtract the mean of col from each element in a col
#     #             self.A = self.A - np.mean(self.A, axis=0)
#     #
#     #         self.U = np.matmul(self.A, self.random_matrix)  # preferred for matrix multiplication
#     #         mprint(f"after updating, y_row: {Counter(self.yrow_raw)}", self.verbose, DEBUG)
#     #     else:  # increased U : n <- n+10
#     #         # Get t first
#     #         n_new, _ = X_raw.shape
#     #         t = int(round(self.params.ratio_kjl * n_new, 0))
#     #         X_raw = sklearn.utils.shuffle(X_raw, random_state=self.random_state)
#     #         X_raw = X_raw[:t, :]  # random select t rows
#     #         self.t = t
#     #         mprint(f't: {self.t}', self.verbose, DEBUG)
#     #
#     #         # # only one column and one row will change comparing with the previous one, so we need to optimize it.
#     #         if std_inst is None:
#     #             self.Xrow = self.Xrow_raw
#     #             X= X_raw
#     #         else:
#     #             self.Xrow = std_inst.transform(self.Xrow_raw)
#     #             X = std_inst.transform(X_raw)
#     #         A1 = getGaussianGram(self.Xrow, X, self.sigma_kjl)  # n x t
#     #         A = np.concatenate([self.uncenter_A, A1], axis=1)  # A: nxn, A1 nxt => nx(n+t)
#     #         A2 = getGaussianGram(X, X, self.sigma_kjl)  # kernel(x, x) = t x t  # A2: txt
#     #         A1 = np.concatenate([A1.T, A2], axis=1)  # A1: tx(t+n)
#     #         self.A = np.concatenate([A, A1], axis=0)  # (n+t)x(n+t)
#     #         self.uncenter_A = self.A
#     #
#     #         centering = True
#     #         if centering:
#     #             # subtract the mean of col from each element in a col
#     #             self.A = self.A - np.mean(self.A, axis=0)
#     #
#     #         # self.Xrow = np.concatenate([self.Xrow, X], axis=0)
#     #         self.Xrow_raw = np.concatenate([self.Xrow_raw, X_raw], axis=0)
#     #         d = self.params.d_kjl
#     #         # update random_matrix
#     #         # np.random.seed(self.random_state)  # there is no need to fixed the new_random_matrix (R)
#     #         M1 = np.random.multivariate_normal([0] * d, np.diag([1] * d), t)  # means, cov, size : nxt
#     #         # print(f'M1: {M1}')
#     #         self.new_random_matrix = np.concatenate([self.random_matrix, M1], axis=0)  # (n+t)xd
#     #         self.random_matrix = self.new_random_matrix
#     #         self.U = np.matmul(self.A, self.random_matrix)
