"""GMM
    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"
"""
import numpy as np
from sklearn.mixture import GaussianMixture

# load quickshift++
# using "pyximport.install()" fails for install quickshfit++ because it requires 'C++' in its setup.py.
# However, 1). pyximport does not use cythonize(). Thus it is not possible to do things like using compiler directives
# at the top of Cython files or compiling Cython code to C++.
# On the other hand, it is not recommended to let pyximport build code on end user side as it hooks into
# their import system.
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# *** Base on that, just use the following command to install quickshift++
# "python3 setup build; python3 setup install" to install "quickshift++"


class GMM(GaussianMixture):

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    # @execute_time
    def decision_function(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        return -1 * self.score_samples(X)

    def predict_proba(self, X):
        return -1 * self.score_samples(X)  #

    # def get_params(self, deep=True):
    #     return self.__dict__

    # def set_params(self, **params):
    #     super(GMM, self).set_params(**params)


def make_symmetric(matrix):
    n, _ = matrix.shape
    for i in range(n):
        for j in range(n):
            v = (matrix[i][j] + matrix[j][i]) / 2
            matrix[i][j], matrix[j][i] = v, v

    # print('matrix is symetric: ', np.all(matrix == matrix.T))
    return matrix


def compute_gmm_init(n_components=2, X=None, n_thres=1000, tot_clusters={}, tot_labels=[], covariance_type='full'):
    """ compute the init parameters of GMM

    Parameters
    ----------
    n_components
    X
    n_thres
    tot_clusters
    tot_labels
    covariance_type

    Returns
    -------

    """
    reg_covar = 1e-6
    weights = []
    N, n_features = X.shape
    means = np.empty((n_components, n_features))
    if covariance_type == 'full':
        covariances = np.empty((n_components, n_features, n_features))
        precisions_init = np.empty((n_components, n_features, n_features))  # # inverse of covariance matrix
    else:  # diag
        covariances = np.empty((n_components, n_features))
        precisions_init = np.empty((n_components, n_features))

    for k, (label_, nk) in enumerate(tot_clusters.items()):
        if k > n_components - 1: break
        idxs = np.where(tot_labels == label_)[0]  # get index of each cluster. np.where return tuple
        X_ = X[idxs]
        weights.append(nk / n_thres)
        means[k] = np.mean(X_, axis=0)
        if covariance_type == 'full':
            diff = X_ - means[k]
            covariances[k] = np.matmul(diff.T, diff) / nk
            covariances[k].flat[::n_features + 1] += reg_covar
            precisions_init[k] = np.linalg.inv(covariances[k])
            precisions_init[k].flat[::n_features + 1] += reg_covar  # make sure it is a positive matrix
        else:
            # X**2 - 2X*means + means**2
            avg_X2 = np.sum(X_ * X_, axis=0) / nk  # 1xd
            avg_means2 = means[k] ** 2  # 1xd
            avg_X_means = means[k] * np.sum(X_, axis=0) / nk  # 1xd
            covariances[k] = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
            precisions_init[k] = 1 / covariances[k]

    weights = np.asarray(weights, dtype=float)

    for k in range(n_components):
        if np.any(precisions_init[k] != precisions_init[k].T):
            # raise RuntimeError('not symmetric')
            precisions_init[k] = make_symmetric(precisions_init[k])

    return weights, means, precisions_init, covariances
