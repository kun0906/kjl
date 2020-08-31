"""GMM

    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"
"""

import numpy as np
from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.cluster import MeanShift
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from datetime import datetime
# load quickshift++
# using "pyximport.install()" fails for install quickshfit++ because it requires 'C++' in its setup.py.
# However, 1). pyximport does not use cythonize(). Thus it is not possible to do things like using compiler directives
# at the top of Cython files or compiling Cython code to C++.
# On the other hand, it is not recommended to let pyximport build code on end user side as it hooks into
# their import system.
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# *** Base on that, just use the following command to install quickshift++
# "python3 setup build; python3 setup install" to install "quickshift++"
from QuickshiftPP import QuickshiftPP


class ONLINE_GMM(GaussianMixture):

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False, covariances_init=None,
                 verbose=0, verbose_interval=10, **kwargs):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.weights_ = weights_init
        self.means_ = means_init
        self.covariances_ = covariances_init
        # self.covariances_init = covariances_init

    def decision_function(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        return -1 * self.score_samples(X)

    # def _e_step(self, X):
    #     """E step.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_feats)
    #
    #     Returns
    #     -------
    #     log_prob_norm : float
    #         Mean of the logarithms of the probabilities of each sample in X
    #
    #     log_responsibility : array, shape (n_samples, n_components)
    #         Logarithm of the posterior probabilities (or responsibilities) of
    #         the point of each sample in X.
    #     """
    #     log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
    #     return np.mean(log_prob_norm), log_resp
    #
    # def _estimate_log_weights(self):
    #     """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.
    #
    #     Returns
    #     -------
    #     log_weight : array, shape (n_components, )
    #     """
    #
    #     return np.log(self.weights_)

    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """

        n_samples, n_feats = X.shape
        n_components = self.n_components

        log_prob = np.zeros((n_samples, n_components))
        log_det = np.zeros((n_samples, n_components))
        for k, (mu, sigma) in enumerate(zip(self.means_, self.covariances_)):
            diff = (X.T - mu[:, np.newaxis])  # X and mu should be column vectors
            log_prob[:, k] = np.diag(-0.5 * np.matmul(np.matmul(diff.T, np.linalg.inv(sigma)), diff))

            v = np.log(np.linalg.det(sigma))
            if np.isnan(v) or np.isinf(v):
                # print(f'np.log(np.linalg.det(sigma)) is inf or nan, so we use 1e-6 as the value.')
                v = 1e-6
            log_det[:, k] = np.ones((n_samples,)) * v

        return -.5 * (n_feats * np.log(2 * np.pi) + log_det) + log_prob

    def _m_step(self, x, log_resp, n_samples):
        """M step.

        Parameters
        ----------
        x : array-like, shape (1, n_feats)

        log_resp : array-like, shape (1, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        if np.isnan(log_resp).any() or np.isinf(log_resp).any():
            print(f'log_resp: {log_resp}')
            return -1
        self.means_, self.covariances_ = self.online_means_covaricances(x, n_samples, self.means_, self.covariances_,
                                                                        log_resp, self.reg_covar)

        # not sure if the online update of the weights (of each component of GMM (i.e., $\pi_k$)) is correct.
        # self.weights_: shape (n_components, )
        # self.weights_ = (n_samples / (n_samples + 1)) * self.weights_ + (
        #             1 / (n_samples + 1)) * np.exp(log_resp).flatten()
        self.weights_ = (n_samples / (n_samples + x.shape[0])) * self.weights_ + (
                    x.shape[0] / (n_samples + x.shape[0])) * np.exp(
            log_resp).flatten()


    def online_means_covaricances(self, x, n_samples, means, covariances, log_resp, reg_covar=1e-6):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
        https://stackoverflow.com/questions/1346824/is-there-any-way-to-find-arithmetic-mean-better-than-sum-n

        Parameters
        ----------
        x : array-like, shape (1, n_feats)

        n_samples: int
            the number of datapoints used to fitted  the model until now.

        means: array with shape (n_components, n_feats)

        covariances: array with shape (n_components, n_feats, n_feats)

        log_resp: vector with shape (1, n_components):  the weights

        reg_covar:
            To avoid that the new covariance matrix is invalid.

        Returns
        -------
            new_means:
            new_covariances:

        """

        def _online_covaricane(x, one_new_mu, one_mu, one_covariances):
            """ get the covariance of the new component.
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

            Parameters
            ----------
            x : array-like, shape (1, n_feats)

            one_new_mu: array-like, shape (n_feats, )
                the updated means of one component

            one_mu: array-like, shape (n_feats, )
                the means of one component

            one_covariances: array-like, shape (n_feats, n_feats)
                the covariances of one component

            Returns
            -------
                new_covariance:
            """

            x = x.flatten()

            rows, cols = one_covariances.shape  # a "n_feats by n_feats" matrix
            for i in range(rows):
                for j in range(cols):
                    if i <= j:
                        one_covariances[i][j] += (x[i] - one_new_mu[i]) * (x[j] - one_mu[j])
                    else:  # j < i
                        one_covariances[i][j] = one_covariances[j][i]

            return one_covariances

        n_components, n_feats = means.shape
        new_means = np.zeros((n_components, n_feats))
        new_covariances = np.zeros((n_components, n_feats, n_feats))
        resp = np.exp(log_resp).flatten()
        for k in range(n_components):
            new_means[k] = resp[k] * (means[k] + (x - means[k]) / (n_samples + 1))
            # new_covariances[k] = corvainces[k] + (X-new_means[k]) * (X-means[k])
            # here would be overflow in add
            new_covariances[k] = resp[k] * \
                                 (covariances[k] + _online_covaricane(x, new_means[k], means[k], covariances[k]))

            new_covariances[k].flat[::n_feats + 1] += reg_covar  # x[startAt:endBefore:step], diagonal items

        return new_means, new_covariances


def meanshift_seek_modes(X, bandwidth=None, thres_n=100):
    start = datetime.now()
    clustering = MeanShift(bandwidth).fit(X)
    end = datetime.now()
    meanshift_training_time = (end - start).total_seconds()
    print("meanshift_training, it took {} seconds".format(meanshift_training_time))

    all_n_clusters = len(set(clustering.labels_))
    all_means_init = clustering.cluster_centers_
    all_labels_ = clustering.labels_

    cluster_centers = []
    for i in range(all_n_clusters):
        idxs = np.where(all_labels_ == i)[0]  # get index of each cluster. np.where return tuple
        if len(idxs) < thres_n:  # only keep the cluster which has more than "thres_n" datapoints
            continue
        # center_cluster_i = np.mean(X[idxs], axis=0)  # here is X, not X_std.
        center_cluster_i = all_means_init[i]  # here is X, not X_std.
        cluster_centers.append(center_cluster_i)

    means_init = np.asarray(cluster_centers, dtype=float)
    n_clusters = means_init.shape[0]
    print(f'--all clusters ({all_n_clusters}) when (bandwidth:{bandwidth}). However, only {n_clusters} '
          f'clusters have at least {thres_n} datapoints. Counter(labels_): {Counter(all_labels_)}, *** '
          f'len(Counter(labels_)): {all_n_clusters}')

    return means_init, n_clusters, meanshift_training_time, all_n_clusters


def quickshift_seek_modes(X, k=None, beta=0.9, thres_n=100):
    """Initialize GMM
            1) Download quickshift++ from github
            2) unzip and move the folder to your project
            3) python3 setup.py build
            4) python3 setup.py install
            5) from QuickshiftPP import QuickshiftPP
        :param X_train:
        :param k:
            # k: number of neighbors in k-NN
            # beta: fluctuation parameter which ranges between 0 and 1.

        :return:
        """
    start = datetime.now()
    if k <= 0 or k > X.shape[0]:
        print(f'k {k} is not correct, so change it to X.shape[0]')
        k = X.shape[0]
    print(f"number of neighbors in k-NN: {k}")
    # Declare a Quickshift++ model with tuning hyperparameters.
    model = QuickshiftPP(k=k, beta=beta)

    # Note the try catch cannot capture the model.fit() error because it is cython. How to capture the exception?
    try:
        model.fit(X)
    except Exception as e:
        msg = f'quickshift++ fit error: {e}'
        raise ValueError(msg)

    end = datetime.now()
    quick_training_time = (end - start).total_seconds()
    # lg.info("quick_training_time took {} seconds".format(quick_training_time))

    start = datetime.now()
    # print('quickshift fit finished')
    all_labels_ = model.memberships
    all_n_clusters = len(set(all_labels_))
    cluster_centers = []
    for i in range(all_n_clusters):
        idxs = np.where(all_labels_ == i)[0]  # get index of each cluster. np.where return tuple
        if len(idxs) < thres_n:  # only keep the cluster which has more than "thres_n" datapoints
            continue
        center_cluster_i = np.mean(X[idxs], axis=0)  # here is X, not X_std.
        cluster_centers.append(center_cluster_i)

    means_init = np.asarray(cluster_centers, dtype=float)
    n_clusters = means_init.shape[0]
    end = datetime.now()
    ignore_clusters_time = (end - start).total_seconds()
    print(f'*** quick_training_time: {quick_training_time}, ignore_clusters_time: {ignore_clusters_time}')
    print(f'--all clusters ({all_n_clusters}) when (k:({k}), beta:{beta}). However, only {n_clusters} '
          f'clusters have at least {thres_n} datapoints. Counter(labels_): {Counter(all_labels_)}, *** '
          f'len(Counter(labels_)): {all_n_clusters}')
    return means_init, n_clusters, quick_training_time, all_n_clusters


def get_means_init(X, k=None, beta=0.9, thres_n=100):
    """Initialize GMM
        1) Download quickshift++ from github
        2) unzip and move the folder to your project
        3) python3 setup.py build
        4) python3 setup.py install
        5) from QuickshiftPP import QuickshiftPP
    :param X_train:
    :param k:
        # k: number of neighbors in k-NN
        # beta: fluctuation parameter which ranges between 0 and 1.

    :return:
    """
    start = datetime.now()
    if k <= 0 or k > X.shape[0]:
        print(f'k {k} is not correct, so change it to X.shape[0]')
        k = X.shape[0]
    print(f"number of neighbors in k-NN: {k}")
    # Declare a Quickshift++ model with tuning hyperparameters.
    model = QuickshiftPP(k=k, beta=beta)

    # Note the try catch cannot capture the model.fit() error because it is cython. How to capture the exception?
    try:
        model.fit(X)
    except Exception as e:
        msg = f'quickshift++ fit error: {e}'
        raise ValueError(msg)

    end = datetime.now()
    quick_training_time = (end - start).total_seconds()
    # lg.info("quick_training_time took {} seconds".format(quick_training_time))

    # print('quickshift fit finished')
    labels_ = model.memberships
    n_clusters = len(set(labels_))
    cluster_centers = []
    for i in range(n_clusters):
        idxs = np.where(labels_ == i)[0]  # get index of each cluster. np.where return tuple
        if len(idxs) < thres_n:  # only keep the cluster which has more than "thres_n" datapoints
            continue
        center_cluster_i = np.mean(X[idxs], axis=0)  # here is X, not X_std.
        cluster_centers.append(center_cluster_i)

    means_init = np.asarray(cluster_centers, dtype=float)
    n_clusters = means_init.shape[0]
    print(f'--all clusters ({len(set(labels_))}) when (k:({k}), beta:{beta}). However, only {n_clusters} '
          f'clusters have at least {thres_n} datapoints. Counter(labels_): {Counter(labels_)}, *** '
          f'len(Counter(labels_)): {len(Counter(labels_))}')
    return means_init, len(set(labels_)), quick_training_time, n_clusters

