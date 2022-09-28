"""GMM

    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"
"""
import warnings

import numpy as np
from collections import Counter

from sklearn.cluster import MeanShift
from sklearn.exceptions import ConvergenceWarning
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
from sklearn.mixture.base import _check_X
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky, _estimate_gaussian_parameters
from sklearn.preprocessing import StandardScaler

# from loguru import logger as lg
from sklearn.utils import check_random_state


class ONLINE_GMM(GaussianMixture):

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

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def decision_function(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        return -1 * self.score_samples(X)

    # def predict_proba(self, X):
    #     return -1 * self.score_samples(X)    #

    # def fit_predict(self, X, y=None):
    #     """Estimate models parameters using X and predict the labels for X.
    #
    #     The method fits the models n_init times and sets the parameters with
    #     which the models has the largest likelihood or lower bound. Within each
    #     trial, the method iterates between E-step and M-step for `max_iter`
    #     times until the change of likelihood or lower bound is less than
    #     `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
    #     raised. After fitting, it predicts the most probable label for the
    #     input data points.
    #
    #     .. versionadded:: 0.20
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_features)
    #         List of n_features-dimensional data points. Each row
    #         corresponds to a single data point.
    #
    #     Returns
    #     -------
    #     labels : array, shape (n_samples,)
    #         Component labels.
    #     """
    #     # X = _check_X(X, self.n_components, ensure_min_samples=1)
    #     self._check_initial_parameters(X)
    #
    #     # if we enable warm_start, we will have a unique initialisation
    #     do_init = not(self.warm_start and hasattr(self, 'converged_'))
    #     n_init = self.n_init if do_init else 1
    #
    #     max_lower_bound = -np.infty
    #     self.converged_ = False
    #
    #     random_state = check_random_state(self.random_state)
    #
    #     n_samples, _ = X.shape
    #     for init in range(n_init):
    #         self._print_verbose_msg_init_beg(init)
    #
    #         if do_init:
    #             self._initialize_parameters(X, random_state)
    #
    #         lower_bound = (-np.infty if do_init else self.lower_bound_)
    #
    #         for n_iter in range(1, self.max_iter + 1):
    #             prev_lower_bound = lower_bound
    #
    #             log_prob_norm, log_resp = self._e_step(X)
    #             self._m_step(X, log_resp)
    #             lower_bound = self._compute_lower_bound(
    #                 log_resp, log_prob_norm)
    #
    #             change = lower_bound - prev_lower_bound
    #             self._print_verbose_msg_iter_end(n_iter, change)
    #
    #             if abs(change) < self.tol:
    #                 self.converged_ = True
    #                 break
    #
    #         self._print_verbose_msg_init_end(lower_bound)
    #
    #         if lower_bound > max_lower_bound:
    #             max_lower_bound = lower_bound
    #             best_params = self._get_parameters()
    #             best_n_iter = n_iter
    #
    #     if not self.converged_:
    #         warnings.warn('Initialization %d did not converge. '
    #                       'Try different init parameters, '
    #                       'or increase max_iter, tol '
    #                       'or check for degenerate data.'
    #                       % (init + 1), ConvergenceWarning)
    #
    #     self._set_parameters(best_params)
    #     self.n_iter_ = best_n_iter
    #     self.lower_bound_ = max_lower_bound
    #
    #     # Always do a final e-step to guarantee that the labels returned by
    #     # fit_predict(X) are always consistent with fit(X).predict(X)
    #     # for any value of max_iter and tol (and any random_state).
    #     _, log_resp = self._e_step(X)
    #
    #     return log_resp.argmax(axis=1)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        # resp = self.resp
        # nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        #
        # return nk/self.n_samples
        return np.log(self.weights_)


    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """

        n_samples, n_feats = X.shape
        n_components = self.n_components

        covariance_type = 'full'
        if covariance_type == 'full':
            log_prob = np.zeros((n_samples, n_components))
            log_det = np.zeros((n_samples, n_components))
            for k, (mu, sigma) in enumerate(zip(self.means_, self.covariances_)):
                diff = (X.T - mu[:, np.newaxis])  # X and mu should be column vectors
                log_prob[:, k] = np.diag(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff))
                log_det[:, k] = np.ones((n_samples,)) * np.log(np.linalg.det(sigma))

        return -.5 * (n_feats * np.log(2 * np.pi) + log_det) + log_prob

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = log_resp.shape
        # self.weights_, self.means_, self.covariances_ = (
        #     _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
        #                                   self.covariance_type))
        # self.weights_ /= n_samples

        resp = np.exp(log_resp)

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]

        n_components, n_features = means.shape
        reg_covar = 1e-6
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covariances[k].flat[::n_features + 1] += reg_covar      # x[startAt:endBefore:step], diagonal items

        self.weights_ = nk / n_samples
        self.means_ = means
        self.covariances_ = covariances



    def obtain_new_variance(self,x, means, covariances):

        min_dist = -1
        idx = 0
        for i, mu, sigma in enumerate(zip(means,covariances)):
            diff = x - mu
            dist = np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff)

            if dist > min_dist:
                min_dist = dist
                idx = i

        diff = x-means[idx]
        sigma = covariances[idx]
        sigma_1v = np.sqrt(np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff))

        if np.linalg.norm(diff) - sigma_1v  < 0:
            raise ValueError('cannot find a good sigma.')


        return sigma_1v



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
    # Declare a Quickshift++ models with tuning hyperparameters.
    model = QuickshiftPP(k=k, beta=beta)

    # Note the try catch cannot capture the models.fit() error because it is cython. How to capture the exception?
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
    # Declare a Quickshift++ models with tuning hyperparameters.
    model = QuickshiftPP(k=k, beta=beta)

    # Note the try catch cannot capture the models.fit() error because it is cython. How to capture the exception?
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
