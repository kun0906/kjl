"""GMM

    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"
"""
import copy

import numpy as np
from collections import Counter

import sklearn
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

        if covariance_type == 'diag' and len(covariances_init.shape) < 3:
            n_feats = covariances_init.shape[1]
            self.covariances_ = np.zeros((n_components, n_feats, n_feats))
            for i in range(n_components):
                np.fill_diagonal(self.covariances_[i], covariances_init[i])
        else:
            self.covariances_ = covariances_init
        # self.covariances_init = covariances_init

    def decision_function(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        return -1 * self.score_samples(X)

    # def _estimate_log_weights(self):
    #     print(np.log(self.weights_))
    #     v = np.log(self.weights_)
    #     v[np.isnan(v)] = 0
    #     return v

    def _e_step_online(self, X):
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
        # return np.mean(log_prob_norm), log_resp
        return log_prob_norm, log_resp

    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | theta).

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

        log_dist = np.zeros((n_samples, n_components))
        log_det = np.zeros((1, n_components))
        for k, (mean, covariance) in enumerate(zip(self.means_, self.covariances_)):
            diff = (X.T - mean[:, np.newaxis])  # X and mu should be column vectors
            if self.covariance_type == 'diag':
                _st = datetime.now()
                diagonal = covariance.flat[::n_feats + 1][:, np.newaxis]
                log_det[:, k] = np.log(np.prod(diagonal))
                # covariance.flat[::n_feats + 1]  = 1/ covariance.flat[::n_feats+1]
                # way1 = np.diag(-0.5 * np.matmul(np.matmul(diff.T, covariance), diff))   # way1 X@ diagonal @X
                # way2 = np.diag(-0.5 * (diff.T * 1/(diagonal.T)) @ diff) # way2: X/diagonal @ X
                log_dist[:, k] = -0.5 * ((diff.T ** 2) @ (1 / diagonal)).flatten()  # way 3: X**2 / diagonal
                # print(way1-way2, way1-log_dist[:, k]) # these are not exact same, due to the float precision.
                _end = datetime.now()
                # if self.verbose > 5: print(f'{k+1}th covariance time: {(_end-_st).total_seconds()}s')
            else:  # full
                log_dist[:, k] = np.diag(-0.5 * np.matmul(np.matmul(diff.T, np.linalg.inv(covariance)), diff))
                v = np.log(np.linalg.det(covariance))
                if np.isnan(v) or np.isinf(v):
                    print(
                        f'np.log(np.linalg.det(covariance)): {v}, {np.linalg.det(covariance)},  is inf or nan, so we use 1e-6 as the value.')
                    v = 1e-6
                log_det[:, k] = v

        # elif covariance_type == 'diag':
        # precisions = precisions_chol ** 2
        # log_prob = (np.sum((means ** 2 * precisions), 1) -
        #             2. * np.dot(X, (means * precisions).T) +
        #             np.dot(X ** 2, precisions.T))

        return -.5 * (n_feats * np.log(2 * np.pi) + log_det) + log_dist

    def _m_step(self, X, resp):
        """M step.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_feats)

        resp : array-like, shape (1, n_components)
            np.exp(log_resp), in which, log_resp is Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        sum_resp_pre : array-like, shape (1, n_components)
            np.sum(np.exp(log_resp_pre), axis=0)
        """
        # reg_covar = 1e-6

        n_components, n_feats = self.means_.shape
        new_means = np.zeros((n_components, n_feats))
        new_covariances = np.zeros((n_components, n_feats, n_feats))

        # element product
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # avoid 0
        new_means = (resp.T @ X) / nk[:, np.newaxis]

        for k in range(n_components):
            # # element product
            _resp = (resp[:, k][:, np.newaxis])
            # nk = _resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # avoid 0
            # new_means[k] = (_resp * X).sum(axis=0) / nk
            new_covariances[k] = (np.matmul((_resp * (X - new_means[k])).T, X - new_means[k])) / nk[k]
            new_covariances[k].flat[::n_feats + 1] += self.reg_covar  # x[startAt:endBefore:step], diagonal items

        self.means_ = new_means
        self.covariances_ = new_covariances
        self.weights_ = nk / X.shape[0]

    def _m_step_online(self, X, resp, sum_resp_pre=None, n_samples_pre=None):
        """M step.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_feats)

        resp : array-like, shape (1, n_components)
            np.exp(log_resp), in which, log_resp is Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        sum_resp_pre : array-like, shape (1, n_components)
            np.sum(np.exp(log_resp_pre), axis=0)
        """

        def batch_mean_covariance(X, means_k, covariances_k, resp_k, sum_resp_pre_k):
            sum_weights = sum_resp_pre_k + np.sum(resp_k, axis=0).item()
            # print(sum_weights, sum_resp_pre_k, np.sum(resp_k, axis=0))
            # *: element-wise product
            new_means = means_k + (np.dot(resp_k.T, (X - means_k)) / sum_weights).flatten()

            # # ## Cn = covariances_k* sum_weights
            # rows, cols = covariances_k.shape  # a "n_feats by n_feats" matrix: symmetric matrix
            # new_covariances = np.zeros((rows, cols))
            # for i in range(rows):
            #     for j in range(cols):
            #         if i <= j:
            #             # Cn
            #             # v = np.sum(resp_k * ((X[:, i] - new_means[i]) * (X[:, j] - means_k[j])).reshape(-1, 1),
            #             #            axis=0).item()
            #             v = (resp_k * (X[:, i] - new_means[i])[:, np.newaxis]).T @ (X[:, j] - means_k[j])[:, np.newaxis]
            #             v = v.item()
            #             new_covariances[i][j] = (covariances_k[i][j] * sum_resp_pre_k + v) / (sum_weights)
            #         else:  # j < i
            #             new_covariances[i][j] = new_covariances[j][i]
            #
            # # assert  np.all(new_covariances == new_covariances.T)
            # # element product
            new_covariances1 = (sum_resp_pre_k * covariances_k + (resp_k * (X - new_means)).T @ (
                        X - means_k)) / sum_weights.T
            # # assert np.all(new_covariances1 == new_covariances1.T)    # the error is very small due to the float precision
            # # print('1:', np.max(new_covariances- new_covariances1))
            # new_covariances2 = (sum_resp_pre_k * covariances_k + (resp_k* X).T @ X)/sum_weights - \
            #                   new_means[:, np.newaxis] @ new_means[:, np.newaxis].T
            #
            # print('2:', np.max(new_covariances - new_covariances2))

            return new_means, new_covariances1

        n_components, n_feats = self.means_.shape
        new_means = np.zeros((n_components, n_feats))
        new_covariances = np.zeros((n_components, n_feats, n_feats))

        for k in range(n_components):
            new_means[k], new_covariances[k] = batch_mean_covariance(X, self.means_[k], self.covariances_[k],
                                                                     resp[:, k].reshape(-1, 1), sum_resp_pre[k])
            new_covariances[k].flat[::n_feats + 1] += self.reg_covar  # x[startAt:endBefore:step], diagonal items

        self.means_ = new_means
        self.covariances_ = new_covariances
        # self.weights_ = self.weights_ + np.sum(resp-self.weights_, axis=0)/ (sum_resp_pre + np.sum(resp, axis=0)) # will generate negative value
        self.weights_ = self.weights_ + np.sum(resp - self.weights_, axis=0) / (n_samples_pre + X.shape[0])
        print('m_step:', self.weights_)

    #
    #
    # def online_means_covaricances(self, x, n_samples, means, covariances, log_resp, reg_covar=1e-6):
    #     """
    #     https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    #     https://stackoverflow.com/questions/1346824/is-there-any-way-to-find-arithmetic-mean-better-than-sum-n
    #
    #     Parameters
    #     ----------
    #     x : array-like, shape (1, n_feats)
    #
    #     n_samples: int
    #         the number of datapoints used to fitted  the model until now.
    #
    #     means: array with shape (n_components, n_feats)
    #
    #     covariances: array with shape (n_components, n_feats, n_feats)
    #
    #     log_resp: vector with shape (1, n_components):  the weights
    #
    #     reg_covar:
    #         To avoid that the new covariance matrix is invalid.
    #
    #     Returns
    #     -------
    #         new_means:
    #         new_covariances:
    #
    #     """
    #
    #     def _online_covaricane(x, one_new_mu, one_covariances, n_samples):
    #         """ get the updated covariance.
    #         https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    #
    #         Parameters
    #         ----------
    #         x : array-like, shape (1, n_feats)
    #
    #         one_new_mu: array-like, shape (n_feats, )
    #             the updated means of one component
    #
    #         one_covariances: array-like, shape (n_feats, n_feats)
    #             the covariances of one component
    #
    #         Returns
    #         -------
    #             new_covariance:
    #         """
    #
    #         x = x.flatten()
    #
    #         rows, cols = one_covariances.shape  # a "n_feats by n_feats" matrix: symmetric matrix
    #         for i in range(rows):
    #             for j in range(cols):
    #                 if i <= j:
    #                     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    #                     #  COV(X,Y) = (C_n)/(n), C_n = C_(n-1) + N/(N-1)(x_n-x_mean_(n))*(y_n-y_mean_(n))
    #                     #
    #                     one_covariances[i][j] = (one_covariances[i][j] * n_samples + (n_samples + 1) / (n_samples) *
    #                                              (x[i] - one_new_mu[i]) * (x[j] - one_new_mu[j])) / (n_samples + 1)
    #                 else:  # j < i
    #                     one_covariances[i][j] = one_covariances[j][i]
    #
    #         return one_covariances
    #
    #     n_components, n_feats = means.shape
    #     new_means = np.zeros((n_components, n_feats))
    #     new_covariances = np.zeros((n_components, n_feats, n_feats))
    #     resp = np.exp(log_resp).flatten()
    #     for k in range(n_components):
    #         new_means[k] = resp[k] * (means[k] + (x - means[k]) / (n_samples + 1))
    #         # here would be overflow in add
    #         new_covariances[k] = resp[k] * _online_covaricane(x, new_means[k], covariances[k], n_samples)
    #         new_covariances[k].flat[::n_feats + 1] += reg_covar  # x[startAt:endBefore:step], diagonal items
    #
    #     return new_means, new_covariances

    def add_new_component(self, x_proj, q_abnormal_thres=0.95, acculumated_X_train_proj=None):
        ##########################################################################################
        # sub-scenario 1.2:  create a new component for the new x
        # self.novelty_thres < _y_score < self.abnormal_thres
        # x is predicted as a novelty datapoint (but still is a normal datapoint), so we create a new
        # component and update GMM.

        self.n_components += 1
        # compute the mean and covariance of the new components
        # For the mean, we use the x value as the mean of the new component
        # (because the new component only has one point (i.e., x)), and append it to the previous means.
        new_mean = x_proj
        new_covar = self.generate_new_covariance(x_proj, self.means_, self.covariances_)
        self.means_ = np.concatenate([self.means_, new_mean], axis=0)
        _, dim = new_mean.shape
        self.covariances_ = np.concatenate([self.covariances_,
                                            new_covar.reshape(1, dim, dim)], axis=0)

        # print(f'new_model.params: {self.get_params()}')
        n = acculumated_X_train_proj.shape[0]
        self.weights_ = np.asarray([n / (n + 1) * v for v in self.weights_])
        self.weights_ = np.concatenate([self.weights_, np.ones((1,)) * (1 / (n + 1))], axis=0)
        #
        # self.sum_resp = np.asarray([n / (n + 1) * v for v in self.sum_resp])
        # self.sum_resp = np.concatenate([self.sum_resp, np.ones((1,)) * (1 / (n + 1))], axis=0)

        self.sum_resp = np.concatenate([self.sum_resp, np.ones((1,))], axis=0)
        self.n_samples += x_proj.shape[0]
        # f_(k+1): the new component of GMM
        n_feats = self.n_components
        # get log probabliity
        diff = (acculumated_X_train_proj - new_mean).T  # X and mu should be column vectors
        log_dist = np.diag(-0.5 * np.matmul(np.matmul(diff.T, np.linalg.inv(new_covar)), diff))
        log_det = np.log(np.linalg.det(new_covar))
        log_det = 1e-6 if np.isnan(log_det) or np.isinf(log_det) else log_det
        f_k_1 = -.5 * (n_feats * np.log(2 * np.pi) + log_det) + log_dist

        self.y_score = n / (n + 1) * self.decision_function(acculumated_X_train_proj) + 1 / (n + 1) * f_k_1
        self.abnormal_thres = np.quantile(self.y_score, q=q_abnormal_thres)  # abnormal threshold

    def generate_new_covariance(self, x, means, covariances, reg_covar=1e-6):
        """ get a initial covariance matrix of the new component.

        Parameters
        ----------
        x : array-like, shape (1, n_feats)

        means: array with shape (n_components, n_feats)

        covariances: array with shape (n_components, n_feats, n_feats)

        reg_covar:
            To avoid that the new covariance matrix is invalid.

        Returns
        -------
            new_covariance: a matrix with shape (1, n_feats, n_feats)
        """

        min_dist = -1
        idx = 0
        for i, (mu, sigma) in enumerate(zip(means, covariances)):
            diff = (x - mu).T  # column vector
            diff /= np.linalg.norm(diff)
            diff[np.isnan(diff)] = 0
            dist = np.matmul(np.matmul(diff.T, np.linalg.inv(sigma)), diff)

            if dist > min_dist:
                min_dist = dist
                idx = i

        diff = (x - means[idx]).T
        diff /= np.linalg.norm(diff)
        sigma = covariances[idx]
        sigma_1v = np.sqrt(np.matmul(np.matmul(diff.T, np.linalg.inv(sigma)), diff)).item()  # a scale

        if np.linalg.norm(diff) - sigma_1v < 0:
            try:
                raise ValueError('cannot find a good sigma.')
            except:
                sigma_1v = reg_covar

        n_feats = x.shape[-1]
        new_covariance = np.zeros((n_feats, n_feats))
        new_covariance.flat[::n_feats + 1] += sigma_1v  # x[startAt:endBefore:step], diagonal items

        return new_covariance



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
