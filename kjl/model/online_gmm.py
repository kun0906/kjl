"""GMM

    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"
"""
import copy

import numpy as np
from collections import Counter

import sklearn
from scipy import linalg
from sklearn import cluster
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
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


class ONLINE_GMM(GaussianMixture):

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False, covariances_init=None,
                 verbose=0, verbose_interval=10, **kwargs):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start, precisions_init=precisions_init,
            verbose=verbose, verbose_interval=verbose_interval)

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.weights_ = weights_init
        self.means_ = means_init
        self.random_state = random_state

        # if covariance_type == 'diag' and len(covariances_init.shape) < 3:
        #     n_feats = covariances_init.shape[1]
        #     self.covariances_ = np.zeros((n_components, n_feats, n_feats))
        #     for i in range(n_components):
        #         np.fill_diagonal(self.covariances_[i], covariances_init[i])
        # else:
        #     self.covariances_ = covariances_init
        if covariance_type == 'diag':
            self.covariances_ = covariances_init    # (n_components, n_features)
        elif covariance_type =='full':
            self.covariances_ = covariances_init  # (n_components, n_features, n_features)

    def decision_function(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        return -1 * self.score_samples(X)

    def get_params(self, deep=True):
        return self.__dict__

    def _initialize1(self):

        # n_samples, _ = X.shape
        # weights, means, covariances = _estimate_gaussian_parameters(
        #     X, resp, self.reg_covar, self.covariance_type)
        # weights /= n_samples

        # self.weights_ = (weights if self.weights_init is None
        #                  else self.weights_init)
        # self.means_ = means if self.means_init is None else self.means_init


        if self.precisions_init is None:
            # self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                self.covariances_, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def _initialize_parameters(self, X, random_state, init = None):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, init=init, n_init=1,
                                           random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize(X, resp)

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
        # return np.mean(log_prob_norm), log_resp
        return log_prob_norm, log_resp

    # def _estimate_log_prob(self, X):
    #     """Estimate the log-probabilities log P(X | theta).
    #
    #     Compute the log-probabilities per each component for each sample.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_feats)
    #
    #     Returns
    #     -------
    #     log_prob : array, shape (n_samples, n_component)
    #     """
    #
    #
    #
    #     n_samples, n_feats = X.shape
    #     n_components = self.n_components
    #
    #     log_dist = np.zeros((n_samples, n_components))
    #     log_det = np.zeros((1, n_components))
    #     for k, (mean, covariance) in enumerate(zip(self.means_, self.covariances_)):
    #         diff = (X.T - mean[:, np.newaxis])  # X and mu should be column vectors
    #         if self.covariance_type == 'diag':
    #             _st = datetime.now()
    #             diagonal = covariance.flat[::n_feats + 1][:, np.newaxis]
    #             log_det[:, k] = np.log(np.prod(diagonal))
    #             # covariance.flat[::n_feats + 1]  = 1/ covariance.flat[::n_feats+1]
    #             # log_dist[:, k] = np.diag(-0.5 * np.matmul(np.matmul(diff.T, covariance), diff))   # way1 X@ diagonal @X
    #             # log_dist[:, k] = np.diag(-0.5 * (diff.T * 1/(diagonal.T)) @ diff) # way2: X/diagonal @ X
    #             log_dist[:, k] = -0.5 * ((diff.T ** 2) @ (1 / diagonal)).flatten()  # way 3: X**2 / diagonal
    #             # print(way1-way2, way1-log_dist[:, k]) # these are not exact same, due to the float precision.
    #             _end = datetime.now()
    #             # if self.verbose > 5: print(f'{k+1}th covariance time: {(_end-_st).total_seconds()}s')
    #         else:  # full
    #             log_dist[:, k] = np.diag(-0.5 * np.matmul(np.matmul(diff.T, np.linalg.inv(covariance)), diff))
    #             v = np.log(np.linalg.det(covariance))
    #             if np.isnan(v) or np.isinf(v):
    #                 print(
    #                     f'np.log(np.linalg.det(covariance)): {v}, {np.linalg.det(covariance)}, '
    #                     f'is inf or nan, so we use 1e-6 as the value.')
    #                 v = 1e-6
    #             log_det[:, k] = v
    #
    #     return -.5 * (n_feats * np.log(2 * np.pi) + log_det) + log_dist

    # def _m_step(self, X, resp):
    #     """M step.
    #
    #     Parameters
    #     ----------
    #     x : array-like, shape (n_samples, n_feats)
    #
    #     resp : array-like, shape (1, n_components)
    #         np.exp(log_resp), in which, log_resp is Logarithm of the posterior probabilities (or responsibilities) of
    #         the point of each sample in X.
    #
    #     sum_resp_pre : array-like, shape (1, n_components)
    #         np.sum(np.exp(log_resp_pre), axis=0)
    #     """
    #     # reg_covar = 1e-6
    #
    #     n_components, n_feats = self.means_.shape
    #     new_means = np.zeros((n_components, n_feats))
    #     new_covariances = np.zeros((n_components, n_feats, n_feats))
    #
    #     # element product
    #     nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # avoid 0
    #     new_means = (resp.T @ X) / nk[:, np.newaxis]
    #
    #     for k in range(n_components):
    #         # # element product
    #         _resp = (resp[:, k][:, np.newaxis])
    #         # nk = _resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # avoid 0
    #         # new_means[k] = (_resp * X).sum(axis=0) / nk
    #         new_covariances[k] = (np.matmul((_resp * (X - new_means[k])).T, X - new_means[k])) / nk[k]
    #         new_covariances[k].flat[::n_feats + 1] += self.reg_covar  # x[startAt:endBefore:step], diagonal items
    #         # print(f'k: {k}, {np.any(new_covariances[k] < 0)}')
    #
    #     self.means_ = new_means
    #     self.covariances_ = new_covariances
    #     self.weights_ = nk / X.shape[0]

    def _m_step_online(self, X, log_resp, sum_resp_pre=None, n_samples_pre=None):
        """M step.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_feats)

        log_resp : array-like, shape (1, n_components)
            np.exp(log_resp), in which, log_resp is Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        sum_resp_pre : array-like, shape (1, n_components)
            np.sum(np.exp(log_resp_pre), axis=0)
        """

        resp = np.exp(log_resp)

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
            # new_covariances1= covariances_k # for test
            # new_covariances1 = (sum_resp_pre_k * covariances_k + (resp_k * (X - new_means)).T @ (
            #             X - means_k)) / sum_weights.T
            # # assert np.all(new_covariances1 == new_covariances1.T)    # the error is very small due to the float precision
            # # print('1:', np.max(new_covariances- new_covariances1))

            if self.covariance_type == 'diag':  # (n_feats, )
                # new_covariances1 = (sum_resp_pre_k * covariances_k + np.diagonal((resp_k * (X - new_means)).T @ (
                #             X - means_k))) / sum_weights.T   # (n_feat, )
                new_covariances1 = (sum_resp_pre_k * covariances_k + np.diagonal((resp_k* X).T @ X))/sum_weights - new_means**2+ \
                                   (sum_resp_pre_k/sum_weights) * means_k**2   # (n_feat, )
                # new_covariances1 = (sum_resp_pre_k * covariances_k + np.sum(resp_k * X ** 2,
                #                                                             axis=0).T) / sum_weights - new_means ** 2  # (n_feat, )
                # if np.any(new_covariances1 <= 0):
                #     new_covariances1 = np.asarray([ self.reg_covar if v <=0 else v for v in new_covariances1])
                new_covariances1 = new_covariances1.flatten()
            elif self.covariance_type=='full':   # n_featsxn_feats
                new_covariances1 = (sum_resp_pre_k * covariances_k + (resp_k * X).T @ X) / sum_weights - \
                                   new_means[:, np.newaxis] @ new_means[:, np.newaxis].T
            # print('2:', np.max(new_covariances - new_covariances2))

            return new_means, new_covariances1

        n_components, n_feats = self.means_.shape
        new_means = np.zeros((n_components, n_feats))
        if self.covariance_type == 'diag':
            new_covariances = np.zeros((n_components, n_feats))
        elif self.covariance_type =='full':
            new_covariances = np.zeros((n_components, n_feats, n_feats))

        for k in range(n_components):
            new_means[k], new_covariances[k] = batch_mean_covariance(X, self.means_[k], self.covariances_[k],
                                                                     resp[:, k].reshape(-1, 1), sum_resp_pre[k])
            new_covariances[k].flat[::n_feats + 1] +=self.reg_covar # x[startAt:endBefore:step], diagonal items
            # print(f'k: {k}, {np.any(new_covariances < 0)}')
        self.means_ = new_means
        self.covariances_ = new_covariances
        # self.weights_ = self.weights_ + np.sum(resp-self.weights_, axis=0)/ (sum_resp_pre + np.sum(resp, axis=0)) # will generate negative value
        self.weights_ = self.weights_ + np.sum(resp - self.weights_, axis=0) / (n_samples_pre + X.shape[0])
        # print('m_step:', self.weights_)


    def add_new_component(self, x_proj, q_abnormal_thres=0.95, acculumated_X_train_proj=None):
        ##########################################################################################
        # sub-scenario:  create a new component for the new x
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
        if self.covariance_type == 'diag':
            self.covariances_ = np.concatenate([self.covariances_,
                                                new_covar.reshape(1, dim)], axis=0)
        else:
            self.covariances_ = np.concatenate([self.covariances_,
                                            new_covar.reshape(1, dim, dim)], axis=0)

        # print(f'new_model.params: {self.get_params()}')
        n = acculumated_X_train_proj.shape[0]
        self.weights_ = np.asarray([n / (n + 1) * v for v in self.weights_])
        self.weights_ = np.concatenate([self.weights_, np.ones((1,)) * (1 / (n + 1))], axis=0)

        self.sum_resp = np.concatenate([self.sum_resp, np.ones((1,))], axis=0)
        # f_(k+1): the new component of GMM
        n_feats = self.n_components
        # get log probabliity of acculumated data to update the threshold.
        diff = (acculumated_X_train_proj - new_mean).T  # X and mu should be column vectors
        if self.covariance_type =='diag':
            log_dist = -0.5 * diff.T**2 @ (1/new_covar)
            log_det = np.product(new_covar)
        else:
            log_dist = np.diag(-0.5 * np.matmul(np.matmul(diff.T, np.linalg.inv(new_covar)), diff))
            log_det = np.log(np.linalg.det(new_covar))
        log_det = 1e-6 if np.isnan(log_det) or np.isinf(log_det) else log_det
        f_k_1 = -.5 * (n_feats * np.log(2 * np.pi) + log_det) + log_dist


        # update self.self.precisions_cholesky_,
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

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
            if self.covariance_type == 'diag':
                dist =  diff.T**2 @ (1/sigma)
            elif self.covariance_type =='full':
                dist = np.matmul(np.matmul(diff.T, np.linalg.inv(sigma)), diff)
            dist = dist.flatten()
            if dist > min_dist:
                min_dist = dist
                idx = i

        diff = (x - means[idx]).T
        diff /= np.linalg.norm(diff)
        sigma = covariances[idx]

        if self.covariance_type =='diag':
            sigma_1v = np.sqrt(diff.T**2 @ (1/sigma)).item()
        else:
            sigma_1v = np.sqrt(np.matmul(np.matmul(diff.T, np.linalg.inv(sigma)), diff)).item()  # a scale

        if np.linalg.norm(diff) - sigma_1v < 0:
            try:
                raise ValueError('cannot find a good sigma.')
            except:
                sigma_1v = reg_covar

        n_feats = x.shape[-1]
        if self.covariance_type =='diag':
            new_covariance = np.asarray([sigma_1v] * n_feats)
        else:
            new_covariance = np.zeros((n_feats, n_feats))
            new_covariance.flat[::n_feats + 1] += sigma_1v  # x[startAt:endBefore:step], diagonal items

        return new_covariance
