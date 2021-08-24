"""GMM
    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"
"""
import cProfile

import numpy as np
from func_timeout import FunctionTimedOut
from sklearn.mixture import GaussianMixture
import time
# load quickshift++
# using "pyximport.install()" fails for install quickshfit++ because it requires 'C++' in its setup.py.
# However, 1). pyximport does not use cythonize(). Thus it is not possible to do things like using compiler directives
# at the top of Cython files or compiling Cython code to C++.
# On the other hand, it is not recommended to let pyximport build code on end user side as it hooks into
# their import system.
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# *** Base on that, just use the following command to install quickshift++
# "python3 setup.py build; python3 setup.py install" to install "quickshift++"
from kjl.mode_seeking.seek_mode import SeekModes
from kjl.models._base import BASE
import cProfile

import numpy as np
from loguru import logger as lg
from kjl.utils import pstats

import copy
import itertools
import os.path as pth
import time
import traceback
from collections import Counter
import cProfile

import numpy as np
import sklearn
# from func_timeout import func_set_timeout, FunctionTimedOut
from joblib import delayed, Parallel
from sklearn import metrics
from sklearn.metrics import pairwise_distances, roc_curve
from loguru import logger as lg

from kjl.projections.kjl import KJL
from kjl.projections.nystrom import Nystrom
from kjl.utils import pstats

FUNC_TIMEOUT = 3 * 60  # (if function takes more than 3 mins, then it will be killed)
np.set_printoptions(precision=2, suppress=True)

from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_curve

from kjl.utils.tool import dump


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



class GMM_MAIN(BASE):

    def __init__(self, params):
        super(GMM_MAIN, self).__init__()

        self.params = params
        self.random_state = params['random_state']

    def train(self, X_train, y_train=None):
        """

        Parameters
        ----------
        X_train
        y_train

        Returns
        -------

        """

        self.train_time = 0
        N, D = X_train.shape

        #####################################################################################################
        # 1.1 normalization
        # pr = cProfile.Profile(time.perf_counter)
        # pr.enable()
        # # if self.params['is_std']:
        # #     self.scaler = StandardScaler(with_mean=self.params['is_std_mean'])
        # #     self.scaler.fit(X_train)
        # #     X_train = self.scaler.transform(X_train)
        # #     # if self.verbose > 10: data_info(X_train, name='X_train')
        # # else:
        # #     pass
        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        # self.std_train_time = ps.total_tt
        self.std_train_time = 0
        self.train_time += self.std_train_time

        #####################################################################################################
        # 1.2. projection
        pr = cProfile.Profile(time.perf_counter)
        pr.enable()
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            self.project = KJL(self.params)
            self.project.fit(X_train, y_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['kjl_d']
            n = self.params['kjl_n']
            q = self.params['kjl_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            self.project = Nystrom(self.params)
            self.project.fit(X_train, y_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['nystrom_d']
            n = self.params['nystrom_n']
            q = self.params['nystrom_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
        else:
            d = D
            n = N
            q = 0.25
        # self.params['is_kjl'] = False # for debugging
        # d, n, q = D, N, self.params['kjl_q']
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        self.proj_train_time = ps.total_tt
        self.train_time += self.proj_train_time

        #####################################################################################################
        # 1.3 seek modes after projection
        pr = cProfile.Profile(time.perf_counter)
        pr.enable()
        if self.params['after_proj']:
            self.thres_n = 0.95  # used to filter clusters
            if 'is_quickshift' in self.params.keys() and self.params['is_quickshift']:
                self.seek_mode = SeekModes(seek_name='quickshift', random_state=self.random_state)
                self.seek_mode.fit(X_train, qs_k=self.params['quickshift_k'], qs_beta=self.params['quickshift_beta'],
                                   thres=self.thres_n, GMM_covariance_type=self.params['GMM_covariance_type'],
                                   n_comp_thres=20)
                self.n_components = self.seek_mode.n_clusters
                self.tot_clusters = self.seek_mode.tot_clusters
                self.params['qs_res'] = {'tot_clusters': self.tot_clusters,
                                         'n_clusters': self.n_components,
                                         'q_proj': q,
                                         'k_qs': self.params['quickshift_k'],
                                         'beta_qs': self.params['quickshift_beta']}
                self.params['GMM_n_components'] = self.n_components

            if 'is_meanshift' in self.params.keys() and self.params['is_meanshift']:
                # if hasattr(self, 'sigma'):
                #     pass
                # else:
                #     dists = pairwise_distances(X_train)
                #     self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
                # self.seek_mode = SeekModes(seek_name='meansshift', random_state=self.random_state)
                # self.means_init, self.n_components, self.clusters, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                #     X_train, bandwidth=self.sigma, thres_n=self.thres_n)
                # self.n_components = self.seek_mode.n_clusters
                # self.tot_clusters = self.seek_mode.tot_clusters
                # self.params['ms_res'] = {'tot_clusters': self.tot_clusters, 'n_clusters': self.n_components,
                #                          'q_proj': q}
                # self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                msg = 'meanshift'
                raise NotImplementedError(msg)

        else:
            pass
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        self.seek_train_time = ps.total_tt
        self.train_time += self.seek_train_time

        #####################################################################################################
        # 2.1 Initialize the models
        pr = cProfile.Profile(time.perf_counter)
        pr.enable()
        model = GMM()
        if self.params['GMM_is_init_all'] and (self.params['is_quickshift'] or self.params['is_meanshift']):
            # get the init params of GMM
            self.weights_init, self.means_init, self.precisions_init, _ = compute_gmm_init(
                n_components=self.params['GMM_n_components'],
                X=X_train, n_thres=self.seek_mode.n_thres,
                tot_clusters=self.tot_clusters,
                tot_labels=self.seek_mode.tot_labels,
                covariance_type=self.params['GMM_covariance_type'])

            model_params = {'n_components': self.params['GMM_n_components'],
                            'covariance_type': self.params['GMM_covariance_type'],
                            'weights_init': self.weights_init,
                            'means_init': self.means_init,
                            'precisions_init': self.precisions_init,
                            'random_state': self.random_state}
        else:
            model_params = {'n_components': self.params['GMM_n_components'],
                            'covariance_type': self.params['GMM_covariance_type'],
                            'means_init': None, 'random_state': self.random_state}
        # set models default parameters
        model.set_params(**model_params)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        self.init_model_time = ps.total_tt
        self.train_time += self.init_model_time
        lg.debug(f'models.get_params(): {model.get_params()}')

        #####################################################################################################
        # 2.2 Train the models
        try:
            self.model, self.model_train_time = self._train(model, X_train)
        except (FunctionTimedOut, Exception) as e:
            lg.warning(f'{e}, retrain with a larger reg_covar')
            model.reg_covar = 1e-5
            self.model, self.model_train_time = self._train(model, X_train)
        self.train_time += self.model_train_time

        #####################################################################################################
        # 3. get space size
        pr = cProfile.Profile(time.perf_counter)
        pr.enable()
        if self.model.covariance_type == 'full':
            # space_size = (d ** 2 + d) * n_comps + n * (d + D)
            self.space_size = (d ** 2 + d) * self.model.n_components + n * (d + D)
        elif self.model.covariance_type == 'diag':
            # space_size = (2* d) * n_comps + n * (d + D)
            self.space_size = (2 * d) * self.model.n_components + n * (d + D)
        else:
            msg = self.model.covariance_type
            raise NotImplementedError(msg)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        self.space_train_time = ps.total_tt
        # self.train_time += self.space_train_time

        self.N = N
        self.D = D

        lg.info(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
                f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
                f'init_model_time: {self.init_model_time}, model_train_time: {self.model_train_time}, '
                f'D:{D}, space_size: {self.space_size}, N:{N}, n_comp: {self.model.n_components}, d: {d}, n: {n}, '
                f'q: {q}')

        return self

    def test(self, X_test, y_test, idx=None):
        return self._test(self.model, X_test, y_test, idx=idx)

