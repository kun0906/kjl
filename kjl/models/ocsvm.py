"""

"""
import numpy as np
from func_timeout import FunctionTimedOut
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances

from kjl.models._base import BASE
from kjl.projections._base import getGaussianGram


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


class _OCSVM(OneClassSVM):

    def __init__(self, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, random_state=100):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
            shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)

    # override decision_function. because test and grid_search will use decision_function first
    def decision_function_backup(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        # return -1 * self.score_samples(X)  # scores = sgn(models(x) - offset)
        # return -1 * (self._decision_function(X).ravel() + self.offset_)
        return -1 * (self._decision_function(X).ravel())

    # override decision_function. try to use numpy.
    def decision_function(self, X):
        # pred_v =  coefficient * kernel(X, support_vectors) + intercept

        # self.dual_coef_: m x 1 (m is the number of support vectors)
        # self.support_vectors : m x D
        # kernel(X, support_vectors.T): nxm
        # X: n x D
        # self.intercept: 1x1
        # pred_v = kernel(X, support_vectors.T)  * self.dual_coef_  + self.intercept : nx1
        if self.kernel == 'rbf':
            # Dist = pairwise_distances(X, Y=self.support_vectors_, metric='euclidean')
            # # K = np.exp(-np.power(Dist, 2) * 1 / self.sigma ** 2)
            # K = np.exp(-self.gamma * np.power(Dist, 2))
            K = getGaussianGram(X, self.support_vectors_, 1/ np.sqrt(self.gamma))   # nxm
            pred_v = np.matmul(K, self.dual_coef_.transpose()).ravel() + self.intercept_
        elif self.kernel == 'linear':
            K = np.matmul(X, self.support_vectors_.transpose())
            pred_v = np.matmul(K, self.dual_coef_.transpose()).ravel() + self.intercept_
        else:
            raise NotImplementedError(self.kernel)

        # print(pred_v - (self._decision_function(X).ravel()))
        # return -1 * (pred_v + self.offset_)
        if np.any(pred_v==np.inf ) or np.any(pred_v==-np.inf):
            pred_v[pred_v==-np.inf] = 0
            pred_v[pred_v ==np.inf] = 0

        return -1 * pred_v

    def predict_proba(self, X):
        return self.decision_function(X)



class OCSVM(BASE):

    def __init__(self, params, kernel = 'rbf', random_state=42):
        super(OCSVM, self).__init__()

        self.params = params
        self.kernel = kernel
        self.random_state = random_state

    def fit(self, X_train, y_train=None):
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

        ######################################################################################################
        # 1.2 OCSVM does not need to seek modes
        self.seek_train_time = 0

        ######################################################################################################
        # 1.3 Projection
        pr = cProfile.Profile(time.perf_counter)
        pr.enable()
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            # self.sigma = np.sqrt(X_train.shape[0]* X_train.var())
            self.project = KJL(self.params)
            self.project.fit(X_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['kjl_d']
            n = self.params['kjl_n']
            q = self.params['kjl_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
            # use 'linear' kernel for OCSVM when kjl is True
            self.params['OCSVM_kernel'] = 'linear'
            model_params = {'kernel': self.params['OCSVM_kernel'], 'nu': self.params['OCSVM_nu']}

        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            self.project = Nystrom(self.params)
            self.project.fit(X_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['nystrom_d']
            n = self.params['nystrom_n']
            q = self.params['nystrom_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
            self.params['OCSVM_kernel'] = 'linear'
            model_params = {'kernel': self.params['OCSVM_kernel'], 'nu': self.params['OCSVM_nu']}

        else:
            # when KJL=False or Nystrom=False, using rbf for OCSVM.
            sigma = np.quantile(pairwise_distances(X_train), self.params['OCSVM_q'])
            self.sigma = 1e-7 if sigma == 0 else sigma
            self.model_gamma = 1 / self.sigma ** 2
            q = self.params['OCSVM_q']
            lg.debug(f'model_sigma: {self.sigma}, model_gamma: {self.model_gamma}, q={q}')
            self.params['OCSVM_kernel'] = 'rbf'
            model_params = {'kernel': self.params['OCSVM_kernel'], 'gamma': self.model_gamma,
                            'nu': 0.5}

        pr.disable()
        ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        self.proj_train_time = ps.total_tt
        self.train_time += self.proj_train_time

        ######################################################################################################
        # 2.1 Initialize the models with preset parameters
        pr = cProfile.Profile(time.perf_counter)
        pr.enable()
        model = _OCSVM()
        # set models default parameters
        model.set_params(**model_params)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        self.init_model_time = ps.total_tt
        self.train_time += self.init_model_time
        lg.info(f'models.get_params(): {model.get_params()}')

        ######################################################################################################
        # 2.2 Build the models with train set
        try:
            self.model, self.model_train_time = self._train(model, X_train)
        except (FunctionTimedOut, Exception) as e:
            lg.warning(f'{e}, try a fixed number of iterations (here is 1000)')
            model.max_iter = 1000  #
            self.model, self.model_train_time = self._train(model, X_train)
        self.train_time += self.model_train_time

        ######################################################################################################
        # 3. Get space size based on support vectors
        pr = cProfile.Profile(time.perf_counter)
        pr.enable()
        n_sv = self.model.support_vectors_.shape[0]  # number of support vectors
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            self.space_size = n_sv + n_sv * d + n * (d + D)
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            self.space_size = n_sv + n_sv * d + n * (d + D)
        else:
            self.space_size = n_sv + n_sv * D
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('line')  # cumulative
        # ps.print_stats()
        self.space_train_time = ps.total_tt
        # self.train_time += self.space_train_time

        self.n_sv = n_sv
        self.D = D
        self.N = N

        lg.info(f'Train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
                f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
                f'init_model_time: {self.init_model_time}, model_train_time: {self.model_train_time}, '
                f'n_sv: {n_sv}, D:{D}, space_size: {self.space_size}, N:{N}, q: {q}')

        return self

    def eval(self, X_test, y_test, idx=None):
        return self._test(self.model, X_test, y_test, idx=idx)

