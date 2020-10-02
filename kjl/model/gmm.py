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
from sklearn.preprocessing import StandardScaler

# from loguru import logger as lg
from kjl.utils.tool import execute_time



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

    @execute_time
    def decision_function(self, X):
        # it must be abnormal score because it will be used in grid search
        # we use y=1 as abnormal score, in grid search it will use y=1 as positive label,
        # so y_score also should be abnormal score
        return -1 * self.score_samples(X)

    def predict_proba(self, X):
        return -1 * self.score_samples(X)    #

    # def get_params(self, deep=True):
    #     return self.__dict__

    # def set_params(self, **params):
    #     super(GMM, self).set_params(**params)


