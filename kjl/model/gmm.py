"""GMM

    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"
"""
import numpy as np
from collections import Counter

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
from kjl.utils.utils import execute_time


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

    # def predict_proba(self, X):
    #     return -1 * self.score_samples(X)    #


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
