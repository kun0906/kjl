"""GMM

    Required: quickshift++ (https://github.com/google/quickshift)
    "python3 setup build; python3 setup install" to install "quickshift++"

    # memory_profiler: for debugging memory leak
    python -m memory_profiler example.py

    from memory_profiler import profile
    @profile
    def func():
        pass

"""
from collections import Counter
from datetime import datetime

import numpy as np
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
from memory_profiler import profile
from sklearn.cluster import MeanShift
from sklearn.metrics import pairwise_distances


class SeekModes:
    def __init__(self, seek_name='quickshift', random_state=42):
        self.seek_name = seek_name
        self.random_state = random_state

    def fit(self, X, thres=0.95, n_comp_thres=20, **kwargs):

        # start = datetime.now()
        if self.seek_name.upper() == 'quickshift'.upper():
            """Initialize GMM
                1) Download quickshift++ from github
                2) unzip and move the folder to your project
                3) python3.7 setup.py build
                4) python3.7 setup.py install
                5) from QuickshiftPP import QuickshiftPP
            :param X_train:
            :param k:
                # k: number of neighbors in k-NN
                # beta: fluctuation parameter which ranges between 0 and 1.

                    :return:
                    
                QuickshiftPP doesn't have random_state parameter, so the result might change.
            """
            k = kwargs['qs_k']
            beta = kwargs['qs_beta']
            if k <= 0 or k > X.shape[0]:
                print(f'k {k} is not correct, so change it to X.shape[0]')
                k = X.shape[0]
            print(f"number of neighbors in k-NN: {k}")
            # Declare a Quickshift++ model with tuning hyperparameters.
            model = QuickshiftPP(k=k, beta=beta)

            # Note the try catch cannot capture the model.fit() error because it is cython. How to capture the exception?
            try:
                print(f'before qs: k={k}, beta={beta}')
                model.fit(X)
                print('after qs')
            except Exception as e:
                msg = f'quickshift++ fit error: {e}'
                raise ValueError(msg)

            # lg.info("quick_training_time took {} seconds".format(quick_training_time))

            # sort by cluster sizes
            tot_clusters = dict(sorted(Counter(model.memberships).items(), key=lambda kv: kv[1], reverse=True))
            tot_labels = model.memberships
            ########################################################################################
            # get gmm_init
            n_thres = 0
            N,D = X.shape
            i = 0
            for i, (label_, n_) in enumerate(tot_clusters.items()):
                n_thres += n_
                if n_thres > int(N*thres) or i >= n_comp_thres-1:
                    # n_thres -= n_
                    break
            self.n_clusters = i + 1
            self.n_thres =n_thres
            self.tot_clusters = tot_clusters
            self.tot_labels = tot_labels

        else:
            msg = f'Error: {self.seek_name}'
            raise NotImplementedError(msg)

        # end = datetime.now()
        # self.seek_train_time = (end - start).total_seconds()


class MODESEEKING():

    def __init__(self, method_name = 'meanshift', random_state=42, verbose = 1, **kwargs):
        self.method_name = method_name
        self.random_state=random_state
        self.verbose = verbose
        for k, v in kwargs.items():
            setattr(self, k, v )

    def fit(self, X,  n_thres = 100):

        if self.method_name == 'meanshift':
            if hasattr(self, 'bandwidth') and not self.bandwidth is None and self.bandwidth > 0:
                pass
            else:
                dists = pairwise_distances(X)
                self.bandwidth = np.quantile(dists, q=0.3)
            ms = MeanShift(self.bandwidth).fit(X)

            labels_all = ms.labels_
            n_clusters_all = len(set(labels_all))
            means_init_all = ms.cluster_centers_

        elif self.method_name == 'quickshift':
            if hasattr(self, 'k') and self.k > 0:
                pass
            else:
                k = 100
                beta = 0.9
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

            labels_all = model.memberships
            n_clusters_all = len(set(labels_all))
            # means_init_all = model.
        else:
            msg = self.method_name
            raise NotImplementedError(msg)

        labels_ = labels_all
        # find the good clusters with n_thres
        good_clusters_centers_mapping = {}
        bad_clusters_centers_mapping = {}
        for i in range(n_clusters_all):
            idxs = np.where(labels_ == i)[0]  # get index of each cluster. np.where return tuple
            if len(idxs) < n_thres:  # only keep the cluster which has more than "thres_n" datapoints
                # ignore these clusters and points
                bad_clusters_centers_mapping[i] = means_init_all[i]
            else:
                good_clusters_centers_mapping[i]= means_init_all[i]
        # merge the bad clusters into the good ones
        for i_cluster, center_i in bad_clusters_centers_mapping.items():
            # find the minimum distances between the bad cluster and all good clusters
            d_min = np.infty
            k_mim = 0
            for k, v in good_clusters_centers_mapping.items():
                d = np.sqrt(np.sum((center_i - v) ** 2, axis=1))
                if d  < d_min:
                    d_min = d
                    k_mim = k
            # only reassign the bad cluster's labels to the good cluster,
            # however, the good cluster's center doesn't update.
            labels_[labels_== i_cluster] = k_mim

        self.means_init_ = np.asarray([v for k, v in good_clusters_centers_mapping.items()], dtype=float)
        self.n_clusters_ = self.means_init_.shape[0]
        u, indices = np.unique(labels_, return_inverse=True)
        self.labels_ = indices # relabel the final clusters
        print(f'Meanshift gets ({n_clusters_all}) clusters with bandwidth({self.bandwidth}). '
              f'However, only {self.n_clusters_} clusters have more than {n_thres} datapoints')

@profile
def meanshift_seek_modes(X, bandwidth=None, thres_n=100):
    start = datetime.now()
    # clustering = MeanShift().fit(X)
    clustering = MeanShift(bandwidth=bandwidth).fit(X)
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
            # ignore these clusters and points
            continue
        # center_cluster_i = np.mean(X[idxs], axis=0)  # here is X, not X_std.
        center_cluster_i = all_means_init[i]  # here is X, not X_std.
        cluster_centers.append(center_cluster_i)

    means_init = np.asarray(cluster_centers, dtype=float)
    n_clusters = means_init.shape[0]
    print(f'--all clusters ({all_n_clusters}) when (bandwidth:{bandwidth}, {clustering.bandwidth}). However, only {n_clusters} '
          f'clusters have at least {thres_n} datapoints. Counter(labels_): {Counter(all_labels_)}, *** '
          f'len(Counter(labels_)): {all_n_clusters}')

    return means_init, n_clusters, meanshift_training_time, all_n_clusters



@profile
def quickshift_seek_modes(X, k=None, beta=0.9, thres_n=100):
    """Initialize GMM
            1) Download quickshift++ from github
            2) unzip and move the folder to your project
            3) python3.7 setup.py build
            4) python3.7 setup.py install
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
        print(f'before qs: k={k}, beta={beta}')
        model.fit(X)
        print('after qs')
    except Exception as e:
        msg = f'quickshift++ fit error: {e}'
        raise ValueError(msg)

    end = datetime.now()
    quick_training_time = (end - start).total_seconds()
    # lg.info("quick_training_time took {} seconds".format(quick_training_time))

    start = datetime.now()
    # print('quickshift fit finished')
    all_labels_ = model.memberships
    # # sort by cluster sizes
    # counter_labels = dict(sorted(Counter(all_labels_).items(), key=lambda kv: kv[1], reverse=True))
    #
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
    counter_labels = dict(sorted(Counter(all_labels_).items(), key=lambda kv: kv[1], reverse=True))
    print(f'*** quick_training_time: {quick_training_time}, ignore_clusters_time: {ignore_clusters_time}')
    print(f'--all clusters ({all_n_clusters}) when (k:({k}), beta:{beta}). However, only {n_clusters} '
          f'clusters have at least {thres_n} datapoints. Counter(labels_): {counter_labels}, *** '
          f'len(Counter(labels_)): {all_n_clusters}')
    return means_init, n_clusters, quick_training_time, counter_labels

#
# def get_means_init(X, k=None, beta=0.9, thres_n=100):
#     """Initialize GMM
#         1) Download quickshift++ from github
#         2) unzip and move the folder to your project
#         3) python3 setup.py build
#         4) python3 setup.py install
#         5) from QuickshiftPP import QuickshiftPP
#     :param X_train:
#     :param k:
#         # k: number of neighbors in k-NN
#         # beta: fluctuation parameter which ranges between 0 and 1.
#
#     :return:
#     """
#     start = datetime.now()
#     if k <= 0 or k > X.shape[0]:
#         print(f'k {k} is not correct, so change it to X.shape[0]')
#         k = X.shape[0]
#     print(f"number of neighbors in k-NN: {k}")
#     # Declare a Quickshift++ model with tuning hyperparameters.
#     model = QuickshiftPP(k=k, beta=beta)
#
#     # Note the try catch cannot capture the model.fit() error because it is cython. How to capture the exception?
#     try:
#         model.fit(X)
#     except Exception as e:
#         msg = f'quickshift++ fit error: {e}'
#         raise ValueError(msg)
#
#     end = datetime.now()
#     quick_training_time = (end - start).total_seconds()
#     # lg.info("quick_training_time took {} seconds".format(quick_training_time))
#
#     # print('quickshift fit finished')
#     labels_ = model.memberships
#     n_clusters = len(set(labels_))
#     cluster_centers = []
#     for i in range(n_clusters):
#         idxs = np.where(labels_ == i)[0]  # get index of each cluster. np.where return tuple
#         if len(idxs) < thres_n:  # only keep the cluster which has more than "thres_n" datapoints
#             continue
#         center_cluster_i = np.mean(X[idxs], axis=0)  # here is X, not X_std.
#         cluster_centers.append(center_cluster_i)
#
#     means_init = np.asarray(cluster_centers, dtype=float)
#     n_clusters = means_init.shape[0]
#     print(f'--all clusters ({len(set(labels_))}) when (k:({k}), beta:{beta}). However, only {n_clusters} '
#           f'clusters have at least {thres_n} datapoints. Counter(labels_): {Counter(labels_)}, *** '
#           f'len(Counter(labels_)): {len(Counter(labels_))}')
#     return means_init, len(set(labels_)), quick_training_time, n_clusters
