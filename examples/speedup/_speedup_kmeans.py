"""Main entrance
    run under "examples"
    PYTHONPATH=../:./ python3.7 speedup/main_kjl_parallel.py > out/main_kjl_parallel.txt 2>&1 &
"""

import copy
import itertools
from collections import Counter
from datetime import datetime

import numpy as np
import sklearn
from func_timeout import func_set_timeout, FunctionTimedOut
from joblib import delayed, Parallel
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from kjl.model.gmm import GMM
from kjl.model.kjl import getGaussianGram, kernelJLInitialize
from kjl.model.nystrom import nystromInitialize
from kjl.model.ocsvm import OCSVM
from kjl.model.seek_mode import quickshift_seek_modes, meanshift_seek_modes
from kjl.utils.data import data_info, dump_data, split_train_val, split_left_test, seperate_normal_abnormal
from kjl.utils.tool import execute_time
from matplotlib import pyplot as plt, cm
# print('PYTHONPATH: ', os.environ['PYTHONPATH'])

FUNC_TIMEOUT = 30 * 60  # (if function takes more than 30mins, then it will be killed)


def plot_data(X, y, title='Data'):
    plt.close()
    plt.figure()
    y_unique = np.unique(y)
    colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        plt.scatter(this_X[:, 0], this_X[:, 1], s=50,
                    c=color[np.newaxis, :],
                    alpha=0.5, edgecolor='k',
                    label=f"Class {this_y} {this_X.shape}")
    plt.legend(loc="best")
    plt.title(title)
    plt.show()
    plt.close()

class BASE:

    def __init__(self, random_state = 42):
        self.random_state = random_state

    def project_kjl(self, X_train, X_test, kjl_params={}):
        debug = False
        if kjl_params['is_kjl']:
            d = kjl_params['kjl_d']
            n = kjl_params['kjl_n']
            q = kjl_params['kjl_q']

            start = datetime.now()
            n = n or max([200, int(np.floor(X_train.shape[0] / 100))])  # n_v: rows; m_v: cols. 200, 100?
            m = n
            if hasattr(self, 'sigma') and self.sigma:
                sigma = self.sigma
            else:
                # compute sigma
                dists = pairwise_distances(X_train)
                if debug:
                    # for debug
                    _qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                    _sigmas = np.quantile(dists, _qs)  # it will cost time
                    print(f'train set\' sigmas with qs: {list(zip(_sigmas, _qs))}')
                sigma = np.quantile(dists, q)
                if sigma == 0:
                    print(f'sigma:{sigma}, and use 1e-7 for the latter experiment.')
                    sigma = 1e-7
            print("sigma: {}".format(sigma))

            # project train data
            # if debug: data_info(X_train, name='before KJL, X_train')
            X_train, U, Xrow = kernelJLInitialize(X_train, sigma, d, m, n, centering=1,
                                                  independent_row_col=0, random_state=self.random_state)
            if debug: data_info(X_train, name='after KJL, X_train')

            end = datetime.now()
            kjl_train_time = (end - start).total_seconds()
            print("kjl on train set took {} seconds".format(kjl_train_time))

            # # # for debug
            # if debug:
            #     data_info(X_test, name='X_test_std')
            #     _qs = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
            #     _sigmas = np.quantile(pairwise_distances(X_test), _qs)
            #     print(f'test set\' sigmas with qs: {list(zip(_sigmas, _qs))}')

            start = datetime.now()
            print("Projecting test data")
            K = getGaussianGram(X_test, Xrow, sigma)  # get kernel matrix using rbf
            X_test = np.matmul(K, U)
            # print(K, K.shape, U, U.shape, X_test)
            if debug: data_info(X_test, name='after KJL, X_test')
            end = datetime.now()
            kjl_test_time = (end - start).total_seconds()
            print("kjl on test set took {} seconds".format(kjl_test_time))
        else:
            kjl_train_time = 0
            kjl_test_time = 0

        self.kjl_train_time = kjl_train_time
        self.kjl_test_time = kjl_test_time
        self.sigma = sigma
        return X_train, X_test

    def project_nystrom(self, X_train, X_test, nystrom_params={}):
        debug = False
        if nystrom_params['is_nystrom']:
            d = nystrom_params['nystrom_d']
            n = nystrom_params['nystrom_n']
            q = nystrom_params['nystrom_q']

            start = datetime.now()
            n = n or max([200, int(np.floor(X_train.shape[0] / 100))])  # n_v: rows; m_v: cols. 200, 100?
            m = n
            if hasattr(self, 'sigma') and self.sigma:
                sigma = self.sigma
            else:
                # compute sigma
                dists = pairwise_distances(X_train)
                if debug:
                    # for debug
                    _qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                    _sigmas = np.quantile(dists, _qs)  # it will cost time
                    print(f'train set\' sigmas with qs: {list(zip(_sigmas, _qs))}')
                sigma = np.quantile(dists, q)
                if sigma == 0:
                    print(f'sigma:{sigma}, and use 1e-7 for the latter experiment.')
                    sigma = 1e-7
            print("sigma: {}".format(sigma))

            # project train data
            Eigvec, Lambda, subX = nystromInitialize(X_train, sigma, n, d, random_state=self.random_state)
            # Phix = nystromFeatures(X_train, subX, sigma, Eigvec, Lambda)
            X_train = np.matmul(np.matmul(getGaussianGram(X_train, subX, sigma), Eigvec),
                                np.diag(1. / np.sqrt(np.diag(Lambda))))

            if debug: data_info(X_train, name='after nystrom, X_train')
            eigvec_lambda = np.matmul(Eigvec, np.diag(1. / np.sqrt(np.diag(Lambda))))

            end = datetime.now()
            nystrom_train_time = (end - start).total_seconds()
            print("nystrom on train set took {} seconds".format(nystrom_train_time))

            # # for debug
            if debug:
                data_info(X_test, name='X_test_std')
                _qs = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
                _sigmas = np.quantile(pairwise_distances(X_test), _qs)
                print(f'test set\' sigmas with qs: {list(zip(_sigmas, _qs))}')

            start = datetime.now()
            print("Projecting test data")
            K = getGaussianGram(X_test, subX, sigma)  # get kernel matrix using rbf
            # X_test = np.matmul(np.matmul(K, Eigvec), np.diag(1. / np.sqrt(np.diag(Lambda))))    # Lambda is dxd, np.diag(Lambda) = 1xd
            X_test = np.matmul(K, eigvec_lambda)
            if debug: data_info(X_test, name='after nystrom, X_test')
            end = datetime.now()
            nystrom_test_time = (end - start).total_seconds()
            print("nystrom on test set took {} seconds".format(nystrom_test_time))
        else:
            nystrom_train_time = 0
            nystrom_test_time = 0

        self.nystrom_train_time = nystrom_train_time
        self.nystrom_test_time = nystrom_test_time
        self.sigma = sigma
        return X_train, X_test

    def standardize(self, X_train, X_test):

        scaler = StandardScaler(with_mean=self.params['is_std_mean'])

        start = datetime.now()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        end = datetime.now()
        self.std_train_time = (end - start).total_seconds()

        start = datetime.now()
        X_test = scaler.transform(X_test)
        end = datetime.now()
        self.std_test_time = (end - start).total_seconds()

        return X_train, X_test

    # # use_signals=False: the fuction cannot return a object that cannot be pickled (here "self" is not pickled,
    # # so it will be PicklingError)
    # # use_signals=True: it only works on main thread (here train_test_intf is not the main thread)
    # @timeout_decorator.timeout(seconds=20 * 60, use_signals=False, timeout_exception=StopIteration)
    # @profile
    @func_set_timeout(FUNC_TIMEOUT)  # seconds
    def _train(self, model, X_train, y_train=None):
        """Train model on the (X_train, y_train)

        Parameters
        ----------
        model
        X_train
        y_train

        Returns
        -------

        """
        start = datetime.now()
        try:
            model.fit(X_train)
        except (TimeoutError, Exception) as e:
            msg = f'fit error: {e}'
            print(msg)
            raise ValueError(f'{msg}: {model.get_params()}')
        end = datetime.now()
        train_time = (end - start).total_seconds()
        print("Fitting model takes {} seconds".format(train_time))

        return model, train_time

    def _test(self, model, X_test, y_test):
        """Evaulate the model on the X_test, y_test

        Parameters
        ----------
        model
        X_test
        y_test

        Returns
        -------
           y_score: abnormal score
           testing_time, auc, apc
        """
        start = datetime.now()
        # For inlier, a small value is used; a larger value is for outlier (positive)
        # it must be abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
        y_score = model.decision_function(X_test)

        """
        if model_name == "Gaussian" and n_components != 1:
            preds = model.predict_proba(X_test)
            pred = 1 - np.prod(1-preds, axis=1)
        else:
            pred = model.score_samples(X_test)
        """
        end = datetime.now()
        testing_time = (end - start).total_seconds()
        print("Test model takes {} seconds".format(testing_time))

        apc = average_precision_score(y_test, y_score, pos_label=1)
        # For binary  y_true, y_score is supposed to be the score of the class with greater label.
        # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
        # pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # auc1 = roc_auc_score(y_test, y_score)
        # print(model.get_params())
        # assert auc==auc1

        # f1, bestEp = selectThreshHold(test_y_i, pred)

        # if auc > max_auc:
        #     max_auc = auc
        #     best_pred = y_score

        print("APC: {}".format(apc))
        print("AUC: {}".format(auc))
        # print("F1: {}".format(f1))

        return y_score, testing_time, auc

    def save(self, data, out_file='.dat'):
        dump_data(data, name=out_file)

class KMeans_MAIN(BASE):

    def __init__(self, params):
        self.params = params
        self.random_state = params['random_state']

    def train_test_intf(self, X_train, y_train, X_test, y_test):
        """ X_test y_test required for gridsearch

        Parameters
        ----------
        X_train
        y_train
        X_test
        y_test

        Returns
        -------

        """

        self.train_time = 0
        self.test_time = 0

        N, D = X_train.shape
        X_train, y_train = copy.deepcopy(X_test), copy.deepcopy(y_test)
        X_test_raw, y_test_raw = copy.deepcopy(X_test), copy.deepcopy(y_test)
        self.params['is_std'] = True
        if self.params['is_std']:
            # should do standardization before using pairwise_distances
            X_train, X_test = self.standardize(X_train, X_test)
            X_test_raw, y_test_raw = copy.deepcopy(X_test), copy.deepcopy(y_test)
            data_info(X_train, name='X_train')
        else:
            self.std_train_time = 0
            self.std_test_time = 0
        self.train_time += self.std_train_time
        self.test_time += self.std_test_time

        if self.params['before_proj']:
            self.thres_n = 0  # used to filter clusters which have less than 10 datapoints
            if 'is_meanshift' in self.params.keys() and self.params['is_meanshift']:
                dists = pairwise_distances(X_train)
                self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                    X_train, bandwidth=self.sigma, thres_n=self.thres_n)
                self.params['ms_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': self.n_components}
                self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            elif 'is_quickshift' in self.params.keys() and self.params['is_quickshift']:
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                    X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                    thres_n=self.thres_n)
                if 'kjl_q' in self.params['kjl_q']:
                    q = self.params['kjl_q']
                elif 'nystrom_q' in self.params['nystrom_q']:
                    q = self.params['nystrom_q']
                else :
                    q = -1
                self.params['qs_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': self.n_components,
                                         'q_proj': q, 'k_qs': self.params['quickshift_k'],
                                         'beta_qs': self.params['quickshift_beta']}
                # only choose the top 20 cluster.
                self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            else:
                self.seek_train_time = 0
                self.seek_test_time = 0
            self.train_time += self.seek_train_time
            self.test_time += self.seek_test_time  # self.seek_test_time = 0

        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            # self.sigma = np.sqrt(X_train.shape[0]* X_train.var())
            X_train, X_test = self.project_kjl(X_train, X_test, kjl_params=self.params)
            self.proj_train_time = self.kjl_train_time
            self.proj_test_time = self.kjl_test_time
            d = self.params['kjl_d']
            n = self.params['kjl_n']
            q = self.params['kjl_q']
            print(f'self.sigma: {self.sigma}, q={q}')
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            X_train, X_test = self.project_nystrom(X_train, X_test, nystrom_params=self.params)
            self.proj_train_time = self.nystrom_train_time
            self.proj_test_time = self.nystrom_test_time
            d = self.params['nystrom_d']
            n = self.params['nystrom_n']
            q = self.params['nystrom_q']
            print(f'self.sigma: {self.sigma}, q={q}')
        else:
            d = D
            n = N
            self.proj_train_time = 0
            self.proj_test_time = 0
        self.train_time += self.proj_train_time
        self.test_time += self.proj_test_time

        if self.params['after_proj']:
            self.thres_n = 0  # used to filter clusters which have less than 10 datapoints
            if 'is_meanshift' in self.params.keys() and self.params['is_meanshift']:
                dists = pairwise_distances(X_train)
                self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                    X_train, bandwidth=self.sigma, thres_n=self.thres_n)
                self.params['ms_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': self.n_components,
                                         'q_proj': q}
                self.params['GMM_n_components'] =  20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            elif 'is_quickshift' in self.params.keys() and self.params['is_quickshift']:
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                    X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                    thres_n=self.thres_n)
                self.params['qs_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': 20 if self.n_components > 20 else self.n_components,
                                         'q_proj':q,
                                         'k_qs': self.params['quickshift_k'],
                                         'beta_qs': self.params['quickshift_beta']}
                self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            else:
                self.seek_train_time = 0
                self.seek_test_time = 0
            self.train_time += self.seek_train_time
            self.test_time += self.seek_test_time  # self.seek_test_time = 0

        self.params['GMM_n_components'] = 2
        model = KMeans()
        model_params = {'n_clusters': self.params['GMM_n_components'],
                         'random_state': self.random_state}
        # set model default parameters
        model.set_params(**model_params)
        print(f'model.get_params(): {model.get_params()}')
        # train model
        try:
            self.model, self.model_train_time = self._train(model, X_train)
        except:
            model.reg_covar = 1e-5
            print(f'retrain with a larger reg_covar')
            self.model, self.model_train_time = self._train(model, X_train)

        # if self.model.covariance_type == 'full':
        #     # space_size = (d ** 2 + d) * n_comps + n * (d + D)
        #     space_size = (d ** 2 + d) * self.model.n_components + n * (d + D)
        # elif self.model.covariance_type == 'diag':
        #     # space_size = (2* d) * n_comps + n * (d + D)
        #     space_size = (2 * d) * self.model.n_components + n * (d + D)
        # else:
        #     msg = self.model.covariance_type
        #     raise NotImplementedError(msg)
        space_size = 0
        self.train_time += self.model_train_time

        data_name = 'test'
        init_set = ''

        # raw dataset
        plot_data(X_test_raw, y_test_raw, title=f'raw on {data_name}, {init_set}')
        # KJL
        plot_data(X_test, y_test, title=f'kjl2 on {data_name}, {init_set}')
        # PCA
        X_embedded = PCA(n_components=2, random_state=100).fit_transform(X_test_raw)
        plot_data(X_embedded, y_test_raw, title=f'pca on {data_name}, {init_set}')

        # TSNE
        X_embedded = TSNE(n_components=2, random_state=100).fit_transform(X_test_raw)
        plot_data(X_embedded, y_test_raw, title=f'tsne on {data_name}, {init_set}')


        print(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
              f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
              f'model_train_time: {self.model_train_time}, D:{D}, space_size: {space_size}, N:{N}')
        self.space_size = space_size
        self.N = N
        self.D = D

        self.y_score, self.model_test_time, self.auc = self._test(self.model, X_test, y_test)
        self.test_time += self.model_test_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, seek_test_time: {self.seek_test_time}'
            f', proj_test_time: {self.proj_test_time}, '
            f'model_test_time: {self.model_test_time}')

        return self


class GMM_MAIN(BASE):

    def __init__(self, params):
        self.params = params
        self.random_state = params['random_state']

    def train_test_intf(self, X_train, y_train, X_test, y_test):
        """ X_test y_test required for gridsearch

        Parameters
        ----------
        X_train
        y_train
        X_test
        y_test

        Returns
        -------

        """
        self.train_time = 0
        self.test_time = 0
        N, D = X_train.shape

        if self.params['is_std']:
            # should do standardization before using pairwise_distances
            X_train, X_test = self.standardize(X_train, X_test)
            data_info(X_train, name='X_train')
        else:
            self.std_train_time = 0
            self.std_test_time = 0
        self.train_time += self.std_train_time
        self.test_time += self.std_test_time

        if self.params['before_proj']:
            self.thres_n = 0  # used to filter clusters which have less than 10 datapoints
            if 'is_meanshift' in self.params.keys() and self.params['is_meanshift']:
                dists = pairwise_distances(X_train)
                self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                    X_train, bandwidth=self.sigma, thres_n=self.thres_n)
                self.params['ms_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': self.n_components}
                self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            elif 'is_quickshift' in self.params.keys() and self.params['is_quickshift']:
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                    X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                    thres_n=self.thres_n)
                if 'kjl_q' in self.params['kjl_q']:
                    q = self.params['kjl_q']
                elif 'nystrom_q' in self.params['nystrom_q']:
                    q = self.params['nystrom_q']
                else :
                    q = -1
                self.params['qs_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': self.n_components,
                                         'q_proj': q, 'k_qs': self.params['quickshift_k'],
                                         'beta_qs': self.params['quickshift_beta']}
                # only choose the top 20 cluster.
                self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            else:
                self.seek_train_time = 0
                self.seek_test_time = 0
            self.train_time += self.seek_train_time
            self.test_time += self.seek_test_time  # self.seek_test_time = 0

        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            # self.sigma = np.sqrt(X_train.shape[0]* X_train.var())
            X_train, X_test = self.project_kjl(X_train, X_test, kjl_params=self.params)
            self.proj_train_time = self.kjl_train_time
            self.proj_test_time = self.kjl_test_time
            d = self.params['kjl_d']
            n = self.params['kjl_n']
            q = self.params['kjl_q']
            print(f'self.sigma: {self.sigma}, q={q}')
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            X_train, X_test = self.project_nystrom(X_train, X_test, nystrom_params=self.params)
            self.proj_train_time = self.nystrom_train_time
            self.proj_test_time = self.nystrom_test_time
            d = self.params['nystrom_d']
            n = self.params['nystrom_n']
            q = self.params['nystrom_q']
            print(f'self.sigma: {self.sigma}, q={q}')
        else:
            d = D
            n = N
            self.proj_train_time = 0
            self.proj_test_time = 0
        self.train_time += self.proj_train_time
        self.test_time += self.proj_test_time

        if self.params['after_proj']:
            self.thres_n = 0  # used to filter clusters which have less than 10 datapoints
            if 'is_meanshift' in self.params.keys() and self.params['is_meanshift']:
                dists = pairwise_distances(X_train)
                self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                    X_train, bandwidth=self.sigma, thres_n=self.thres_n)
                self.params['ms_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': self.n_components,
                                         'q_proj': q}
                self.params['GMM_n_components'] =  20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            elif 'is_quickshift' in self.params.keys() and self.params['is_quickshift']:
                self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                    X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                    thres_n=self.thres_n)
                self.params['qs_res'] = {'tot_n_clusters': self.all_n_clusters, 'n_clusters': 20 if self.n_components > 20 else self.n_components,
                                         'q_proj':q,
                                         'k_qs': self.params['quickshift_k'],
                                         'beta_qs': self.params['quickshift_beta']}
                self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                self.seek_test_time = 0
            else:
                self.seek_train_time = 0
                self.seek_test_time = 0
            self.train_time += self.seek_train_time
            self.test_time += self.seek_test_time  # self.seek_test_time = 0

        model = GMM()
        model_params = {'n_components': self.params['GMM_n_components'],
                        'covariance_type': self.params['GMM_covariance_type'],
                        'means_init': None, 'random_state': self.random_state}
        # set model default parameters
        model.set_params(**model_params)
        print(f'model.get_params(): {model.get_params()}')
        # train model
        try:
            self.model, self.model_train_time = self._train(model, X_train)
        except:
            model.reg_covar = 1e-5
            print(f'retrain with a larger reg_covar')
            self.model, self.model_train_time = self._train(model, X_train)

        if self.model.covariance_type == 'full':
            # space_size = (d ** 2 + d) * n_comps + n * (d + D)
            space_size = (d ** 2 + d) * self.model.n_components + n * (d + D)
        elif self.model.covariance_type == 'diag':
            # space_size = (2* d) * n_comps + n * (d + D)
            space_size = (2 * d) * self.model.n_components + n * (d + D)
        else:
            msg = self.model.covariance_type
            raise NotImplementedError(msg)
        self.train_time += self.model_train_time

        print(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
              f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
              f'model_train_time: {self.model_train_time}, D:{D}, space_size: {space_size}, N:{N}')
        self.space_size = space_size
        self.N = N
        self.D = D

        self.y_score, self.model_test_time, self.auc = self._test(self.model, X_test, y_test)
        self.test_time += self.model_test_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, seek_test_time: {self.seek_test_time}'
            f', proj_test_time: {self.proj_test_time}, '
            f'model_test_time: {self.model_test_time}')

        return self


class OCSVM_MAIN(BASE):

    def __init__(self, params):

        self.params = params
        self.random_state = params['random_state']

    def train_test_intf(self, X_train, y_train, X_test, y_test):
        """ X_test y_test required for gridsearch

        Parameters
        ----------
        X_train
        y_train
        X_test
        y_test

        Returns
        -------

        """
        self.train_time = 0
        self.test_time = 0
        N, D = X_train.shape

        if self.params['is_std']:
            # should do standardization before using pairwise_distances
            X_train, X_test = self.standardize(X_train, X_test)
        else:
            self.std_train_time = 0
            self.std_test_time = 0
        self.train_time += self.std_train_time
        self.test_time += self.std_test_time

        # OCSVM does not need seek mode
        # self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
        self.seek_train_time = 0
        self.seek_test_time = 0

        model = OCSVM()
        kjl = self.params['is_kjl']
        # kernel = 'linear'
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            # self.sigma = np.sqrt(X_train.shape[0]* X_train.var())
            X_train, X_test = self.project_kjl(X_train, X_test, kjl_params=self.params)
            self.proj_train_time = self.kjl_train_time
            self.proj_test_time = self.kjl_test_time
            d = self.params['kjl_d']
            n = self.params['kjl_n']
            q = self.params['kjl_q']
            print(f'self.sigma: {self.sigma}, q={q}')
            # # without normalization, svm takes infinite time. but why?
            # X_train, X_test = self.standardize(X_train, X_test)
            # self.train_time += self.std_train_time
            # self.test_time += self.std_test_time
            # if kjl=True, then use 'linear' kernel for OCSVM
            self.kernel = 'linear'
            model_params = {'kernel': self.kernel, 'nu': self.params['OCSVM_nu']}

            self.train_time += self.proj_train_time
            self.test_time += self.proj_test_time
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            X_train, X_test = self.project_nystrom(X_train, X_test, nystrom_params=self.params)
            self.proj_train_time = self.nystrom_train_time
            self.proj_test_time = self.nystrom_test_time
            d = self.params['nystrom_d']
            n = self.params['nystrom_n']
            q = self.params['nystrom_q']
            print(f'self.sigma: {self.sigma}, q={q}')
            # # without normalization, svm takes infinite time. but why?
            # X_train, X_test = self.standardize(X_train, X_test)
            # self.train_time += self.std_train_time
            # self.test_time += self.std_test_time
            # if kjl=True, then use 'linear' kernel for OCSVM
            self.kernel = 'linear'
            model_params = {'kernel': self.kernel, 'nu': self.params['OCSVM_nu']}

            self.train_time += self.proj_train_time
            self.test_time += self.proj_test_time
        # elif kernel=='linear':
        #     self.params['kernel'] = kernel
        #     sigma = np.quantile(pairwise_distances(X_train), self.params['OCSVM_q'])
        #     q=self.params['OCSVM_q']
        #     print(f'kernel: {kernel}, model_gamma: {self.model_gamma}, q={q}')
        #     K = getGaussianGram(X_train, X_train, sigma, goFast=0)
        #
        #     # model_params = {'kernel': self.params['kernel'], 'gamma': 'scale', 'nu': self.params['OCSVM_nu']} # default params by sklearn
        #     model_params = {'kernel': self.params['kernel'], 'gamma': self.model_gamma, 'nu': self.params['OCSVM_nu']}
        #     self.kjl_train_time = 0
        #     self.kjl_test_time = 0
        #     self.train_time += self.kjl_train_time
        #     self.test_time += self.kjl_test_time
        else:  # when KJL=False, we use rbf
            self.params['kernel'] = 'rbf'
            sigma = np.quantile(pairwise_distances(X_train), self.params['OCSVM_q'])
            sigma = 1e-6 if sigma == 0  else sigma
            self.model_gamma = 1 / sigma ** 2
            q = self.params['OCSVM_q']
            print(f'model_sigma: {sigma}, model_gamma: {self.model_gamma}, q={q}')
            # model_params = {'kernel': self.params['kernel'], 'gamma': 'scale', 'nu': self.params['OCSVM_nu']} # default params by sklearn
            model_params = {'kernel': self.params['kernel'], 'gamma': self.model_gamma, 'nu': self.params['OCSVM_nu']}
            self.proj_train_time = 0
            self.proj_test_time = 0
            self.train_time += self.proj_train_time
            self.test_time += self.proj_test_time

        # set model default parameters
        model.set_params(**model_params)
        print(f'model.get_params()L {model.get_params()}')
        # train model
        try:
            self.model, self.model_train_time = self._train(model, X_train)
        except Exception as e:
            raise TimeoutError(e)

        n_sv = self.model.support_vectors_.shape[0]  # number of support vectors
        if kjl:
            space_size = n_sv + n_sv * D + n * (d + D)
        else:
            space_size = n_sv + n_sv * D
        self.train_time += self.model_train_time
        # print(f'{self.model.get_params()}, {self.params}')

        print(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
              f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
              f'model_train_time: {self.model_train_time}, n_sv: {n_sv}, D:{D}, space_size: {space_size}, N:{N}')
        self.n_sv = n_sv
        self.D = D
        self.space_size = space_size
        self.N = N

        self.y_score, self.model_test_time, self.auc = self._test(self.model, X_test, y_test)
        self.test_time += self.model_test_time
        print(f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, seek_test_time: '
              f'{self.seek_test_time}, proj_test_time: {self.proj_test_time}, '
              f'model_test_time: {self.model_test_time}')
        print('tet')
        # return self


#
# def _model_train_test_backup(X, y, params, **kwargs):
#     """
#
#     Parameters
#     ----------
#     normal_data
#     abnormal_data
#     params
#     args
#
#     Returns
#     -------
#
#     """
#     try:
#         for k, v in kwargs.items():
#             params[k] = v
#
#         n_repeats = params['n_repeats']
#         train_times = []
#         test_times = []
#         aucs = []
#         # keep that all the algorithms have the same input data by setting random_state
#         for i in range(n_repeats):
#             print(f"\n\n==={i + 1}/{n_repeats}(n_repeats): {params}===")
#             X_train, y_train, X_val, y_val,  X_test, y_test = split_train_test(X, y,
#                                                                 train_size=5000, random_state=(i+1) * 100)
#
#             if "GMM" in params['detector_name']:
#                 model = GMM_MAIN(params)
#             elif "OCSVM" in params['detector_name']:
#                 model = OCSVM_MAIN(params)
#             if params['mode'] =='best':
#                 model.train_test_intf(X_train, y_train, X_val, y_val)
#             elif params['mode'] == 'default':
#                 model.train_test_intf(X_train, y_train, X_test, y_test)
#
#             train_times.append(model.train_time)
#             test_times.append(model.test_time)
#             aucs.append(model.auc)
#
#         info = {'train_times': train_times, 'test_times': test_times, 'aucs': aucs, 'apcs': '',
#                 'params': params,
#                 'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
#
#     except Exception as e:
#         traceback.print_exc()
#         info = {'train_times': [0.0], 'test_times': [0.0], 'aucs': [0.0], 'apcs': '',
#                 'params': params,
#                 'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
#     return info

#
# @execute_time
# def get_best_results_backup(X, y, params, random_state=42):
#     """
#
#     Parameters
#     ----------
#     normal_data
#     abnormal_data
#     case
#
#     Returns
#     -------
#
#     """
#     params_copy = params
#     params = copy.deepcopy(params_copy)
#     X_copy = X
#     X =  copy.deepcopy(X_copy)
#     y_copy = y
#     y = copy.deepcopy(y_copy)
#     random_state_copy = random_state
#     random_state = copy.deepcopy(random_state_copy)
#     params['random_state'] = random_state
#
#     parallel = Parallel(n_jobs=25, verbose=30)
#
#     if not params['is_gs']:
#         params['mode'] = 'default'  # evaluate on the test_set to find the best params
#     else:
#         params['mode'] = 'best'    # evaluate on the val_set to find the best params
#     # best params and defaults params use the same API
#     if params['detector_name'] == 'GMM':
#         # GMM with grid search
#         n_components_arr = params['GMM_n_components']
#         if 'GMM_n_components' in params.keys(): del params['GMM_n_components']
#         if not params['is_kjl'] and not params['is_nystrom']:  # only GMM
#             # GMM-gs:True
#             with parallel:
#                 outs = parallel(delayed(_model_train_test)(X, y, copy.deepcopy(params),
#                                                            GMM_n_components=n_components) for n_components, _ in
#                                 list(itertools.product(n_components_arr, [0])))
#
#         elif params['is_kjl']:
#             kjl_ds = params['kjl_ds']
#             kjl_ns = params['kjl_ns']
#             kjl_qs = params['kjl_qs']
#             if 'kjl_ds' in params.keys(): del params['kjl_ds']
#             if 'kjl_ns' in params.keys(): del params['kjl_ns']
#             if 'kjl_qs' in params.keys(): del params['kjl_qs']
#             if not params['is_quickshift'] and not params['is_meanshift']:
#                 # GMM-gs:True-kjl:True
#                 with parallel:
#                     outs = parallel(
#                         delayed(_model_train_test)(X, y, copy.deepcopy(params), kjl_d=kjl_d, kjl_n=kjl_n,
#                                                    kjl_q=kjl_q,
#                                                    GMM_n_components=n_components) for kjl_d, kjl_n, kjl_q, n_components
#                         in
#                         list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))
#
#             elif params['is_quickshift']:
#                 # GMM-gs:True-kjl:True-quickshift:True
#                 quickshift_ks = params['quickshift_ks']
#                 quickshift_betas = params['quickshift_betas']
#                 if 'quickshift_ks' in params.keys(): del params['quickshift_ks']
#                 if 'quickshift_betas' in params.keys(): del params['quickshift_betas']
#                 with parallel:
#                     outs = parallel(
#                         delayed(_model_train_test)(X, y, copy.deepcopy(params),
#                                                    kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
#                                                    quickshift_k=quickshift_k, quickshift_beta=quickshift_beta)
#                         for kjl_d, kjl_n, kjl_q, quickshift_k, quickshift_beta in
#                         list(itertools.product(kjl_ds, kjl_ns, kjl_qs, quickshift_ks, quickshift_betas)))
#
#             elif params['is_meanshift']:
#                 # GMM-gs:True-kjl:True-meanshift:True
#                 meanshift_qs = params[
#                     'meanshift_qs']  # meanshift uses the same kjl_qs, and only needs to tune one of them
#                 if 'meanshift_qs' in params.keys(): del params['meanshift_qs']
#                 with parallel:
#                     # outs = parallel(delayed(_model_train_test)(X, y, params, kjl_d=kjl_d,
#                     #                                            kjl_n=kjl_n, kjl_q=kjl_q, n_components=n_components)
#                     #                 for kjl_d, kjl_n, kjl_q, n_components in
#                     #                 list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))
#
#                     outs = parallel(delayed(_model_train_test)(X, y, copy.deepcopy(params), kjl_d=kjl_d,
#                                                                kjl_n=kjl_n, kjl_q=kjl_q)
#                                     for kjl_d, kjl_n, kjl_q in
#                                     list(itertools.product(kjl_ds, kjl_ns, kjl_qs)))
#             else:
#                 msg = params['is_kjl']
#                 raise NotImplementedError(f'Error: kjl={msg}')
#
#         elif params['is_nystrom']:
#             # GMM-gs:True-nystrom:True
#             nystrom_ns = params['nystrom_ns']
#             nystrom_ds = params['nystrom_ds']
#             nystrom_qs = params['nystrom_qs']
#             if 'nystrom_ns' in params.keys(): del params['nystrom_ns']
#             if 'nystrom_ds' in params.keys(): del params['nystrom_ds']
#             if 'nystrom_qs' in params.keys(): del params['nystrom_qs']
#             with parallel:
#                 outs = parallel(delayed(_model_train_test)(X, y, copy.deepcopy(params), nystrom_n=nystrom_n,
#                                                            nystrom_d=nystrom_d, nystrom_q=nystrom_q,
#                                                            GMM_n_components=n_components) for
#                                 nystrom_n, nystrom_d, nystrom_q, n_components in
#                                 list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, n_components_arr)))
#         else:
#             msg = params['is_kjl']
#             raise NotImplementedError(f'Error: kjl={msg}')
#
#     elif params['detector_name'] == 'OCSVM':
#         if not params['is_kjl']:
#             # msg = params['is_kjl']
#             # raise NotImplementedError(f'Error: kjl={msg}')
#             with parallel:
#                 model_qs = params['OCSVM_qs']
#                 model_nus = params['OCSVM_nus']
#                 if 'OCSVM_qs' in params.keys(): del params['OCSVM_qs']
#                 if 'OCSVM_nus' in params.keys(): del params['OCSVM_nus']
#
#                 outs = parallel(delayed(_model_train_test)(X, y, copy.deepcopy(params), OCSVM_q=OCSVM_q,
#                                                            OCSVM_nu=OCSVM_nu) for _, _, _, OCSVM_q, OCSVM_nu in
#                                 list(itertools.product([0], [0], [0], model_qs, model_nus)))
#         else:  # gs=True, kjl = True and for OCSVM ('linear')
#             with parallel:
#
#                 kjl_ds = params['kjl_ds']
#                 kjl_ns = params['kjl_ns']
#                 kjl_qs = params['kjl_qs']
#                 if 'kjl_ds' in params.keys(): del params['kjl_ds']
#                 if 'kjl_ns' in params.keys(): del params['kjl_ns']
#                 if 'kjl_qs' in params.keys(): del params['kjl_qs']
#                 model_nus = params['OCSVM_nus']
#                 if 'OCSVM_nus' in params.keys(): del params['OCSVM_nus']
#                 if not params['is_quickshift'] and not params['is_meanshift']:
#                     # GMM-gs:True-kjl:True
#                     with parallel:
#                         outs = parallel(
#                             delayed(_model_train_test)(X, y, copy.deepcopy(params), kjl_d=kjl_d, kjl_n=kjl_n,
#                                                        kjl_q=kjl_q,
#                                                        OCSVM_nu=OCSVM_nu) for
#                             kjl_d, kjl_n, kjl_q, OCSVM_nu
#                             in
#                             list(itertools.product(kjl_ds, kjl_ns, kjl_qs, model_nus)))
#
#
#     else:
#         msg = params['detector_name']
#         raise NotImplementedError(f'Error: detector_name={msg}')
#
#     # get the best avg auc from n_repeats experiments
#     best_avg_auc = -1
#     for out in outs:
#         if np.mean(out['aucs']) > best_avg_auc:
#             best_avg_auc = np.mean(out['aucs'])
#             best_results = copy.deepcopy(out)
#
#     # it's better to save all middle results too
#     middle_results = outs
#
#     print('---get accurate time of training and testing with the best params---')
#     best_params = best_results['params']
#     params['mode'] = 'default'  # evaluate on the test_set to get the final results
#     out = _model_train_test(X, y, params=best_params)
#     # double check the results if you can
#     # assert best_avg_auc == np.mean(out['aucs'])
#     # print(best_avg_auc, np.mean(out['aucs']), best_results, out)
#     best_results = out
#
#     return best_results, middle_results
#
# def _model_train_test(X_train, y_train, X_test, y_test, params, **kwargs):
#     """
#
#     Parameters
#     ----------
#     normal_data
#     abnormal_data
#     params
#     args
#
#     Returns
#     -------
#
#     """
#     # ##################### memory allocation snapshot
#     #
#     # tracemalloc.start()
#     #
#     # start_time = time.time()
#     # snapshot1 = tracemalloc.take_snapshot()
#
#     print(kwargs, params)
#     try:
#         for k, v in kwargs.items():
#             params[k] = v
#
#         if "GMM" in params['detector_name']:
#             model = GMM_MAIN(params)
#         elif "OCSVM" in params['detector_name']:
#             model = OCSVM_MAIN(params)
#         model.train_test_intf(X_train, y_train, X_test, y_test)
#         print('train_test')
#
#         info = {'train_time': model.train_time, 'test_time': model.test_time, 'auc': model.auc, 'apc': '',
#                 'params': model.params, 'space_size': model.space_size,
#                 'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
#     # except (timeout_decorator.TimeoutError, pickle.PickleError) as e:
#     #     info = {'train_time': 0.0, 'test_time': 0.0, 'auc': 0.0, 'apc': '',
#     #             'params': params, 'space_size': 0.0,
#     #             'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
#     except (FunctionTimedOut, Exception) as e:
#         print(f"FunctionTimedOut Error: {e}")
#         # traceback.print_exc()
#         info = {'train_time': 0.0, 'test_time': 0.0, 'auc': 0.0, 'apc': '',
#                 'params': params, 'space_size': 0.0,
#                 'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
#
#     # ################### Second memory allocation snapshot
#     #
#     # snapshot2 = tracemalloc.take_snapshot()
#     # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
#     #
#     # print("[ Top 10 ]")
#     # for stat in top_stats[:5]:
#     #     print(stat)
#
#     return info


def _model_train_test(X_train, y_train, X_test, y_test, params, **kwargs):
    """

    Parameters
    ----------
    normal_data
    abnormal_data
    params
    args

    Returns
    -------



# PCA
X_embedded = PCA(n_components=n_components, random_state=100).fit_transform(X)
plot_data(X_embedded, y, title=f'pca on {data_name}, {init_set}')

# TSNE
X_embedded = TSNE(n_components=n_components, random_state=100).fit_transform(X)
plot_data(X_embedded, y, title=f'tsne on {data_name}, {init_set}')

    """
    # ##################### memory allocation snapshot
    #
    # tracemalloc.start()
    #
    # start_time = time.time()
    # snapshot1 = tracemalloc.take_snapshot()

    print(kwargs, params)
    try:
        for k, v in kwargs.items():
            params[k] = v

        model = KMeans_MAIN(params)
        model.train_test_intf(X_train, y_train, X_test, y_test)
        print('train_test')

        info = {'train_time': model.train_time, 'test_time': model.test_time, 'auc': model.auc, 'apc': '',
                'params': model.params, 'space_size': model.space_size,
                'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
    # except (timeout_decorator.TimeoutError, pickle.PickleError) as e:
    #     info = {'train_time': 0.0, 'test_time': 0.0, 'auc': 0.0, 'apc': '',
    #             'params': params, 'space_size': 0.0,
    #             'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
    except (Exception) as e:
        print(f"FunctionTimedOut Error: {e}")
        # traceback.print_exc()
        info = {'train_time': 0.0, 'test_time': 0.0, 'auc': 0.0, 'apc': '',
                'params': params, 'space_size': 0.0,
                'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}

    # ################### Second memory allocation snapshot
    #
    # snapshot2 = tracemalloc.take_snapshot()
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    #
    # print("[ Top 10 ]")
    # for stat in top_stats[:5]:
    #     print(stat)

    return info

@execute_time
def get_best_results(X_train, y_train, X_val, y_val, X_test, y_test, params, random_state=42):
    """

    Parameters
    ----------
    normal_data
    abnormal_data
    case

    Returns
    -------

    """
    is_gs = params['is_gs']
    if is_gs:
        # pass
        print(f'--is_gs: {is_gs}, X_val != X_test')
        if len(y_val) < 10:
            print('just for avoiding error for gs=True')
            X_val, y_val = sklearn.utils.resample(X_test, y_test, n_samples=10, stratify=y_test)
    else:
        X_val = X_test
        y_val = y_test
        print(f'--is_gs: {is_gs}, X_val == X_test')
    print(f'X_train.shape: {X_train.shape}, y_train: {Counter(y_train)}')
    print(f'X_val.shape: {X_val.shape}, y_val: {Counter(y_val)}')
    print(f'X_test.shape: {X_test.shape}, y_test: {Counter(y_test)}')
    params_copy = params
    params = copy.deepcopy(params_copy)
    # X_copy = X
    # X =  copy.deepcopy(X_copy)
    # y_copy = y
    # y = copy.deepcopy(y_copy)
    random_state_copy = random_state
    random_state = copy.deepcopy(random_state_copy)
    params['random_state'] = random_state

    parallel = Parallel(n_jobs=params['n_jobs'], verbose=30)

    # if not params['is_gs']:
    #     params['mode'] = 'default'  # evaluate on the test_set to find the best params
    # else:
    #     params['mode'] = 'best'    # evaluate on the val_set to find the best params
    # best params and defaults params use the same API
    if params['detector_name'] == 'GMM':
        # GMM with grid search
        n_components_arr = params['GMM_n_components']
        if 'GMM_n_components' in params.keys(): del params['GMM_n_components']
        if not params['is_kjl'] and not params['is_nystrom']:  # only GMM
            # GMM-gs:True
            with parallel:
                outs = parallel(delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params),
                                                           GMM_n_components=n_components) for n_components, _ in
                                list(itertools.product(n_components_arr, [0])))

        elif params['is_kjl']:
            kjl_ds = params['kjl_ds']
            kjl_ns = params['kjl_ns']
            kjl_qs = params['kjl_qs']
            if 'kjl_ds' in params.keys(): del params['kjl_ds']
            if 'kjl_ns' in params.keys(): del params['kjl_ns']
            if 'kjl_qs' in params.keys(): del params['kjl_qs']
            if not params['is_quickshift'] and not params['is_meanshift']:
                # GMM-gs:True-kjl:True
                with parallel:
                    outs = parallel(
                        delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params), kjl_d=kjl_d,
                                                   kjl_n=kjl_n,
                                                   kjl_q=kjl_q,
                                                   GMM_n_components=n_components) for kjl_d, kjl_n, kjl_q, n_components
                        in
                        list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))

            elif params['is_quickshift']:
                # GMM-gs:True-kjl:True-quickshift:True
                quickshift_ks = params['quickshift_ks']
                quickshift_betas = params['quickshift_betas']
                if 'quickshift_ks' in params.keys(): del params['quickshift_ks']
                if 'quickshift_betas' in params.keys(): del params['quickshift_betas']
                with parallel:
                    outs = parallel(
                        delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params),
                                                   kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                                   quickshift_k=quickshift_k, quickshift_beta=quickshift_beta)
                        for kjl_d, kjl_n, kjl_q, quickshift_k, quickshift_beta in
                        list(itertools.product(kjl_ds, kjl_ns, kjl_qs, quickshift_ks, quickshift_betas)))

            elif params['is_meanshift']:
                # GMM-gs:True-kjl:True-meanshift:True
                meanshift_qs = params[
                    'meanshift_qs']  # meanshift uses the same kjl_qs, and only needs to tune one of them
                if 'meanshift_qs' in params.keys(): del params['meanshift_qs']
                with parallel:
                    # outs = parallel(delayed(_model_train_test)(X, y, params, kjl_d=kjl_d,
                    #                                            kjl_n=kjl_n, kjl_q=kjl_q, n_components=n_components)
                    #                 for kjl_d, kjl_n, kjl_q, n_components in
                    #                 list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))

                    outs = parallel(
                        delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params), kjl_d=kjl_d,
                                                   kjl_n=kjl_n, kjl_q=kjl_q)
                        for kjl_d, kjl_n, kjl_q in
                        list(itertools.product(kjl_ds, kjl_ns, kjl_qs)))
            else:
                msg = params['is_kjl']
                raise NotImplementedError(f'Error: kjl={msg}')

        elif params['is_nystrom']:
            # GMM-gs:True-nystrom:True
            nystrom_ns = params['nystrom_ns']
            nystrom_ds = params['nystrom_ds']
            nystrom_qs = params['nystrom_qs']
            if 'nystrom_ns' in params.keys(): del params['nystrom_ns']
            if 'nystrom_ds' in params.keys(): del params['nystrom_ds']
            if 'nystrom_qs' in params.keys(): del params['nystrom_qs']
            if not params['is_quickshift'] and not params['is_meanshift']:
                with parallel:
                    outs = parallel(delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params),
                                                               nystrom_n=nystrom_n,
                                                               nystrom_d=nystrom_d, nystrom_q=nystrom_q,
                                                               GMM_n_components=n_components) for
                                    nystrom_n, nystrom_d, nystrom_q, n_components in
                                    list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, n_components_arr)))

            elif params['is_quickshift']:
                # GMM-gs:True-kjl:True-quickshift:True
                quickshift_ks = params['quickshift_ks']
                quickshift_betas = params['quickshift_betas']
                if 'quickshift_ks' in params.keys(): del params['quickshift_ks']
                if 'quickshift_betas' in params.keys(): del params['quickshift_betas']
                with parallel:
                    outs = parallel(
                        delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params),
                                                   nystrom_n=nystrom_n,
                                                   nystrom_d=nystrom_d, nystrom_q=nystrom_q,
                                                   quickshift_k=quickshift_k, quickshift_beta=quickshift_beta)
                        for nystrom_n, nystrom_d, nystrom_q, quickshift_k, quickshift_beta in
                        list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, quickshift_ks, quickshift_betas)))

            elif params['is_meanshift']:
                # GMM-gs:True-kjl:True-meanshift:True
                meanshift_qs = params[
                    'meanshift_qs']  # meanshift uses the same kjl_qs, and only needs to tune one of them
                if 'meanshift_qs' in params.keys(): del params['meanshift_qs']
                with parallel:
                    # outs = parallel(delayed(_model_train_test)(X, y, params, kjl_d=kjl_d,
                    #                                            kjl_n=kjl_n, kjl_q=kjl_q, n_components=n_components)
                    #                 for kjl_d, kjl_n, kjl_q, n_components in
                    #                 list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))

                    outs = parallel(
                        delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params),
                                                   nystrom_n=nystrom_n,
                                                   nystrom_d=nystrom_d, nystrom_q=nystrom_q, kjl_q=nystrom_q)
                        for nystrom_n, nystrom_d, nystrom_q in
                        list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs)))
        else:
            msg = params['is_kjl']
            raise NotImplementedError(f'Error: kjl={msg}')

    elif params['detector_name'] == 'OCSVM':

        if 'is_kjl' in params.keys() and params['is_kjl']:  # gs=True, kjl = True and for OCSVM ('linear')
            print('---')
            kjl_ds = params['kjl_ds']
            kjl_ns = params['kjl_ns']
            kjl_qs = params['kjl_qs']
            if 'kjl_ds' in params.keys(): del params['kjl_ds']
            if 'kjl_ns' in params.keys(): del params['kjl_ns']
            if 'kjl_qs' in params.keys(): del params['kjl_qs']
            model_nus = params['OCSVM_nus']
            if 'OCSVM_nus' in params.keys(): del params['OCSVM_nus']
            with parallel:
                outs = parallel(
                    delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params), kjl_d=kjl_d,
                                               kjl_n=kjl_n,
                                               kjl_q=kjl_q,
                                               OCSVM_nu=OCSVM_nu) for
                    kjl_d, kjl_n, kjl_q, OCSVM_nu in list(itertools.product(kjl_ds, kjl_ns, kjl_qs, model_nus)))

        elif 'is_nystrom' in params.keys() and params['is_nystrom']:  # gs=True, nystrom = True and for OCSVM ('linear')
            print('---')
            # GMM-gs:True-nystrom:True
            nystrom_ns = params['nystrom_ns']
            nystrom_ds = params['nystrom_ds']
            nystrom_qs = params['nystrom_qs']
            if 'nystrom_ns' in params.keys(): del params['nystrom_ns']
            if 'nystrom_ds' in params.keys(): del params['nystrom_ds']
            if 'nystrom_qs' in params.keys(): del params['nystrom_qs']
            model_nus = params['OCSVM_nus']
            if 'OCSVM_nus' in params.keys(): del params['OCSVM_nus']
            with parallel:
                outs = parallel(
                    delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params),
                                               nystrom_n=nystrom_n,
                                               nystrom_d=nystrom_d, nystrom_q=nystrom_q,
                                               OCSVM_nu=OCSVM_nu) for
                    nystrom_n, nystrom_d, nystrom_q, OCSVM_nu in
                    list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, model_nus)))

        else: # not params['is_kjl'] and not params['is_nystrom']:
            # msg = params['is_kjl']
            # raise NotImplementedError(f'Error: kjl={msg}')
            with parallel:
                model_qs = params['OCSVM_qs']
                model_nus = params['OCSVM_nus']
                if 'OCSVM_qs' in params.keys(): del params['OCSVM_qs']
                if 'OCSVM_nus' in params.keys(): del params['OCSVM_nus']

                outs = parallel(
                    delayed(_model_train_test)(X_train, y_train, X_val, y_val, copy.deepcopy(params), OCSVM_q=OCSVM_q,
                                               OCSVM_nu=OCSVM_nu) for _, _, _, OCSVM_q, OCSVM_nu in
                    list(itertools.product([0], [0], [0], model_qs, model_nus)))

    else:
        msg = params['detector_name']
        raise NotImplementedError(f'Error: detector_name={msg}')

    # get the best avg auc from n_repeats experiments
    best_avg_auc = -1
    print(outs)
    for out in outs:
        if np.mean(out['auc']) > best_avg_auc:
            best_avg_auc = np.mean(out['auc'])
            best_results = copy.deepcopy(out)

    # it's better to save all middle results too
    middle_results = outs

    print('---get accurate time of training and testing with the best params---')
    best_params = best_results['params']
    params['mode'] = 'default'  # evaluate on the test_set to get the final results
    out = _model_train_test(X_train, y_train, X_test, y_test, params=best_params)
    # double check the results if you can
    # assert best_avg_auc == np.mean(out['aucs'])
    # print(best_avg_auc, np.mean(out['aucs']), best_results, out)
    best_results = out

    return copy.deepcopy(best_results), copy.deepcopy(middle_results)


def single_main(model_cfg, data_cfg):
    # print(f'single_main.kwargs: {kwargs.items()}')
    # (data_name, data_file), (X, y) = kwargs['data']
    # case, params = kwargs['params']
    data_name = data_cfg['data_name']
    data_file = data_cfg['data_file']
    (X, y) = data_cfg['data']
    feat_type = data_cfg['feat']

    model_name = model_cfg['model_name']
    train_size = model_cfg['train_size']
    is_gs = model_cfg['is_gs']
    if 'GMM_covariance_type' in model_cfg.keys():
        GMM_covariance_type = model_cfg['GMM_covariance_type']
    else:
        GMM_covariance_type = None

    if feat_type.upper().startswith('SAMP_'):
        # only find the maximum one
        best_auc = -1
        print(f'****X: {X}')
        for q_samp_rate in X.keys():
            X_, y_ = X[q_samp_rate], y[q_samp_rate]
            n_repeats = model_cfg['n_repeats']
            random_state = model_cfg['random_state']
            # params['GMM_n_components'] = [int(X.shape[1])]
            X_normal, y_normal, X_abnormal, y_abnormal = seperate_normal_abnormal(X_, y_, random_state=random_state)
            # get the unique test set
            X_normal, y_normal, X_abnormal, y_abnormal, X_test, y_test = split_left_test(X_normal, y_normal, X_abnormal,
                                                                                         y_abnormal, test_size=600,
                                                                                         random_state=random_state)
            train_times = []
            test_times = []
            aucs = []
            space_sizes = []
            _params = []
            _middle_results = []
            _best_train_times = []
            _best_test_times = []
            _best_aucs = []
            _best_space_sizes = []
            _best__params = []
            _best_middle_results = []
            for i in range(n_repeats):
                print(f"\n\n==={i + 1}/{n_repeats}(n_repeats): {model_cfg}===")
                X_train, y_train, X_val, y_val = split_train_val(X_normal, y_normal, X_abnormal, y_abnormal,
                                                                 train_size=train_size,
                                                                 val_size=int(len(y_test) * 0.25),
                                                                 random_state=(i + 1) * 100)
                _best_results_i, _middle_results_i = get_best_results(X_train, y_train, X_val, y_val, X_test, y_test,
                                                                      copy.deepcopy(model_cfg),
                                                                      random_state=random_state)
                _middle_results.append(_middle_results_i)

                train_times.append(_best_results_i['train_time'])
                test_times.append(_best_results_i['test_time'])
                aucs.append(_best_results_i['auc'])
                space_sizes.append(_best_results_i['space_size'])
                _params.append(_best_results_i['params'])

            # find the best average AUC from all q_samp_rate
            if np.mean(aucs) > best_auc:
                _best_train_times = copy.deepcopy(train_times)
                _best_test_times = copy.deepcopy(test_times)
                _best_aucs = copy.deepcopy(aucs)
                _best_space_sizes = copy.deepcopy(space_sizes)
                _best_params = copy.deepcopy(_params)
                _best_middle_results = copy.deepcopy(_middle_results)

        if is_gs:
            print(f'--is_gs: {is_gs}, X_val != X_test')
        else:
            X_val = X_test
        _best_results = {'train_times': _best_train_times, 'test_times': _best_test_times, 'aucs': _best_aucs, 'apcs': '',
                         'params': _best_params,
                         'space_sizes': _best_space_sizes,
                         'X_train_shape': X_train.shape, 'X_val_shape': X_val.shape, 'X_test_shape': X_test.shape}

        result = ((f'{data_name}|{data_file}', model_name), (_best_results, _best_middle_results))
        # # dump each result to disk to avoid runtime error in parallel context.
        # dump_data(result, out_file=(f'{os.path.dirname(data_file)}/gs_{is_gs}-{GMM_covariance_type}/{case}.dat'))

    else:

        n_repeats = model_cfg['n_repeats']
        random_state = model_cfg['random_state']
        # params['GMM_n_components'] = [int(X.shape[1])]
        X_normal, y_normal, X_abnormal, y_abnormal = seperate_normal_abnormal(X, y, random_state=random_state)
        # get the unique test set
        X_normal, y_normal, X_abnormal, y_abnormal, X_test, y_test = split_left_test(X_normal, y_normal, X_abnormal,
                                                                                     y_abnormal, test_size=600,
                                                                                     random_state=random_state)
        train_times = []
        test_times = []
        aucs = []
        space_sizes = []
        _params = []
        _middle_results = []
        for i in range(n_repeats):
            print(f"\n\n==={i + 1}/{n_repeats}(n_repeats): {model_cfg}===")
            X_train, y_train, X_val, y_val = split_train_val(X_normal, y_normal, X_abnormal, y_abnormal,
                                                             train_size=train_size, val_size=int(len(y_test) * 0.25),
                                                             random_state=(i + 1) * 100)
            _best_results_i, _middle_results_i = get_best_results(X_train, y_train, X_val, y_val, X_test, y_test,
                                                                  copy.deepcopy(model_cfg),
                                                                  random_state=random_state)
            _middle_results.append(_middle_results_i)

            train_times.append(_best_results_i['train_time'])
            test_times.append(_best_results_i['test_time'])
            aucs.append(_best_results_i['auc'])
            space_sizes.append(_best_results_i['space_size'])
            _params.append(_best_results_i['params'])

        if is_gs:
            print(f'--is_gs: {is_gs}, X_val != X_test')
        else:
            X_val = X_test
        _best_results = {'train_times': train_times, 'test_times': test_times, 'aucs': aucs, 'apcs': '',
                         'params': _params,
                         'space_sizes': space_sizes,
                         'X_train_shape': X_train.shape, 'X_val_shape': X_val.shape, 'X_test_shape': X_test.shape}

        result = ((f'{data_name}|{data_file}', model_name), (_best_results, _middle_results))
        # # dump each result to disk to avoid runtime error in parallel context.
        # dump_data(result, out_file=(f'{os.path.dirname(data_file)}/gs_{is_gs}-{GMM_covariance_type}/{case}.dat'))

    return result

#
# def main(random_state, n_jobs=-1, n_repeats=1):
#     datasets = [
#         #     # # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',
#         'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
#         #     # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
#         #     # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
#         #     # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
#         #     # # # # #
#         #     # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
#         #     # # # # # #
#         #     'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
#         # 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196',
#         # #     # # #
#         # #     # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
#         # #     # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
#         # 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
#         # #     # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
#         # #     # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
#         # #     # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
#         # #     # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
#         # #     # #
#         # #     # # #
#         # #     # 'WRCCDC/2020-03-20',
#         # #     # 'DEFCON/ctf26',
#         # 'ISTS/2015',
#         # 'MACCDC/2012',
#         #
#         # #     # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
#         # 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
#         # #     # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
#         # #     # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
#         # #
#
#     ]
#     feat_set = 'iat_size'
#     header = False
#
#     in_dir = 'data/data_kjl'
#     out_dir = 'out/data_kjl'
#     results = {}
#     for data_name in datasets:
#         # 1. get data params
#         in_expand_dir = pth.join(in_dir, data_name, feat_set, f'header:{header}')
#         out_expand_dir = pth.join(out_dir, data_name, feat_set, f'header:{header}')
#         normal_file = f'{in_expand_dir}/normal.dat'
#         abnormal_file = f'{in_expand_dir}/abnormal.dat'
#         print(normal_file, abnormal_file)
#
#         if not pth.exists(normal_file) or not pth.exists(abnormal_file):
#             _normal_file = pth.splitext(normal_file)[0] + '.csv'
#             _abnormal_file = pth.splitext(abnormal_file)[0] + '.csv'
#             # extract data from csv file
#             normal_data, abnormal_data = extract_data(_normal_file, _abnormal_file,
#                                                       meta_data={'idxs_feat': [0, -1],
#                                                                  'train_size': -1,
#                                                                  'test_size': -1})
#             # transform data format
#             dump_data(normal_data, normal_file)
#             dump_data(abnormal_data, abnormal_file)
#         else:
#             normal_data = load_data(normal_file)
#             abnormal_data = load_data(abnormal_file)
#
#         # try to make case more smaller and specific
#         gs = True
#
#         cases = [  # OCSVM-gs:True
#             {'detector_name': 'OCSVM', 'gs': gs},
#
#             # # GMM-gs:True
#             # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs},
#             # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs},
#             #
#             # # # GMM-gs:True-kjl:True
#             # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True},
#             # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': True},
#             # #
#             # # GMM-gs:True-kjl:True-nystrom:True   # nystrom will take more time than kjl
#             # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': False, 'nystrom': True},
#             # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': False, 'nystrom': True},
#             #
#             # # GMM-gs:True-kjl:True-quickshift:True
#             # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True, 'quickshift': True},
#             # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': True, 'quickshift': True},
#             # #
#             # # GMM-gs:True-kjl:True-meanshift:True
#             # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True, 'meanshift': True},
#             # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': True, 'meanshift': True},
#         ]
#
#         for case in cases:
#             case['n_repeats'] = n_repeats
#             case['n_jobs'] = n_jobs
#             if 'gs' not in case.keys():
#                 case['gs'] = False
#                 if case['detector_name'] == 'OCSVM':
#                     case['model_qs'] = [0.3]
#                     case['model_nus'] = [0.1]
#                 elif case['detector_name'] == 'GMM':
#                     case['n_components'] = [1]
#                 else:
#                     raise ValueError(f'Error: {case}')
#             else:
#                 case['gs_search_type'] = 'grid_search'
#                 if case['detector_name'] == 'OCSVM':
#                     case['model_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#                                         0.95]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#                     case['model_nus'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#                 elif case['detector_name'] == 'GMM':
#                     case['n_components'] = [1, 5, 10, 15, 20, 25, 30, 35, 40,
#                                             45]  # [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
#                 else:
#                     raise ValueError(f'Error: {case}')
#
#             if 'kjl' not in case.keys():
#                 case['kjl'] = False
#                 case['kjl_ns'] = [100]
#                 case['kjl_ds'] = [10]
#                 case['kjl_qs'] = [0.3]
#             else:
#                 if case['kjl']:
#                     case['kjl_ns'] = [100]
#                     case['kjl_ds'] = [10]
#                     case['kjl_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#                                       0.95]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#
#             if 'quickshift' not in case.keys():
#                 case['quickshift'] = False
#                 case['quickshift_ks'] = [100]
#                 case['quickshift_betas'] = [0.9]
#             else:
#                 case['quickshift_ks'] = [100, 1000]
#                 case['quickshift_betas'] = [0.1, 0.3, 0.5, 0.7, 0.9]
#
#             if 'meanshift' not in case.keys():
#                 case['meanshift'] = False
#             else:
#                 case['meanshift_qs'] = ''  # use the same qs of kjl_qs
#
#             if 'nystrom' not in case.keys():
#                 case['nystrom'] = False
#                 case['nystrom_ns'] = [100]
#                 case['nystrom_ds'] = [10]
#                 case['nystrom_qs'] = [0.3]
#             else:
#                 if case['nystrom']:
#                     case['nystrom_ns'] = [100]
#                     case['nystrom_ds'] = [10]
#                     case['nystrom_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#                                           0.95]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#
#             case_str = '-'.join([f'{k}_{v}' for k, v in case.items() if
#                                  k in ['detector_name', 'covariance_type', 'gs', 'kjl', 'nystrom', 'quickshift',
#                                        'meanshift']])
#             try:
#                 # get each result
#                 print(f"\n\n\n***{case}***, {case_str}")
#                 _best_results, _middle_results = get_best_results(normal_data, abnormal_data, case, random_state)
#
#                 # save each result first
#                 out_file = pth.abspath(f'{out_expand_dir}/{case_str}.csv')
#                 print('+++', out_file)
#                 save_each_result(_best_results, case_str, out_file)
#
#                 dump_data(_middle_results, out_file + '-middle_results.dat')
#
#                 results[(in_expand_dir, case_str)] = (_best_results, _middle_results)
#             except Exception as e:
#                 traceback.print_exc()
#                 print(f"some error exists in {case}")
#
#     # save results first
#     out_file = f'{out_dir}/all_results.csv'
#     print(f'\n\n***{out_file}***')
#     # Todo: format the results
#     dat2csv(results, out_file)
#     print("\n\n---finish succeeded!")
#
#
# if __name__ == '__main__':
#     main(random_state=42, n_jobs=-1, n_repeats=1)
