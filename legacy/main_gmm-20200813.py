"""Main entrance
    run under "applications"
    PYTHONPATH=../:./ python3.7 main_kjl_parallel.py > out/main_kjl_parallel.txt 2>&1 &
"""

import copy
import itertools
import os
import os.path as pt
import traceback
from datetime import datetime

import numpy as np
from joblib import delayed, Parallel
from sklearn import metrics
from sklearn.metrics import pairwise_distances, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler

from kjl.models.gmm import GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.models.online_gmm import ONLINE_GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.models.kjl import kernelJLInitialize, getGaussianGram
from kjl.models.nystrom import nystromInitialize
from kjl.models.ocsvm import OCSVM
from kjl.utils.data import data_info, split_train_test, load_data, extract_data, dump_data, save_each_result, \
    save_result
from kjl.utils.utils import execute_time

RANDOM_STATE = 42

print('PYTHONPATH: ', os.environ['PYTHONPATH'])


class BASE:

    def __init__(self):
        pass

    def project_kjl(self, X_train, X_test, kjl_params={}):
        debug = False
        if kjl_params['kjl']:
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

            # # for debug
            if debug:
                data_info(X_test, name='X_test_std')
                _qs = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
                _sigmas = np.quantile(pairwise_distances(X_test), _qs)
                print(f'test set\' sigmas with qs: {list(zip(_sigmas, _qs))}')

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
        return X_train, X_test

    def project_nystrom(self, X_train, X_test, nystrom_params={}):
        debug = False
        if nystrom_params['nystrom']:
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
            X_test = np.matmul(np.matmul(K, Eigvec), np.diag(1. / np.sqrt(np.diag(Lambda))))
            if debug: data_info(X_test, name='after nystrom, X_test')
            end = datetime.now()
            nystrom_test_time = (end - start).total_seconds()
            print("nystrom on test set took {} seconds".format(nystrom_test_time))
        else:
            nystrom_train_time = 0
            nystrom_test_time = 0

        self.nystrom_train_time = nystrom_train_time
        self.nystrom_test_time = nystrom_test_time
        return X_train, X_test

    def standardize(self, X_train, X_test):
        scaler = StandardScaler()

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

    def _train(self, model, X_train, y_train=None):
        """Train models on the (X_train, y_train)

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
        except Exception as e:
            msg = f'fit error: {e}'
            print(msg)
            raise ValueError(f'{msg}: {model.get_params()}')
        end = datetime.now()
        train_time = (end - start).total_seconds()
        print("Fitting models takes {} seconds".format(train_time))

        return model, train_time

    def _test(self, model, X_test, y_test):
        """Evaulate the models on the X_test, y_test

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
            preds = models.predict_proba(X_test)
            pred = 1 - np.prod(1-preds, axis=1)
        else:
            pred = models.score_samples(X_test)
        """
        end = datetime.now()
        testing_time = (end - start).total_seconds()
        print("Test models takes {} seconds".format(testing_time))

        apc = average_precision_score(y_test, y_score, pos_label=1)
        # For binary  y_true, y_score is supposed to be the score of the class with greater label.
        # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
        # pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # auc1 = roc_auc_score(y_test, y_score)
        # print(models.get_params())
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

        # should do standardization before using pairwise_distances
        X_train, X_test = self.standardize(X_train, X_test)
        self.train_time += self.std_train_time
        self.test_time += self.std_test_time

        self.thres_n = 100  # used to filter clusters which have less than 100 datapoints
        if 'meanshift' in self.params.keys() and self.params['meanshift']:
            dists = pairwise_distances(X_train)
            self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
            self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                X_train, bandwidth=self.sigma, thres_n=self.thres_n)
            self.params['n_components'] = self.n_components
            self.seek_test_time = 0
        elif 'quickshift' in self.params.keys() and self.params['quickshift']:
            self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                thres_n=self.thres_n)
            self.params['n_components'] = self.n_components
            self.seek_test_time = 0
        else:
            self.seek_train_time = 0
            self.seek_test_time = 0
        self.train_time += self.seek_train_time
        self.test_time += self.seek_test_time  # self.seek_test_time = 0

        if 'kjl' in self.params.keys() and self.params['kjl']:
            X_train, X_test = self.project_kjl(X_train, X_test, kjl_params=self.params)
            self.proj_train_time = self.kjl_train_time
            self.proj_test_time = self.kjl_test_time
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            X_train, X_test = self.project_nystrom(X_train, X_test, nystrom_params=self.params)
            self.proj_train_time = self.nystrom_train_time
            self.proj_test_time = self.nystrom_test_time
        else:
            self.proj_train_time = 0
            self.proj_test_time = 0
        self.train_time += self.proj_train_time
        self.test_time += self.proj_test_time

        model = GMM()
        model_params = {'n_components': self.params['n_components'], 'covariance_type': self.params['covariance_type'],
                        'means_init': None, 'random_state': self.random_state}
        # set models default parameters
        model.set_params(**model_params)
        print(model.get_params())
        # train models
        self.model, self.model_train_time = self._train(model, X_train)

        self.train_time += self.model_train_time

        print(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
              f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
              f'model_train_time: {self.model_train_time}')

        self.y_score, self.model_test_time, self.auc = self._test(self.model, X_test, y_test)
        self.test_time += self.model_test_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, seek_test_time: {self.seek_test_time}'
            f', proj_test_time: {self.proj_test_time}, '
            f'model_test_time: {self.model_test_time}')

        return self


class ONLINE_GMM_MAIN(BASE, ONLINE_GMM):

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

        # should do standardization before using pairwise_distances
        X_train, X_test = self.standardize(X_train, X_test)
        self.train_time += self.std_train_time
        self.test_time += self.std_test_time

        self.thres_n = 100  # used to filter clusters which have less than 100 datapoints
        if 'meanshift' in self.params.keys() and self.params['meanshift']:
            dists = pairwise_distances(X_train)
            self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
            self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                X_train, bandwidth=self.sigma, thres_n=self.thres_n)
            self.params['n_components'] = self.n_components
            self.seek_test_time = 0
        elif 'quickshift' in self.params.keys() and self.params['quickshift']:
            self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                thres_n=self.thres_n)
            self.params['n_components'] = self.n_components
            self.seek_test_time = 0
        else:
            self.seek_train_time = 0
            self.seek_test_time = 0
        self.train_time += self.seek_train_time
        self.test_time += self.seek_test_time  # self.seek_test_time = 0

        if 'kjl' in self.params.keys() and self.params['kjl']:
            X_train, X_test = self.project_kjl(X_train, X_test, kjl_params=self.params)
            self.proj_train_time = self.kjl_train_time
            self.proj_test_time = self.kjl_test_time
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            X_train, X_test = self.project_nystrom(X_train, X_test, nystrom_params=self.params)
            self.proj_train_time = self.nystrom_train_time
            self.proj_test_time = self.nystrom_test_time
        else:
            self.proj_train_time = 0
            self.proj_test_time = 0
        self.train_time += self.proj_train_time
        self.test_time += self.proj_test_time

        model = ONLINE_GMM()
        model_params = {'n_components': self.params['n_components'],
                        'covariance_type': self.params['covariance_type'],
                        'means_init': None, 'random_state': self.random_state}
        # set models default parameters
        model.set_params(**model_params)
        print(model.get_params())
        # train models
        self.model, self.model_train_time = self._train(model, X_train)

        self.train_time += self.model_train_time

        y_train_score = model.decision_function(X_train)
        self.abnormal_thres = np.quantile(y_train_score, q=0.99)  # abnormal threshold
        self.novelty_thres = np.quantile(y_train_score, q=0.55)  # normal threshold
        print(f'novelty_thres: {self.novelty_thres}, abnormal_thres: {self.abnormal_thres}')

        _, self.model.log_resp = self.model._e_step(X_train)
        self.model.n_samples = X_train.shape[0]
        self.model.X_train = X_train

        print(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
              f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
              f'model_train_time: {self.model_train_time}')

        self.y_score, self.model_test_time, self.auc = self._test(self.model, X_test, y_test)
        self.test_time += self.model_test_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, seek_test_time: {self.seek_test_time}'
            f', proj_test_time: {self.proj_test_time}, '
            f'model_test_time: {self.model_test_time}')

        return self

    def _test(self, model, X_test, y_test):
        """Evaulate the models on the X_test, y_test

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

        y_score = []

        for i, _X_test in enumerate(X_test):  # each time only arrives one datapoint.
            start = datetime.now()
            # For inlier, a small value is used; a larger value is for outlier (positive)
            # it must be abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
            _X_test = _X_test.reshape(1, -1)
            _y_score = model.decision_function(_X_test)

            """
            if model_name == "Gaussian" and n_components != 1:
                preds = models.predict_proba(X_test)
                pred = 1 - np.prod(1-preds, axis=1)
            else:
                pred = models.score_samples(X_test)
            """
            end = datetime.now()
            testing_time = (end - start).total_seconds()
            print("i:{}, Test models takes {} seconds, y_score: {}".format(i, testing_time, _y_score))

            ### update the current models

            # scenarios 1: train models
            #   sub_scenarios: add a new component
            #   sub_scenarios: drop the abnormal datapoint

            # scenarios 2: train models until convergent
            #   sub_scenarios: add a new component
            #   sub_scenarios: drop the abnormal datapoint
            model.warm_start = True
            self.tol = 1e-3
            self.max_iter = 1

            # lower_bound = (-np.infty if do_init else self.lower_bound_)
            lower_bound = -np.infty
            prev_lower_bound = lower_bound
            if _y_score < self.abnormal_thres:  # normal datapoint, update GMM
                if _y_score >= self.novelty_thres:

                    # self.novelty_thres < _y_score < self.abnormal_thres
                    # novelty datapoint (but still is a normal datapoint) create a new component, update GMM
                    # n_feats = X_test.shape[1]

                    # log_prob_norm, log_resp = models._e_step(_X_test)  # get log_prob and resp

                    model.n_components += 1
                    n_samples, n_feats = _X_test.shape
                    model.log_resp = np.concatenate([model.log_resp, np.zeros((model.n_samples, 1))], axis=1)
                    model.n_samples += n_samples
                    new_log_resp = np.zeros((1, model.n_components))
                    new_log_resp[-1] = 1
                    model.log_resp = np.concatenate([model.log_resp , new_log_resp], axis=0)

                    print(f'models.n_components: {model.n_components}')

                    model.X_train = np.concatenate([model.X_train, _X_test], axis=0)
                    log_prob_norm = 0

                    for n_iter in range(1, self.max_iter + 1):
                        prev_lower_bound = lower_bound

                        # nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
                        # models.weights_ = np.concatenate([models.weights_, np.asarray([0.1])], axis=0)
                        # models.means_ = np.concatenate([models.means_, np.zeros((1, n_feats))], axis=0)
                        # models.covariances_ = np.concatenate(
                        #    [models.covariances_, np.ones((n_feats, n_feats)).reshape((1, n_feats, n_feats))], axis=0)

                        # models.precisions_cholesky_ = np.concatenate(
                        #    [models.precisions_cholesky_, np.ones((n_feats, n_feats)).reshape((1, n_feats, n_feats))], axis=0)
                        # self.models, self.model_train_time = self._train(models, _X_test)


                        # models._m_step(_X_test, models.log_resp)  # update weight, mean, covariance
                        model._m_step(model.X_train, model.log_resp)  # update weight, mean, covariance

                        lower_bound = model._compute_lower_bound(
                            model.log_resp, log_prob_norm)

                        change = lower_bound - prev_lower_bound
                        # self._print_verbose_msg_iter_end(n_iter, change)

                        if abs(change) < self.tol:
                            self.converged_ = True
                            print(f'n_iter: {n_iter}')
                            break

                        log_prob_norm, model.log_resp = model._e_step(model.X_train)  # get log_prob and resp


                    print(self.model.get_params())

                else:  # normal datapoint, update GMM
                    # self.models, self.model_train_time = self._train(models, _X_test)

                    model.X_train = np.concatenate([model.X_train, _X_test], axis=0)
                    n_samples, n_feats = _X_test.shape
                    model.n_samples += n_samples
                    for n_iter in range(1, self.max_iter + 1):
                        prev_lower_bound = lower_bound

                        log_prob_norm, log_resp = model._e_step(model.X_train)  # get log_prob and resp
                        model._m_step(model.X_train, log_resp)  # update mean, covariance and weight
                        lower_bound = model._compute_lower_bound(
                            log_resp, log_prob_norm)

                        change = lower_bound - prev_lower_bound
                        # self._print_verbose_msg_iter_end(n_iter, change)

                        if abs(change) < self.tol:
                            self.converged_ = True
                            print(f'n_iter: {n_iter}')
                            break
                    model.log_resp = log_resp
                        # log_prob_norm, log_resp = models._e_step(_X_test)  # get log_prob and resp
                        # ith_component= np.argmax(log_resp)
                        # # update mean, covariance and weight

                        # models.n_samples = _X_test.shape[0]
                        # log_prob_norm, log_resp = models._estimate_log_prob_resp(_X_test)
                        # log_prob_norm, log_resp = np.mean(log_prob_norm), log_resp

                        # models.means_[ith_component] = []
                        # models._m_step(_X_test, log_resp)
                        # resp = log_resp

                        # nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
                        # means = np.dot(resp.T, _X_test) / nk[:, np.newaxis]
                        #
                        # n_components, n_features = means.shape
                        # reg_covar = 1e-6
                        # covariances = np.empty((n_components, n_features, n_features))
                        # for k in range(n_components):
                        #     diff = _X_test - means[k]
                        #     covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                        #     covariances[k].flat[::n_features + 1] += reg_covar

                        # models.weights_ = nk / models.n_samples
                        # models.means_ = means
                        # models.covariances_ = covariances

                        # lower_bound = models._compute_lower_bound(
                        #     log_resp, log_prob_norm)
                        #
                        # change = lower_bound - prev_lower_bound
                        # if abs(change) < self.tol:
                        #     self.converged_ = True
                        #     break

                    # print(self.models.get_params())

            else:  # if _y_score >= self.abnormal_thres: # normal datapoint, update GMM
                print('this flow is an abnormal flow, so we drop it.')
            y_score.append(_y_score)

        apc = average_precision_score(y_test, y_score, pos_label=1)
        # For binary  y_true, y_score is supposed to be the score of the class with greater label.
        # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
        # pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # auc1 = roc_auc_score(y_test, y_score)
        # print(models.get_params())
        # assert auc==auc1

        # f1, bestEp = selectThreshHold(test_y_i, pred)

        # if auc > max_auc:
        #     max_auc = auc
        #     best_pred = y_score

        print("APC: {}".format(apc))
        print("AUC: {}".format(auc))
        # print("F1: {}".format(f1))

        return y_score, testing_time, auc


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

        # should do standardization before using pairwise_distances
        X_train, X_test = self.standardize(X_train, X_test)
        self.train_time += self.std_train_time
        self.test_time += self.std_test_time

        # OCSVM does not need seek mode
        self.thres_n = 100  # used to filter clusters which have less than 100 datapoints
        self.seek_train_time = None
        self.seek_test_time = None

        X_train, X_test = self.project_kjl(X_train, X_test, kjl_params=self.params)
        self.train_time += self.kjl_train_time
        self.test_time += self.kjl_test_time

        model = OCSVM()
        self.params['kernel'] = 'rbf'
        kjl = self.params['kjl']
        if kjl:  # if kjl=True, then use 'linear' kernel for OCSVM, so only tune 'nu' in this case
            self.kernel = 'linear'
            self.model_gamma = self.params['model_gamma']
            model_params = {'kernel': self.kernel, 'gamma': self.params['model_gamma'], 'nu': self.params['model_nu']}
        else:
            self.model_gamma = 1 / (np.quantile(pairwise_distances(X_train), self.params['model_q']) ** 2)
            model_params = {'kernel': self.params['kernel'], 'gamma': self.model_gamma, 'nu': self.params['model_nu']}
        # set models default parameters
        model.set_params(**model_params)
        # print(models.get_params())
        # train models
        self.model, self.model_train_time = self._train(model, X_train)

        self.train_time += self.model_train_time
        # print(f'{self.models.get_params()}, {self.params}')

        print(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
              f'{self.seek_train_time}, kjl_train_time: {self.kjl_train_time}, '
              f'model_train_time: {self.model_train_time}')

        self.y_score, self.model_test_time, self.auc = self._test(self.model, X_test, y_test)
        self.test_time += self.model_test_time
        print(f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, seek_test_time: '
              f'{self.seek_test_time}, kjl_test_time: {self.kjl_test_time}, '
              f'model_test_time: {self.model_test_time}')

        return self


def _model_train_test(normal_data, abnormal_data, params, **kargs):
    """

    Parameters
    ----------
    normal_data
    abnormal_data
    params
    args

    Returns
    -------

    """
    try:
        for k, v in kargs.items():
            params[k] = v

        n_repeats = params['n_repeats']
        train_times = []
        test_times = []
        aucs = []
        # keep that all the algorithms have the same input data by setting random_state
        for i in range(n_repeats):
            print(f"\n\n==={i + 1}/{n_repeats}(n_repeats): {params}===")
            X_train, y_train, X_test, y_test = split_train_test(normal_data, abnormal_data,
                                                                train_size=0.8, test_size=-1, random_state=i * 100)

            if "GMM" in params['detector_name']:
                if params['online_gmm']:
                    model = ONLINE_GMM_MAIN(params)
                else:
                    model = GMM_MAIN(params)
            elif "OCSVM" in params['detector_name']:
                model = OCSVM_MAIN(params)

            model.train_test_intf(X_train, y_train, X_test, y_test)

            train_times.append(model.train_time)
            test_times.append(model.test_time)
            aucs.append(model.auc)

        info = {'train_times': train_times, 'test_times': test_times, 'aucs': aucs, 'apcs': '',
                'params': params,
                'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}

    except Exception as e:
        traceback.print_exc()
        info = {'train_times': [0.0], 'test_times': [0.0], 'aucs': [0.0], 'apcs': '',
                'params': params,
                'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}
    return info


@execute_time
def get_best_results(normal_data, abnormal_data, case, random_state=42):
    """

    Parameters
    ----------
    normal_data
    abnormal_data
    case

    Returns
    -------

    """
    params = copy.deepcopy(case)
    params['random_state'] = random_state

    parallel = Parallel(n_jobs=params['n_jobs'], verbose=30)

    if params['detector_name'] == 'GMM':
        # GMM with grid search
        if params['gs']:
            n_components_arr = params['n_components']
            if 'n_components' in params.keys(): del params['n_components']
            if not params['kjl'] and not params['nystrom']:  # only GMM
                # GMM-gs:True
                with parallel:
                    outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params,
                                                               n_components=n_components) for n_components, _ in
                                    list(itertools.product(n_components_arr, [0])))

            elif params['kjl']:
                kjl_ds = params['kjl_ds']
                kjl_ns = params['kjl_ns']
                kjl_qs = params['kjl_qs']
                if 'kjl_ds' in params.keys(): del params['kjl_ds']
                if 'kjl_ns' in params.keys(): del params['kjl_ns']
                if 'kjl_qs' in params.keys(): del params['kjl_qs']
                if not params['quickshift'] and not params['meanshift']:
                    # GMM-gs:True-kjl:True
                    with parallel:
                        outs = parallel(
                            delayed(_model_train_test)(normal_data, abnormal_data, params, kjl_d=kjl_d, kjl_n=kjl_n,
                                                       kjl_q=kjl_q,
                                                       n_components=n_components) for kjl_d, kjl_n, kjl_q, n_components
                            in
                            list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))

                elif params['quickshift']:
                    # GMM-gs:True-kjl:True-quickshift:True
                    quickshift_ks = params['quickshift_ks']
                    quickshift_betas = params['quickshift_betas']
                    if 'quickshift_ks' in params.keys(): del params['quickshift_ks']
                    if 'quickshift_betas' in params.keys(): del params['quickshift_betas']
                    with parallel:
                        outs = parallel(
                            delayed(_model_train_test)(normal_data, abnormal_data, params,
                                                       kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                                       quickshift_k=quickshift_k, quickshift_beta=quickshift_beta)
                            for kjl_d, kjl_n, kjl_q, quickshift_k, quickshift_beta in
                            list(itertools.product(kjl_ds, kjl_ns, kjl_qs, quickshift_ks, quickshift_betas)))

                elif params['meanshift']:
                    # GMM-gs:True-kjl:True-meanshift:True
                    meanshift_qs = params[
                        'meanshift_qs']  # meanshift uses the same kjl_qs, and only needs to tune one of them
                    with parallel:
                        # outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, kjl_d=kjl_d,
                        #                                            kjl_n=kjl_n, kjl_q=kjl_q, n_components=n_components)
                        #                 for kjl_d, kjl_n, kjl_q, n_components in
                        #                 list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))

                        outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, kjl_d=kjl_d,
                                                                   kjl_n=kjl_n, kjl_q=kjl_q)
                                        for kjl_d, kjl_n, kjl_q in
                                        list(itertools.product(kjl_ds, kjl_ns, kjl_qs)))
                else:
                    msg = params['kjl']
                    raise NotImplementedError(f'Error: kjl={msg}')

            elif params['nystrom']:
                # GMM-gs:True-nystrom:True
                nystrom_ns = params['nystrom_ns']
                nystrom_ds = params['nystrom_ds']
                nystrom_qs = params['nystrom_qs']
                if 'nystrom_ns' in params.keys(): del params['nystrom_ns']
                if 'nystrom_ds' in params.keys(): del params['nystrom_ds']
                if 'nystrom_qs' in params.keys(): del params['nystrom_qs']
                with parallel:
                    outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, nystrom_n=nystrom_n,
                                                               nystrom_d=nystrom_d, nystrom_q=nystrom_q,
                                                               n_components=n_components) for
                                    nystrom_n, nystrom_d, nystrom_q, n_components in
                                    list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, n_components_arr)))
            else:
                msg = params['kjl']
                raise NotImplementedError(f'Error: kjl={msg}')
        else:
            msg = params['gs']
            raise NotImplementedError(f'Error: gs={msg}')

    elif params['detector_name'] == 'OCSVM':
        if params['gs']:
            if params['kjl']:
                msg = params['kjl']
                raise NotImplementedError(f'Error: kjl={msg}')
            else:  # gs=True, kjl = False and for OCSVM
                with parallel:
                    model_qs = params['model_qs']
                    model_nus = params['model_nus']
                    if 'model_qs' in params.keys(): del params['model_qs']
                    if 'model_nus' in params.keys(): del params['model_nus']

                    outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, model_q=model_q,
                                                               model_nu=model_nu) for _, _, _, model_q, model_nu in
                                    list(itertools.product([0], [0], [0], model_qs, model_nus)))
        else:
            msg = params['gs']
            raise NotImplementedError(f'Error: gs={msg}')

    else:
        msg = params['detector_name']
        raise NotImplementedError(f'Error: detector_name={msg}')

    # get the best avg auc from n_repeats experiments
    best_avg_auc = -1
    for out in outs:
        if np.mean(out['aucs']) > best_avg_auc:
            best_avg_auc = np.mean(out['aucs'])
            best_results = copy.deepcopy(out)

    # it's better to save all middle results too
    middle_results = outs

    print('---get accurate time of training and testing with the best params---')
    best_params = best_results['params']
    out = _model_train_test(normal_data, abnormal_data, params=best_params)
    # double check the results if you can
    # assert best_avg_auc == np.mean(out['aucs'])
    # print(best_avg_auc, np.mean(out['aucs']), best_results, out)
    best_results = out

    return best_results, middle_results


@execute_time
def main(random_state, n_jobs=-1, n_repeats=1):
    datasets = [
        #     # # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',
        'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
        #     # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
        #     # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
        #     # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
        #     # # # # #
        #     # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
        #     # # # # # #
        #     'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
        # 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196',
        # #     # # #
        # #     # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        # #     # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        # 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
        # #     # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
        # #     # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
        # #     # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
        # #     # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
        # #     # #
        # #     # # #
        # #     # 'WRCCDC/2020-03-20',
        # #     # 'DEFCON/ctf26',
        # 'ISTS/2015',
        # 'MACCDC/2012',
        #
        # #     # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
        # 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
        # #     # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
        # #     # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
        # #

    ]
    feat_set = 'iat_size'
    header = False

    in_dir = 'data/data_kjl'
    out_dir = 'out/data_kjl'
    results = {}
    for data_name in datasets:
        # 1. get data params
        in_expand_dir = pt.join(in_dir, data_name, feat_set, f'header:{header}')
        out_expand_dir = pt.join(out_dir, data_name, feat_set, f'header:{header}')
        normal_file = f'{in_expand_dir}/normal.dat'
        abnormal_file = f'{in_expand_dir}/abnormal.dat'
        print(normal_file, abnormal_file)

        if not pt.exists(normal_file) or not pt.exists(abnormal_file):
            _normal_file = pt.splitext(normal_file)[0] + '.csv'
            _abnormal_file = pt.splitext(abnormal_file)[0] + '.csv'
            # extract data from csv file
            normal_data, abnormal_data = extract_data(_normal_file, _abnormal_file,
                                                      meta_data={'idxs_feat': [0, -1],
                                                                 'train_size': -1,
                                                                 'test_size': -1})
            # transform data format
            dump_data(normal_data, normal_file)
            dump_data(abnormal_data, abnormal_file)
        else:
            normal_data = load_data(normal_file)
            abnormal_data = load_data(abnormal_file)

        # try to make case more smaller and specific
        gs = True

        cases = [  # OCSVM-gs:True
            # {'detector_name': 'OCSVM', 'gs': gs},

            # # GMM-gs:True
            {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs},
            # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs},
            #
            # # # GMM-gs:True-kjl:True
            # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True},
            # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': True},
            # #
            # # GMM-gs:True-kjl:True-nystrom:True   # nystrom will take more time than kjl
            # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': False, 'nystrom': True},
            # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': False, 'nystrom': True},
            #
            # # GMM-gs:True-kjl:True-quickshift:True
            # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True, 'quickshift': True},
            # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': True, 'quickshift': True},
            # #
            # # GMM-gs:True-kjl:True-meanshift:True
            # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True, 'meanshift': True},
            # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': True, 'meanshift': True},
        ]

        for case in cases:
            case['n_repeats'] = n_repeats
            case['n_jobs'] = n_jobs
            if 'gs' not in case.keys():
                case['gs'] = False
            else:
                case['gs_search_type'] = 'grid_search'
                if case['detector_name'] == 'OCSVM':
                    case['model_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                        0.95]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                    case['model_nus'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                elif case['detector_name'] == 'GMM':
                    case['n_components'] = [6]  # [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
                    case['online_gmm'] = True
                else:
                    raise ValueError(f'Error: {case}')

            if 'kjl' not in case.keys():
                case['kjl'] = False
            else:
                if case['kjl']:
                    case['kjl_ns'] = [100]
                    case['kjl_ds'] = [10]
                    case['kjl_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                      0.95]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

            if 'quickshift' not in case.keys():
                case['quickshift'] = False
            else:
                case['quickshift_ks'] = [100, 1000]
                case['quickshift_betas'] = [0.1, 0.3, 0.5, 0.7, 0.9]

            if 'meanshift' not in case.keys():
                case['meanshift'] = False
            else:
                case['meanshift_qs'] = ''  # use the same qs of kjl_qs

            if 'nystrom' not in case.keys():
                case['nystrom'] = False
            else:
                if case['nystrom']:
                    case['nystrom_ns'] = [100]
                    case['nystrom_ds'] = [10]
                    case['nystrom_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                          0.95]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

            case_str = '-'.join([f'{k}_{v}' for k, v in case.items() if
                                 k in ['detector_name', 'covariance_type', 'gs', 'kjl', 'nystrom', 'quickshift',
                                       'meanshift']])
            try:
                # get each result
                print(f"\n\n\n***{case}***, {case_str}")
                _best_results, _middle_results = get_best_results(normal_data, abnormal_data, case, random_state)

                # save each result first
                out_file = pt.abspath(f'{out_expand_dir}/{case_str}.csv')
                print('+++', out_file)
                save_each_result(_best_results, case_str, out_file)

                dump_data(_middle_results, out_file + '-middle_results.dat')

                results[(in_expand_dir, case_str)] = (_best_results, _middle_results)
            except Exception as e:
                traceback.print_exc()
                print(f"some error exists in {case}")

    # save results first
    out_file = f'{out_dir}/all_results.csv'
    print(f'\n\n***{out_file}***')
    # Todo: format the results
    save_result(results, out_file)
    print("\n\n---finish succeeded!")


if __name__ == '__main__':
    main(random_state=RANDOM_STATE, n_jobs=1, n_repeats=1)
