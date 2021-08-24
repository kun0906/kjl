"""Main entrance of online GMM experiments

Run the below command under "applications/"
    PYTHONPATH=../:./ python3.7 main_online_gmm.py > out/main_online_gmm.txt 2>&1 &
"""

import copy
import itertools
import os
import os.path as pth
import traceback
from datetime import datetime

import numpy as np
from joblib import delayed, Parallel
from sklearn import metrics
from sklearn.metrics import pairwise_distances, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kjl.models._base import BASE_MODEL
from kjl.models.gmm import GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.models.online_gmm import ONLINE_GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.models.kjl import kernelJLInitialize, getGaussianGram, KJL
from kjl.models.nystrom import nystromInitialize, NYSTROM
from kjl.models.ocsvm import OCSVM
from kjl.models.standardization import STD
from kjl.utils.data import data_info, split_train_test, load_data, extract_data, dump_data, save_each_result, \
    save_result, batch, _get_line
from kjl.utils.utils import execute_time, func_running_time

RANDOM_STATE = 42

print('PYTHONPATH: ', os.environ['PYTHONPATH'])


class BASE_MODEL():

    def get_best_model(self, X_train, y_train, X_test, y_test):

        self.init_X_train = X_train
        self.init_y_train = y_train

        case = 'kjl-gmm_full'
        if case == 'kjl-gmm_full':
            params = {}
            # params['qs_']   # quickshift++
            params['n_components'] = [1, 2]
            params['kjl_ns'] = [100]
            params['kjl_ds'] = [10]
            params['kjl_qs'] = [0.1, 0.2]

        # self.params_copy = copy(self.params)
        best_auc = -1
        for n_components, kjl_d, kjl_n, kjl_q in list(
                itertools.product(params['n_components'], params['kjl_ds'], params['kjl_ns'], params['kjl_qs'])):
            self.init_params = {'n_components': n_components, 'kjl_d': kjl_d, 'kjl_n': kjl_n, 'kjl_q': kjl_q}
            self.params['n_components'] = n_components
            self.params['kjl_d'] = kjl_d
            self.params['kjl_n'] = kjl_n
            self.params['kjl_q'] = kjl_q
            self.init_model = GMM(n_components=n_components)

            # Fit a models on the train set (how to get the best parameters of GMM on the init_set?)
            self._init_train(self.init_model, X_train, y_train)  # update self.init_model
            # Evaluate the models on the test set
            self._init_test(self.init_model, X_test, y_test)
            print(f'auc: {self.auc}')
            # for out in outs:
            if best_auc <= self.auc:
                best_auc = self.auc
                best_params = self.model.get_params()
                self.params_copy = copy.deepcopy(self.params)

        # get best models with best_params
        self.params = self.params_copy
        self.init_model = GMM()
        self.init_model.set_params(**best_params)

        # Fit a models on the train set (how to get the best parameters of GMM on the init_set?)
        self._init_train(self.init_model, X_train, y_train)  # update self.init_model
        # Evaluate the models on the test set
        self._init_test(self.init_model, X_test, y_test)
        print(f'init_train_time: {self.train_time}, init_test_time: {self.test_time}, init_auc: {self.auc}')

        ##########################################################################################
        # online train
        y_train_score = self.init_model.decision_function(X_train)
        self.abnormal_thres = np.quantile(y_train_score, q=0.99)  # abnormal threshold
        self.novelty_thres = np.quantile(y_train_score, q=0.85)  # normal threshold
        print(f'novelty_thres: {self.novelty_thres}, abnormal_thres: {self.abnormal_thres}')

        _, self.model.log_resp = self.model._e_step(X_train)
        self.model.n_samples = X_train.shape[0]
        self.model.X_train = X_train

        return self.init_model

    def _init_train(self, model, X_train, y_train=None):
        """Train models on the initial set (init_set)

        Parameters
        ----------
        model: models instance

        X_train: array
            n_samples, n_feats = size(X_train)
            Its shape is (n_samples, n_feats)

        y_train: array
            Its shape is (n_samples, )

        Returns
        -------
            self:

        """

        self.train_time = 0

        ##########################################################################################
        # Step 1: Preprocessing the data, which includes standarization, mode seeking, and kernel projection.
        # Step 1.1: Standardize the data first
        self.std_inst = STD()
        _, std_train_time = func_running_time(self.std_inst.fit, X_train)
        self.train_time += std_train_time
        X_train, std_train_time = func_running_time(self.std_inst.transform, X_train)
        self.train_time += std_train_time

        # Step 1.2: Seek modes of the data by quickshift++ or meanshift
        self.thres_n = 100  # used to filter clusters which have less than 100 datapoints
        if 'meanshift' in self.params.keys() and self.params['meanshift']:
            dists = pairwise_distances(X_train)
            self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
            self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                X_train, bandwidth=self.sigma, thres_n=self.thres_n)
            self.params['n_components'] = self.n_components
        elif 'quickshift' in self.params.keys() and self.params['quickshift']:
            self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                thres_n=self.thres_n)
            self.params['n_components'] = self.n_components
        else:
            self.seek_train_time = 0
        self.train_time += self.seek_train_time

        # Step 1.3. Project the data onto a lower space with KJL or Nystrom
        proj_train_time = 0.0
        if 'kjl' in self.params.keys() and self.params['kjl']:
            # X_train, self.U_kjl, self.Xrow_kjl, self.sigma_kjl, self.random_matrix, self.A = self.train_project_kjl(
            #     X_train,
            #     kjl_params=self.params,
            #     debug=self.debug)
            self.kjl_inst = KJL(self.params)
            _, kjl_train_time = func_running_time(self.kjl_inst.fit, X_train)
            proj_train_time += kjl_train_time
            X_train, kjl_train_time = func_running_time(self.kjl_inst.transform, X_train)
            proj_train_time += kjl_train_time
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            # X_train, self.subX_nystrom, self.sigma_nystrom, self.Eigvec_nystrom, self.Lambda_nystrom = \
            #     self.train_project_nystrom(X_train, nystrom_params=self.params, debug=self.debug)
            # self.proj_train_time = self.nystrom_train_time

            self.nystrom_inst = NYSTROM()
            _, nystrom_train_time = func_running_time(self.nystrom_inst.fit, X_train)
            proj_train_time += nystrom_train_time
            X_train, nystrom_train_time = func_running_time(self.nystrom_inst.transform, X_train)
            proj_train_time += nystrom_train_time
        else:
            proj_train_time = 0.0
        self.train_time += proj_train_time

        ##########################################################################################
        # Step 2: fit a models
        model_params = {'n_components': self.params['n_components'],
                        'covariance_type': self.params['covariance_type'],
                        'means_init': None, 'random_state': self.random_state}
        # set models default parameters
        model.set_params(**model_params)
        if self.verbose > 5: print(model.get_params())
        # train models
        self.model, model_train_time = func_running_time(model.fit, X_train)
        self.train_time += model_train_time

        return self

    def _init_test(self, model, X_test, y_test):
        """Evaluate the models on the set set

        Parameters
        ----------
        model:
            a fitted models on the train set

        X_test: array
            n_samples, n_feats = size(X_test)
            Its shape is (n_samples, n_feats)

        y_test: array
            Its shape is (n_samples, )

        Returns
        -------
            self:
        """

        self.test_time = 0

        ##########################################################################################
        # Step 1: Preprocessing
        # Step 1.1: standardize the data first
        # start = datetime.now()
        # X_test = self.scaler.transform(X_test)
        # end = datetime.now()
        # self.std_test_time = (end - start).total_seconds()
        X_test, std_test_time = func_running_time(self.std_inst.transform, X_test)
        self.test_time += std_test_time

        # Step 1.2: seek modes of the data by quickshift++ or meanshift
        seek_test_time = 0
        self.test_time += seek_test_time

        # Step 1.3: project the data onto a lower space with KJL or Nystrom
        proj_test_time = 0.0
        if 'kjl' in self.params.keys() and self.params['kjl']:
            # X_test = self.project_X_with_kjl(X_test, self.U_kjl, self.Xrow_kjl, self.sigma_kjl,
            #                                  kjl_params=self.params, debug=self.debug)
            # self.proj_test_time = self.kjl_test_time
            X_test, proj_test_time = func_running_time(self.kjl_inst.transform, X_test)

        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            # X_test = self.project_X_with_nystrom(X_test, self.subX_nystrom, self.sigma_nystrom, self.Eigvec_nystrom,
            #                                      self.Lambda_nystrom, nystrom_params=self.params, debug=self.debug)
            # self.proj_test_time = self.nystrom_test_time
            X_test, proj_test_time = func_running_time(self.nystrom_inst.transform, X_test)
        else:
            proj_test_time = 0
        self.test_time += proj_test_time

        ##########################################################################################
        # Step 2: Evaluate GMM on the test set
        y_score, model_test_time = func_running_time(model.decision_function, X_test)
        self.test_time += model_test_time
        self.auc, _ = func_running_time(self.get_score, y_test, y_score)

        print(f'Total test time: {self.test_time} <= std_test_time: {std_test_time}, '
              f'seek_test_time: {seek_test_time}'
              f', proj_test_time: {proj_test_time}, '
              f'model_test_time: {model_test_time}')

        return self

    def get_score(self, y_test, y_score):
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return auc


class ONLINE_GMM_MAIN(BASE_MODEL, ONLINE_GMM):

    def __init__(self, params):
        """Main class of online GMM experiment

        Parameters
        ----------
        params
        """
        self.params = params

        self.random_state = params['random_state']
        self.verbose = params['verbose']

        # stores important results
        self.info = {}

    def train_test_model(self, X_train, y_train, X_test, y_test):
        """ Train and test models

        Parameters
        ----------
        X_train: array
            n_samples, n_feats = size(X_train)
            Its shape is (n_samples, n_feats)
        y_train: array
            Its shape is (n_samples, )

        X_test: array
            n_samples, n_feats = size(X_test)
            Its shape is (n_samples, n_feats)

        y_test: array
            Its shape is (n_samples, )

        Returns
        -------
            self:

        """
        ##########################################################################################
        # Step 1. Split train set into two subsets: initial set (init_set) and new arrival set (arvl_set)
        # with ratio 3:7.
        X_train, X_arrival, y_train, y_arrival = train_test_split(X_train, y_train, test_size=0.7,
                                                                  random_state=self.random_state)
        self.n_samples, self.n_feats = X_train.shape

        ##########################################################################################
        # # Step 2. Get initial models (init_model) on initial set (init_set) and evaluate it on test set
        # self.init_model = GMM()
        # # Fit a models on the train set (how to get the best parameters of GMM on the init_set?)
        # self._init_train(self.init_model, X_train, y_train) # update self.init_model
        # # Evaluate the models on the test set
        # self._init_test(self.init_model, X_test, y_test)
        # print(f'init_train_time: {self.train_time}, init_test_time: {self.test_time}, init_auc: {self.auc}')
        # self.init_params = {}
        self.init_model = self.get_best_model(X_train, y_train, X_test, y_test)

        ##########################################################################################
        # Step 3. Online train and evaluate models
        online_train_times = []
        online_test_times = []
        online_aucs = []
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=10)):
            if i == 0:
                self.model = self.init_model

            # online train models (update GMM models values, such as, means, covariances, kjl_U, and n_components)
            self._online_train(X_batch, Y_batch)  # update self.models
            online_train_times.append(self.train_time)

            # online test models
            self._online_test(X_test, y_test)
            online_test_times.append(self.test_time)
            online_aucs.append(self.auc)

            if self.verbose > 5:
                print(f'***batch: {i + 1}, train_time: {self.train_time}, test_time: {self.test_time}, auc: {self.auc}')

        self.info['train_times'] = online_train_times
        self.info['test_times'] = online_test_times
        self.info['aucs'] = online_aucs
        self.info['params'] = {}

        # info = {'train_times': [0.0], 'test_times': [0.0], 'aucs': [0.0], 'apcs': '',
        #                 'params': params,
        #                 'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}

        return self

    def _online_train(self, X_batch, y_batch=None):
        """Online train the models: using the X_batch to retrain and update the current models incrementally.

        Parameters
        ----------
        models: models instance
            a fitted models on the init_set

        X_batch: array
            n_samples, n_feats = size(X_batch)
            Its shape is (n_samples, n_feats)

        y_batch: array
            Its shape is (n_samples, )

        Returns
        -------
           y_score: abnormal score
           testing_time, auc, apc
        """
        ##########################################################################################
        # Step 1: predict if each datapoint is normal or not.
        # new_model will retrain on each new datapoint (x) with the previous parameters of the current models (models)

        # In each time, we only process one datapoint (x)
        start_0 = datetime.now()
        for i, x in enumerate(X_batch):
            x = x.reshape(1, -1)

            model_online_train_time = 0

            ##########################################################################################
            ### Step 1: Preprocessing
            # Step 1.1: standardization
            # a) transform x
            new_x = self.std_inst.transform(x)
            # b) update the mean and scaler of self.std_inst with 'x'
            self.std_inst.update(x)
            x = new_x

            # Step 1.2: Seek the modes by quickshift++ or meanshift
            seek_online_train_time = 0
            model_online_train_time += seek_online_train_time

            # Step 3: Project the data onto a lower space with kjl or nystrom
            if 'kjl' in self.params.keys() and self.params['kjl']:
                new_x = np.copy(x)
                # a) Project the data
                new_x = self.kjl_inst.transform(new_x)
                proj_online_train_time = self.kjl_inst.kjl_test_time
                # b) Update kjl: self.U_kjl, self.Xrow_kjl.
                self.kjl_inst.update(x)
                x = new_x

            elif 'nystrom' in self.params.keys() and self.params['nystrom']:
                new_x = np.copy(x)
                # a) Project the data
                new_x = self.nystrom_inst.transform(new_x)
                proj_online_train_time = self.nystrom_train_time
                # b) Update nystrom_inst
                self.nystrom_inst.update(x)
                x = new_x

            else:
                proj_online_train_time = 0
            model_online_train_time += proj_online_train_time

            ##########################################################################################
            # Step 2: Obtain the abnormal score
            # For inlier, a small value is used; a larger value is for outlier (positive)
            # here the output should be the abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
            y_score, testing_time = func_running_time(self.model.decision_function, x)
            print("i:{}, online models prediction takes {} seconds, y_score: {}".format(i, testing_time, y_score))
            model_test_time = testing_time

            model_online_train_time += model_test_time
            # if self.verbose > 5:
            #     print(f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, '
            #           f'seek_test_time: {self.seek_test_time}'
            #     f', proj_test_time: {self.proj_test_time}, '
            #     f'model_test_time: {self.model_test_time}')

            ##########################################################################################
            # Step 3. online train a GMM based on the previous GMM (such as means, covariances)
            # and use it to replace the previous one.
            new_model = ONLINE_GMM(n_components=model.n_components, weights_init=model.weights_,
                                   means_init=model.means_, covariances_init=model.covariances_)
            # new_model.n_components = models.n_components
            # new_model.weights_ = models.weights_
            # new_model.means_ = models.means_
            # new_model.covariances_ = models.covariances_
            # new_model.log_resp = models.log_resp
            new_model.warm_start = True
            new_model.n_samples = model.n_samples
            # online update the models
            model = self._online_update(new_model, y_score, x, lower_bound=-np.infty, to_convergent=False)

        end_0 = datetime.now()
        model_online_train_time = (end_0 - start_0).total_seconds()
        print(f'Total test time: {model_online_train_time}')

        return model, model_online_train_time

    def _online_test(self, X_test, y_test):
        """Evaluate models on the test set

        Parameters
        ----------
        ----------
        models: models instance
            a fitted models on the init_set

        X_test: array
            n_samples, n_feats = size(X_test)
            Its shape is (n_samples, n_feats)

        y_test: array
            Its shape is (n_samples, )

        Returns
        -------

        """
        self.test_time = 0

        ##########################################################################################
        # Step 1: preprocessing, which includes standardization, mode seeking and projection
        # Step 1.1: standardization
        X_test, std_test_time = func_running_time(self.std_inst.transform, X_test)
        self.test_time += std_test_time

        # seek_test_time
        seek_test_time = 0
        self.test_time += seek_test_time

        # project time
        if 'kjl' in self.params.keys() and self.params['kjl']:
            X_test, proj_test_time = func_running_time(self.kjl_inst.transform, X_test)

        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            X_tesX_test, proj_test_time = func_running_time(self.nystrom_inst.transform, X_test)
        else:
            proj_test_time = 0
        self.test_time += proj_test_time

        # evaluate GMM on the test set
        # self.y_score, model_test_time, self.auc = self._test(self.models, X_test, y_test)
        y_score, model_test_time = func_running_time(model.decision_function, X_test)
        self.auc, _ = func_running_time(self.get_score, y_test, y_score)
        self.test_time += model_test_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {std_test_time}, seek_test_time: {seek_test_time}'
            f', proj_test_time: {proj_test_time}, '
            f'model_test_time: {model_test_time}')

        return self

    def _online_update(self, new_model, _y_score, x, lower_bound=-np.infty, to_convergent=False):
        if _y_score < self.abnormal_thres:
            # According to the _y_score, the x is predicted as a normal datapoint.

            new_model.n_samples += 1

            # two sub-scenarios:
            # 1): just update the previous components
            # 2): create a new component for the new x
            if _y_score < self.novelty_thres:
                # the x is predicted as a normal point, so we just need to update the previous components
                if to_convergent:
                    # get log_prob and resp
                    log_prob_norm, log_resp = new_model._e_step(x)
                    new_model.log_resp = log_resp
                    # use m_step to update params: weights (i.e., mixing coefficients), means, and covariances with x and
                    # the previous params: log_resp (the log probability of each component), means and covariances
                    new_model._m_step(x, new_model.log_resp,
                                      new_model.n_samples - 1)  # update mean, covariance and weight

                else:
                    for n_iter in range(1, new_model.max_iter + 1):
                        prev_lower_bound = lower_bound

                        # get log_prob and resp
                        log_prob_norm, log_resp = new_model._e_step(x)
                        new_model.log_resp = log_resp

                        # use m_step to update params: weights (i.e., mixing coefficients), means, and covariances with x and
                        # the previous params: log_resp (the log probability of each component), means and covariances
                        new_model._m_step(x, new_model.log_resp,
                                          new_model.n_samples - 1)  # update mean, covariance and weight

                        # should be reconsidered again?
                        # get the difference
                        lower_bound = new_model._compute_lower_bound(log_resp, log_prob_norm)
                        change = lower_bound - prev_lower_bound
                        if abs(change) < new_model.tol:
                            self.converged_ = True
                            print(f'n_iter: {n_iter}')
                            break
            else:  # _y_score >= self.novelty_thres:
                # self.novelty_thres < _y_score < self.abnormal_thres
                # x is predicted as a novelty datapoint (but still is a normal datapoint), so we create a new
                # component and update GMM.

                new_model.n_components += 1
                # # assign the x to a new class and expand the previous "log_resp", which is used to obtain
                # # the "weights" of each component.
                # new_model.log_resp = np.concatenate([models.log_resp, np.zeros((models.n_samples, 1))], axis=1)
                log_resp = np.zeros((1, new_model.n_components))
                log_resp[-1] = 1
                # new_model.log_resp = np.concatenate([new_model.log_resp, log_resp], axis=0)
                new_model.log_resp = log_resp
                new_model.weights_ = np.concatenate([new_model.weights_, np.zeros((1,)) + 1e-6], axis=0)

                # compute the mean and covariance of the new components
                # For the mean, we use the x value as the mean of the new component
                # (because the new component only has one point (i.e., x)), and append it to the previous means.
                new_mean = x
                new_covar = self.generate_new_covariance(x, new_model.means_, new_model.covariances_)
                new_model.means_ = np.concatenate([new_model.means_, new_mean], axis=0)
                _, dim = new_mean.shape
                new_model.covariances_ = np.concatenate([new_model.covariances_,
                                                         new_covar.reshape(1, dim, dim)], axis=0)

                print(f'new_model.params: {new_model.get_params()}')

                if to_convergent:
                    pass
                else:
                    # train the new models on x, update params, and use the new models to update the previous models
                    for n_iter in range(1, new_model.max_iter + 1):
                        # for n_iter in range(1, self.max_iter ):
                        # convergence conditions: check if self.max_iter or self.tol exceeds the preset value.
                        prev_lower_bound = lower_bound

                        # use m_step to update params: weights, means, and covariances with x and the initialized
                        # params: log_resp, means and covariances
                        new_model._m_step(x, new_model.log_resp,
                                          new_model.n_samples - 1)  # update weight, means, covariances

                        # get log_prob and resp
                        log_prob_norm, log_resp = new_model._e_step(x)
                        new_model.log_resp = log_resp

                        # get the difference
                        lower_bound = new_model._compute_lower_bound(log_resp, log_prob_norm)
                        change = lower_bound - prev_lower_bound
                        if abs(change) < new_model.tol:
                            self.converged_ = True
                            print(f'n_iter: {n_iter}')
                            break

            # update the current models with the new_model
            model = new_model
        else:
            # if _y_score >= self.abnormal_thres, the x is predicted as a abnormal flow, so we should drop it.
            print('this flow is an abnormal flow, so we drop it.')
            model = new_model

        return model

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


class BATCH_GMM_MAIN(BASE_MODEL):

    def __init__(self, params):
        """Main class of online GMM experiment

        Parameters
        ----------
        params: dict
            the parameters for this experiment
        """
        self.params = params

        self.random_state = params['random_state']
        self.verbose = params['verbose']

        # stores important results
        self.info = {}

    def train_test_model(self, X_train, y_train, X_test, y_test):
        """ Train and test models

        Parameters
        ----------
        X_train: array
            n_samples, n_feats = size(X_train)
            Its shape is (n_samples, n_feats)
        y_train: array
            Its shape is (n_samples, )

        X_test: array
            n_samples, n_feats = size(X_test)
            Its shape is (n_samples, n_feats)

        y_test: array
            Its shape is (n_samples, )

        Returns
        -------
            self:

        """
        ##########################################################################################
        # Step 1. Split train set into two subsets: initial set (init_set) and new arrival set (arvl_set)
        # with ratio 3:7.
        X_train, X_arrival, y_train, y_arrival = train_test_split(X_train, y_train, test_size=0.7,
                                                                  random_state=self.random_state)
        self.n_samples, self.n_feats = X_train.shape

        ##########################################################################################
        # # Step 2. Get initial models (init_model) on initial set (init_set) and evaluate it on test set
        # self.init_model = GMM()
        # # Fit a models on the train set (how to get the best parameters of GMM on the init_set?)
        # self._init_train(self.init_model, X_train, y_train) # update self.init_model
        # # Evaluate the models on the test set
        # self._init_test(self.init_model, X_test, y_test)
        # print(f'init_train_time: {self.train_time}, init_test_time: {self.test_time}, init_auc: {self.auc}')
        # self.init_params = {}
        self.init_model = self.get_best_model(X_train, y_train, X_test, y_test)

        ##########################################################################################
        # Step 3. train the models on the batch data (previous+batch) and evaluate it.
        batch_train_times = []
        batch_test_times = []
        batch_aucs = []
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=1000)):
            if i == 0:
                self.model = self.init_model

            self._batch_train(X_batch, Y_batch)
            batch_train_times.append(self.train_time)

            # batch test models
            self._batch_test(X_test, y_test)
            batch_test_times.append(self.test_time)
            batch_aucs.append(self.auc)

            if self.verbose > 5:
                print(f'***batch: {i + 1}, train_time: {self.train_time}, test_time: {self.test_time}, auc: {self.auc}')

        self.info['train_times'] = batch_train_times
        self.info['test_times'] = batch_test_times
        self.info['aucs'] = batch_aucs

        return self

    def _batch_train(self, X_batch, y_batch=None):

        """batch train the models: using the (X_batch + previous data) to retrain a new models
        and use it to update the current models.

        Parameters
        ----------
        models: models instance
            a fitted models on the init_set, which is used to predict a new datatpoint is normal or not.

        X_batch: array
            n_samples, n_feats = size(X_batch)
            Its shape is (n_samples, n_feats)

        y_batch: array
            Its shape is (n_samples, )

        Returns
        -------
           self
        """

        start_0 = datetime.now()
        model_train_time = 0
        X_batch_copy = copy.deepcopy(X_batch)

        ##########################################################################################
        # U the models to predict X_batch first, and according to the result,
        # only the normal data will be incorporated with previous data to train a new models instead of the current one.

        ### Step 1: Preprocessing
        # Step 1.1: standardization
        # a) transform X_batch
        X_batch = self.std_inst.transform(X_batch)

        # Step 1.2: Seek the modes by quickshift++ or meanshift
        seek_train_time = 0
        model_train_time += seek_train_time

        # Step 3: Project the data onto a lower space with kjl or nystrom
        if 'kjl' in self.params.keys() and self.params['kjl']:
            # Project the data
            X_batch, proj_train_time = func_running_time(self.kjl_inst.transform, X_batch)
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            # Project the data
            X_batch, proj_train_time = func_running_time(self.nystrom_inst.transform, X_batch)
        else:
            proj_train_time = 0
        model_train_time += proj_train_time

        ##########################################################################################
        # Step 2: Obtain the abnormal score
        # For inlier, a small value is used; a larger value is for outlier (positive)
        # here the output should be the abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
        y_score, model_predict_time = func_running_time(self.model.decision_function, X_batch)
        # print("i:{}, batch models prediction takes {} seconds, y_score: {}".format(0, testing_time, y_score))

        model_train_time += model_predict_time
        # if self.verbose > 5:
        #     print(f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, '
        #           f'seek_test_time: {self.seek_test_time}'
        #     f', proj_test_time: {self.proj_test_time}, '
        #     f'model_test_time: {self.model_test_time}')

        ##########################################################################################
        # Step 3. Batch train a GMM from scratch
        # and use it to replace the previous one.
        # Step 3.1: concatenate previous data and X_batch (only normal data)
        new_x = []
        abnormal_cnt = 0
        for x, y in zip(X_batch_copy, y_score):
            if y < self.abnormal_thres:
                new_x.append(x)
            else:
                abnormal_cnt += 1
        new_x = np.asarray(new_x)
        self.init_X_train = np.concatenate([self.init_X_train, new_x], axis=0)
        self.init_y_train = np.zeros((self.init_X_train.shape[0],))  # normal is '0'

        # Step 3.2: train a new models and use it to instead of the current one.
        # we use models.n_compoents to initialize n_components or the value found by quickhsift++ or meanshift
        new_model = GMM(n_components=self.model.n_components)
        self._init_train(new_model, self.init_X_train, self.init_y_train)  # self.train_time

        # the datapoint is predicted as a abnormal flow, so we should drop it.
        print(f'{abnormal_cnt} flows are predicted as abnormal, so we drop them.')

        end_0 = datetime.now()
        model_batch_train_time = (end_0 - start_0).total_seconds()
        print(f'Total batch time: {model_batch_train_time}')

        # replace the current models with new_model
        self.model = new_model

        return self

    def _batch_test(self, X_batch, y_batch):
        self._init_test(self.model, X_batch, y_batch)


# def _model_train_test(normal_data, abnormal_data, params, **kargs):
#     """Get result with one of combinations of params
#
#     Parameters
#     ----------
#     normal_data: array with shape (n_samples, n_feats)
#
#     abnormal_data:  array with shape (n_samples, n_feats)
#
#     params: dict
#         the configuation of parameters for this experiment, such as, case, models parameters, and parallel.
#
#     kargs: dict
#         not used yet.
#
#     Returns
#     -------
#     info: dict
#         the experiment result
#     """
#     try:
#         for k, v in kargs.items():
#             params[k] = v
#
#         n_repeats = params['n_repeats']
#         train_times = []
#         test_times = []
#         aucs = []
#         # keep that all the algorithms have the same input data by setting random_state
#         for i in range(n_repeats):
#             print(f"\n\n==={i + 1}/{n_repeats}(n_repeats): {params}===")
#             X_train, y_train, X_test, y_test = split_train_test(normal_data, abnormal_data,
#                                                                 train_size=0.8, test_size=-1, random_state=i * 100)
#
#             if "GMM" in params['detector_name']:
#                 if params['online_gmm']:
#                     models = ONLINE_GMM_MAIN(params)
#                 # elif params['offline_gmm']:
#                 #     models = OFFLINE_GMM_MAIN(params)
#                 # else:
#                 #     models = GMM_MAIN(params)
#             # elif "OCSVM" in params['detector_name']:
#             #     models = OCSVM_MAIN(params)
#
#             models.train_test_model(X_train, y_train, X_test, y_test)
#
#             train_times.append(models.train_time)
#             test_times.append(models.test_time)
#             aucs.append(models.auc)
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
# def get_best_results(normal_data, abnormal_data, case, random_state=42):
#     """ Tuning hyperparameters to find the best result
#
#     Parameters
#     ----------
#     normal_data: array with shape (n_normal_samples, n_feats)
#
#     abnormal_data: array with shape (n_abnormal_samples, n_feats)
#
#     case:
#         which algorithm we are using
#
#     Returns
#     -------
#         best_results:
#             only the best result with best parameters.
#
#         middle_results:
#             all results with different parameters.
#     """
#     params = copy.deepcopy(case)
#     params['random_state'] = random_state
#
#     parallel = Parallel(n_jobs=params['n_jobs'], verbose=30)
#
#     if params['detector_name'] == 'GMM':
#         # GMM with grid search
#         if params['gs']:
#             n_components_arr = params['n_components']
#             if 'n_components' in params.keys(): del params['n_components']
#             if not params['kjl'] and not params['nystrom']:  # only GMM
#                 # GMM-gs:True
#                 with parallel:
#                     outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params,
#                                                                n_components=n_components) for n_components, _ in
#                                     list(itertools.product(n_components_arr, [0])))
#
#             elif params['kjl']:
#                 kjl_ds = params['kjl_ds']
#                 kjl_ns = params['kjl_ns']
#                 kjl_qs = params['kjl_qs']
#                 if 'kjl_ds' in params.keys(): del params['kjl_ds']
#                 if 'kjl_ns' in params.keys(): del params['kjl_ns']
#                 if 'kjl_qs' in params.keys(): del params['kjl_qs']
#                 if not params['quickshift'] and not params['meanshift']:
#                     # GMM-gs:True-kjl:True
#                     with parallel:
#                         outs = parallel(
#                             delayed(_model_train_test)(normal_data, abnormal_data, params, kjl_d=kjl_d, kjl_n=kjl_n,
#                                                        kjl_q=kjl_q,
#                                                        n_components=n_components) for kjl_d, kjl_n, kjl_q, n_components
#                             in
#                             list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))
#
#                 elif params['quickshift']:
#                     # GMM-gs:True-kjl:True-quickshift:True
#                     quickshift_ks = params['quickshift_ks']
#                     quickshift_betas = params['quickshift_betas']
#                     if 'quickshift_ks' in params.keys(): del params['quickshift_ks']
#                     if 'quickshift_betas' in params.keys(): del params['quickshift_betas']
#                     with parallel:
#                         outs = parallel(
#                             delayed(_model_train_test)(normal_data, abnormal_data, params,
#                                                        kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
#                                                        quickshift_k=quickshift_k, quickshift_beta=quickshift_beta)
#                             for kjl_d, kjl_n, kjl_q, quickshift_k, quickshift_beta in
#                             list(itertools.product(kjl_ds, kjl_ns, kjl_qs, quickshift_ks, quickshift_betas)))
#
#                 elif params['meanshift']:
#                     # GMM-gs:True-kjl:True-meanshift:True
#                     meanshift_qs = params[
#                         'meanshift_qs']  # meanshift uses the same kjl_qs, and only needs to tune one of them
#                     with parallel:
#                         # outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, kjl_d=kjl_d,
#                         #                                            kjl_n=kjl_n, kjl_q=kjl_q, n_components=n_components)
#                         #                 for kjl_d, kjl_n, kjl_q, n_components in
#                         #                 list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))
#
#                         outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, kjl_d=kjl_d,
#                                                                    kjl_n=kjl_n, kjl_q=kjl_q)
#                                         for kjl_d, kjl_n, kjl_q in
#                                         list(itertools.product(kjl_ds, kjl_ns, kjl_qs)))
#                 else:
#                     msg = params['kjl']
#                     raise NotImplementedError(f'Error: kjl={msg}')
#
#             elif params['nystrom']:
#                 # GMM-gs:True-nystrom:True
#                 nystrom_ns = params['nystrom_ns']
#                 nystrom_ds = params['nystrom_ds']
#                 nystrom_qs = params['nystrom_qs']
#                 if 'nystrom_ns' in params.keys(): del params['nystrom_ns']
#                 if 'nystrom_ds' in params.keys(): del params['nystrom_ds']
#                 if 'nystrom_qs' in params.keys(): del params['nystrom_qs']
#                 with parallel:
#                     outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, nystrom_n=nystrom_n,
#                                                                nystrom_d=nystrom_d, nystrom_q=nystrom_q,
#                                                                n_components=n_components) for
#                                     nystrom_n, nystrom_d, nystrom_q, n_components in
#                                     list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, n_components_arr)))
#             else:
#                 msg = params['kjl']
#                 raise NotImplementedError(f'Error: kjl={msg}')
#         else:
#             msg = params['gs']
#             raise NotImplementedError(f'Error: gs={msg}')
#
#     elif params['detector_name'] == 'OCSVM':
#         if params['gs']:
#             if params['kjl']:
#                 msg = params['kjl']
#                 raise NotImplementedError(f'Error: kjl={msg}')
#             else:  # gs=True, kjl = False and for OCSVM
#                 with parallel:
#                     model_qs = params['model_qs']
#                     model_nus = params['model_nus']
#                     if 'model_qs' in params.keys(): del params['model_qs']
#                     if 'model_nus' in params.keys(): del params['model_nus']
#
#                     outs = parallel(delayed(_model_train_test)(normal_data, abnormal_data, params, model_q=model_q,
#                                                                model_nu=model_nu) for _, _, _, model_q, model_nu in
#                                     list(itertools.product([0], [0], [0], model_qs, model_nus)))
#         else:
#             msg = params['gs']
#             raise NotImplementedError(f'Error: gs={msg}')
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
#     out = _model_train_test(normal_data, abnormal_data, params=best_params)
#     # double check the results if you can
#     # assert best_avg_auc == np.mean(out['aucs'])
#     # print(best_avg_auc, np.mean(out['aucs']), best_results, out)
#     best_results = out
#
#     return best_results, middle_results

def get_normal_abnormal(normal_file, abnormal_file, random_state=42):
    if not pth.exists(normal_file) or not pth.exists(abnormal_file):
        _normal_file = pth.splitext(normal_file)[0] + '.csv'
        _abnormal_file = pth.splitext(abnormal_file)[0] + '.csv'
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

    return normal_data, abnormal_data


def save_result(result, out_file):
    dump_data(result, pth.splitext(out_file)[0] + '.dat')

    with open(out_file, 'w') as f:
        keys = []
        for (in_dir, case_str), (best_results, middle_results) in result.items():
            if case_str not in keys:
                keys.append(case_str)
        print(keys)

        for key in keys:
            print('\n\n')
            for (in_dir, case_str), (best_results, middle_results) in result.items():
                # print(case_str, key)
                if case_str != key:
                    continue
                data = best_results
                try:
                    aucs = data['aucs']
                    # params = data['params']
                    train_times = data['train_times']
                    test_times = data['test_times']

                    # _prefix, _line, _suffex = _get_line(data, feat_set='iat_size')
                    # line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs: {aucs} with best_params: {params}: {_suffex}'
                    _prefix = ''
                    _line = ''
                    params = ''
                    _suffex = ''

                    aucs_str = "-".join([str(v) for v in aucs])
                    train_times_str = "-".join([str(v) for v in train_times])
                    test_times_str = "-".join([str(v) for v in test_times])

                    line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs:{aucs_str}, train_times:{train_times_str}, test_times:{test_times_str}, with params: {params}: {_suffex}'

                except Exception as e:
                    traceback.print_exc()
                    line = ''
                f.write(line + '\n')
                print(line)
            f.write('\n')


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

    # All experiment cases. Aach case includes online and batch
    gs = True
    cases = [  # OCSVM-gs:True
        # {'detector_name': 'OCSVM', 'gs': gs},

        # # GMM-gs:True
        # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs},
        # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs},
        #
        # # # GMM-gs:True-kjl:True
        {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True},
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
    # each case includes online and batch
    online = True  # online: True, otherwise, batch

    # All the results will be stored in the 'results'
    results = {}

    for data_name in datasets:
        ##########################################################################################
        # Step 1: extract normal and abnormal data from input files.
        in_expand_dir = pth.join(in_dir, data_name, feat_set, f'header:{header}')
        out_expand_dir = pth.join(out_dir, data_name, feat_set, f'header:{header}')
        normal_file = f'{in_expand_dir}/normal.dat'
        abnormal_file = f'{in_expand_dir}/abnormal.dat'
        print(normal_file, abnormal_file)
        normal_data, abnormal_data = get_normal_abnormal(normal_file, abnormal_file, random_state=random_state)

        ##########################################################################################
        # Step 2: conduct experiments for each case
        for case in cases:
            case['random_state'] = random_state
            case['verbose'] = 10
            case['online'] = online  # online: True, otherwise, batch

            case_str = '-'.join([f'{k}_{v}' for k, v in case.items() if
                                 k in ['detector_name', 'covariance_type', 'gs', 'kjl', 'nystrom', 'quickshift',
                                       'meanshift', 'online']])
            try:
                # 3. get each result
                print(f"\n\n\n***{case}***, {case_str}")
                # _best_results, _middle_results = get_best_results(normal_data, abnormal_data, case, random_state)
                X_train, y_train, X_test, y_test = split_train_test(normal_data, abnormal_data, train_size=0.8,
                                                                    test_size=-1, random_state=random_state)

                if 'GMM' == case['detector_name']:
                    if case['online']:
                        model = ONLINE_GMM_MAIN(case)
                    else:  # 'batch', i.e., batch_GMM
                        model = BATCH_GMM_MAIN(case)
                else:
                    raise NotImplementedError()
                model.train_test_model(X_train, y_train, X_test, y_test)
                _best_results = model.info  # models.info['train_times']
                _middle_results = {}

                # # save each result first
                out_file = pth.abspath(f'{out_expand_dir}/{case_str}.csv')
                print('+++', out_file)
                # save_each_result(_best_results, case_str, out_file)
                #
                # dump_data(_middle_results, out_file + '-middle_results.dat')
                #
                results[(in_expand_dir, case_str)] = (_best_results, _middle_results)
            except Exception as e:
                traceback.print_exc()
                print(f"some error exists in {case}")

    ##########################################################################################
    # # 4. save results first
    out_file = f'{out_dir}/all-online_{online}-results.csv'
    print(f'\n\n***{out_file}***')
    # # Todo: format the results
    save_result(results, out_file)
    print("\n\n---finish succeeded!")


if __name__ == '__main__':
    main(random_state=RANDOM_STATE, n_jobs=1, n_repeats=1)
