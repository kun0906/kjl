"""Main entrance of online GMM experiments

Run the below command under "examples/"
    PYTHONPATH=../:./ python3.7 main_online_gmm.py > out/main_online_gmm.txt 2>&1 &
"""

import copy
import itertools
import os
import os.path as pth
import traceback
from datetime import datetime

import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances, roc_curve
from sklearn.model_selection import train_test_split

from kjl.model.gmm import GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.model.kjl import KJL
from kjl.model.nystrom import NYSTROM
from kjl.model.online_gmm import ONLINE_GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.model.standardization import STD
from kjl.utils.data import split_train_test, load_data, extract_data, dump_data, save_result, batch, data_info
from kjl.utils.utils import execute_time, func_running_time

RANDOM_STATE = 42

print('PYTHONPATH: ', os.environ['PYTHONPATH'])


class BASE_MODEL():

    def get_best_model(self, X_train, y_train, X_test, y_test):

        self.init_X_train = X_train
        self.init_y_train = y_train

        ##########################################################################################
        # Step 1: configure case
        # case = 'GMM_full-gs_True-kjl_True-nystrom_False-quickshift_False-meanshift_False'
        if self.params['detector_name'] == 'GMM' and self.params['covariance_type'] == 'diag' and \
                self.params['gs'] and self.params['kjl'] and not self.params['quickshift'] and \
                not self.params['meanshift']:
            params = {}
            # best params for the combination of UNB1 and UNB2
            params['n_components'] = [15]
            params['kjl_qs'] = [0.6]

            # # grid search
            # params['n_components'] =  [1,  5, 10, 15, 20, 25, 30, 35, 40, 45]
            # params['kjl_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

            params['kjl_ns'] = [100]
            params['kjl_ds'] = [10]

        else:
            raise ValueError(self.params)

        ##########################################################################################
        # Step 2: find the best parameters
        best_auc = -1
        for n_components, kjl_d, kjl_n, kjl_q in list(itertools.product(params['n_components'],
                                                                        params['kjl_ds'], params['kjl_ns'],
                                                                        params['kjl_qs'])):
            # self.init_params = {'n_components': n_components, 'kjl_d': kjl_d, 'kjl_n': kjl_n, 'kjl_q': kjl_q}
            self.params['n_components'] = n_components
            self.params['kjl_d'] = kjl_d
            self.params['kjl_n'] = kjl_n
            self.params['kjl_q'] = kjl_q
            self.model = GMM(n_components=n_components)

            # Fit a model on the train set
            self._init_train(X_train, y_train)  # fit self.model with self.params on train set
            # Evaluate the model on the test set
            self._init_test(X_test, y_test)  # get self.auc
            print(f'auc: {self.auc}')
            # for out in outs:
            if best_auc <= self.auc:
                best_auc = self.auc
                best_model_params = copy.deepcopy(self.model.get_params())
                best_params = copy.deepcopy(self.params)

        ##########################################################################################
        print('\n\n******best model')
        # Step 3: get the best model with best_params
        self.params = best_params
        self.model = GMM()
        self.model.set_params(**best_model_params)
        print(f'params: {self.params}')
        print(f'model_params: {self.model.get_params()}')

        # Fit a model on the init set
        self._init_train(X_train, y_train)  # fit self.model
        # Evaluate the model on the test set
        self._init_test(X_test, y_test)
        print(f'***init_train_time: {self.train_time}, init_test_time: {self.test_time}, init_auc: {self.auc},'
              f'novelty_thres: {self.novelty_thres}, abnormal_thres: {self.abnormal_thres}')

        # store all important results
        self.init_info = {'train_time': self.train_time, 'test_time': self.test_time, 'auc': self.auc,
                          'novelty_thres': self.novelty_thres, 'abnormal_thres': self.abnormal_thres,
                          'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape, 'params': best_params,
                          'model_params': self.model.get_params()}
        return self.model

    def _init_train(self, X_train, y_train=None):
        """Train model on the initial set (init_set)

        Parameters
        ----------
        model: model instance

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

        # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
        # self.thres_n = 100  # used to filter clusters which have less than 100 datapoints
        # if 'meanshift' in self.params.keys() and self.params['meanshift']:
        #     dists = pairwise_distances(X_train)
        #     self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
        #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
        #         X_train, bandwidth=self.sigma, thres_n=self.thres_n)
        #     self.params['n_components'] = self.n_components
        # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
        #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
        #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
        #         thres_n=self.thres_n)
        #     self.params['n_components'] = self.n_components
        # else:
        #     self.seek_train_time = 0
        # self.train_time += self.seek_train_time

        # Step 1.3. Project the data onto a lower space with KJL or Nystrom
        proj_train_time = 0.0
        if 'kjl' in self.params.keys() and self.params['kjl']:
            self.kjl_inst = KJL(self.params)
            _, kjl_train_time = func_running_time(self.kjl_inst.fit, X_train)
            proj_train_time += kjl_train_time
            X_train, kjl_train_time = func_running_time(self.kjl_inst.transform, X_train)
            proj_train_time += kjl_train_time
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            self.nystrom_inst = NYSTROM(self.params)
            _, nystrom_train_time = func_running_time(self.nystrom_inst.fit, X_train)
            proj_train_time += nystrom_train_time
            X_train, nystrom_train_time = func_running_time(self.nystrom_inst.transform, X_train)
            proj_train_time += nystrom_train_time
        else:
            proj_train_time = 0.0
        self.train_time += proj_train_time

        ##########################################################################################
        # Step 2: fit a model
        model_params = {'n_components': self.params['n_components'],
                        'covariance_type': self.params['covariance_type'],
                        'means_init': None, 'random_state': self.random_state}
        # set model default parameters
        self.model.set_params(**model_params)
        if self.verbose > 5: print(self.model.get_params())
        # train model
        _, model_fit_time = func_running_time(self.model.fit, X_train)
        self.train_time += model_fit_time

        ##########################################################################################
        # Step 3: get the threshold used to decide if a new flow is normal
        # the following values will be used in the online update phase
        y_score, _ = func_running_time(self.model.decision_function, X_train)
        self.abnormal_thres = np.quantile(y_score, q=self.params['q_abnormal_thres'])  # abnormal threshold
        self.novelty_thres = np.quantile(y_score, q=0.85)  # normal threshold
        print(f'novelty_thres: {self.novelty_thres}, abnormal_thres: {self.abnormal_thres}')
        _, self.model.log_resp = self.model._e_step(X_train)
        self.model.n_samples = X_train.shape[0]
        # self.model.X_train_proj = X_train  # X_train:  after standardization and projection
        self.model.sum_resp = np.sum(np.exp(self.model.log_resp), axis=0)
        self.model.y_score = y_score
        self.X_train_proj = X_train  # self.model.X_train_proj = X_train

        return self

    def _init_test(self, X_test, y_test):
        """Evaluate the model on the set set

        Parameters
        ----------
        model:
            a fitted model on the train set

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
        X_test, std_test_time = func_running_time(self.std_inst.transform, X_test)
        self.test_time += std_test_time

        # Step 1.2: project the data onto a lower space with KJL or Nystrom
        if 'kjl' in self.params.keys() and self.params['kjl']:
            X_test, proj_test_time = func_running_time(self.kjl_inst.transform, X_test)
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            X_test, proj_test_time = func_running_time(self.nystrom_inst.transform, X_test)
        else:
            proj_test_time = 0
        self.test_time += proj_test_time

        ##########################################################################################
        # Step 2: Evaluate GMM on the test set
        y_score, model_test_time = func_running_time(self.model.decision_function, X_test)
        self.test_time += model_test_time
        self.auc, _ = func_running_time(self.get_score, y_test, y_score)

        print(f'Total test time: {self.test_time} <= std_test_time: {std_test_time}, '
              f'proj_test_time: {proj_test_time}, '
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

    def train_test_model(self, X_train, y_train, X_arrival, y_arrival, X_test, y_test):
        """ Train and test model

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
        self.n_samples, self.n_feats = X_train.shape
        ##########################################################################################
        # # Step 1. Get initial model (init_model) on initial set (init_set) and evaluate it on test set
        self.init_model = self.get_best_model(X_train, y_train, X_test, y_test)

        ##########################################################################################
        # Step 2. Online train and evaluate model
        online_train_times = [{'train_time': self.init_info['train_time'],
                               'preprocessing': 0,
                               'first_train': 0,
                               'iteration_train': 0,
                               'rescore': 0}]
        online_test_times = [{'test_time': self.init_info['test_time'],
                              'std_time': 0,
                              'proj_time': 0,
                              'predict_time': 0,
                              'auc_time': 0}]

        online_aucs = [self.init_info['auc']]
        online_novelty_threses = [self.init_info['novelty_thres']]
        online_abnormal_threses = [self.init_info['abnormal_thres']]
        online_model_params = [self.init_info['model_params']]
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=self.params['batch_size'])):
            print(f'\n***batch: {i + 1}')
            if i == 0:
                self.model = copy.deepcopy(self.init_model)
                self.acculumated_X_train_proj = self.X_train_proj

            # online train model (update GMM model values, such as, means, covariances, kjl_U, and n_components)
            self._online_train(X_batch, Y_batch)  # update self.model
            online_train_times.append(self.train_time)
            online_novelty_threses.append(self.novelty_thres)
            online_abnormal_threses.append(self.abnormal_thres)
            online_model_params.append(self.model.get_params())

            # online test model
            self._online_test(X_test, y_test)
            online_test_times.append(self.test_time)
            online_aucs.append(self.auc)

            if self.verbose > 5:
                print(f'***batch: {i + 1}, online_train_time: {self.train_time}, '
                      f'test_time: {self.test_time}, auc: {self.auc}')

        self.info['train_times'] = online_train_times
        self.info['test_times'] = online_test_times
        self.info['aucs'] = online_aucs
        self.info['novelty_threses'] = online_novelty_threses
        self.info['abnormal_threses'] = online_abnormal_threses
        self.info['model_params'] = online_model_params
        self.info['params'] = {}

        return self

    def _online_train(self, X_batch, y_batch=None):
        """Online train the model: using the X_batch to retrain and update the current model incrementally.

        Parameters
        ----------
        model: model instance
            a fitted model on the init_set

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

        start_0 = datetime.now()
        model_train_time = 0
        # X_batch_copy = copy.deepcopy(X_batch)

        ##########################################################################################
        # Use the model to predict X_batch first, and according to the result,
        # only the normal data will be used to train a new model instead of the current one.

        start_1 = datetime.now()
        ### Step 1: Preprocessing
        # Step 1.1: standardization
        # transform x
        X_batch_std, std_transform_time = func_running_time(self.std_inst.transform, X_batch)
        model_train_time += std_transform_time

        # Step 1.2: Seek the modes by quickshift++ or meanshift
        seek_train_time = 0
        model_train_time += seek_train_time

        # Step 1.3: Project the data onto a lower space with kjl or nystrom
        proj_train_time = 0
        if 'kjl' in self.params.keys() and self.params['kjl']:
            # Project the data
            X_batch_proj, proj_transform_time = func_running_time(self.kjl_inst.transform, X_batch_std)
            proj_train_time += proj_transform_time
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            # Project the data
            X_batch_proj, proj_transform_time, = func_running_time(self.nystrom_inst.transform, X_batch_std)
            proj_train_time += proj_transform_time
        else:
            X_batch_proj = X_batch_std
            proj_train_time = 0
        model_train_time += proj_train_time

        ##########################################################################################
        # Step 2: Obtain the abnormal score
        # For inlier, a small value is used; a larger value is for outlier (positive)
        # here the output should be the abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
        y_score, model_predict_time = func_running_time(self.model.decision_function, X_batch_proj)
        # print("i:{}, online model prediction takes {} seconds, y_score: {}".format(0, testing_time, y_score))
        model_train_time += model_predict_time
        if self.verbose > 8: data_info(y_score.reshape(-1, 1), name='y_score')
        # if self.verbose > 5:
        #     print(f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, '
        #           f'seek_test_time: {self.seek_test_time}'
        #     f', proj_test_time: {self.proj_test_time}, '
        #     f'model_test_time: {self.model_test_time}')

        # Step 2.2: update std
        # update the mean and scaler of self.std_inst with 'x'
        _, std_update_time = func_running_time(self.std_inst.update,
                                               X_batch)  # use the original X_batch to update std_inst
        X_batch_std, std_transform_time = func_running_time(self.std_inst.transform, X_batch)
        # Step 2.3: update projection: kjl or nystrom
        if 'kjl' in self.params.keys() and self.params['kjl']:
            #  Update kjl: self.U_kjl, self.Xrow_kjl.
            # use the X_batch without projection to update kjl_inst
            _, proj_update_time = func_running_time(self.kjl_inst.update, X_batch_std)
            X_batch_proj, proj_transform_time = func_running_time(self.kjl_inst.transform, X_batch_std)

        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            # Update nystrom_inst
            _, proj_update_time = func_running_time(self.nystrom_inst.update, X_batch_std)
            X_batch_proj, proj_transform_time = func_running_time(self.nystrom_inst.transform, X_batch_std)
        else:
            proj_update_time = 0

        end_1 = datetime.now()
        online_preprocessing_time = (end_1 - start_1).total_seconds()

        start_2 = datetime.now()
        # Step 2.4: train a copied model on the batch data
        # 1) train on the normal data first (without needing to create any new component)
        # 2) train on the abnormal data (create new components)
        # cond_normal = y_score < self.novelty_thres  # return bool values:  normal index
        normal_idx = np.where((y_score <= self.abnormal_thres) == True)
        abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
        X_batch_proj_normal = X_batch_proj[normal_idx] if len(normal_idx[0]) > 0 else []
        X_batch_proj_abnormal = X_batch_proj[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

        new_model = ONLINE_GMM(n_components=self.model.n_components,
                               covariance_type=self.model.covariance_type,
                               weights_init=self.model.weights_,
                               means_init=self.model.means_,
                               covariances_init=self.model.covariances_,
                               verbose=self.verbose,
                               warm_start=True)
        new_model.n_samples = self.model.n_samples
        new_model.sum_resp = self.model.sum_resp  # shape (1, d): sum of exp(log_resp)
        # self.acculumated_X_train_proj = self.model.X_train_proj
        if self.verbose > 5: print(f'X_batch.shape: {X_batch.shape}')
        while (len(X_batch_proj_normal) > 0) or (len(X_batch_proj_abnormal) > 0):

            if len(X_batch_proj_normal) > 0:
                # 1) train on the normal data first (without needing to create any new component)
                _st = datetime.now()
                log_prob_norm, log_resp = new_model._e_step_online(X_batch_proj_normal)
                new_model._m_step_online(X_batch_proj_normal, np.exp(log_resp), sum_resp_pre=new_model.sum_resp,
                                         n_samples_pre=self.acculumated_X_train_proj.shape[0])

                self.acculumated_X_train_proj = np.concatenate([self.acculumated_X_train_proj, X_batch_proj_normal],
                                                               axis=0)
                self.abnormal_thres = self.update_abnormal_thres(new_model, self.acculumated_X_train_proj)
                _end = datetime.now()
                _tot = (_end - _st).total_seconds()
                print(f'processing {len(X_batch_proj_normal)} normal flows takes {_tot} seconds.')
                X_batch_proj_normal = []
            elif len(X_batch_proj_abnormal) > 0:
                # 2) train on the abnormal data (create new components)
                # each time only focuses on one abnormal datapoint
                _st = datetime.now()
                idx = 0
                x_proj = X_batch_proj_abnormal[idx].reshape(1, -1)
                X_batch_proj_abnormal = np.delete(X_batch_proj_abnormal, idx, axis=0)

                self.acculumated_X_train_proj = np.concatenate([self.acculumated_X_train_proj, x_proj], axis=0)
                new_model.add_new_component(x_proj, self.params['q_abnormal_thres'], self.acculumated_X_train_proj)
                _end = datetime.now()
                _tot = (_end - _st).total_seconds()
                print(f'processing 1 abnormal flow takes {_tot} seconds.')
            else:
                break

            if len(X_batch_proj_abnormal) > 0:
                y_score = new_model.decision_function(X_batch_proj_abnormal)
                # cond_normal = y_score < self.novelty_thres  # return bool values:  normal index
                normal_idx = np.where((y_score <= self.abnormal_thres) == True)
                abnormal_idx = np.where((y_score > self.abnormal_thres) == True)

                X_batch_proj_normal = X_batch_proj_abnormal[normal_idx] if len(normal_idx[0]) > 0 else []
                X_batch_proj_abnormal = X_batch_proj_abnormal[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

        end_2 = datetime.now()
        online_frist_train_time = (end_2 - start_2).total_seconds()

        start_3 = datetime.now()
        # train the new model until it converges
        i = 0
        new_model.converged_ = False
        if not new_model.converged_:  new_model.max_iter = 100
        prev_lower_bound = -np.infty
        if self.verbose > 5: print(f'acculumated_X_train_proj: {self.acculumated_X_train_proj.shape}')
        while (i < new_model.max_iter) and (not new_model.converged_):
            _st = datetime.now()
            # log_prob_norm, log_resp = new_model._e_step(self.acculumated_X_train_proj)
            (log_prob_norm, log_resp), e_time = func_running_time(new_model._e_step, self.acculumated_X_train_proj)
            log_prob_norm = np.mean(log_prob_norm)

            # get the difference
            lower_bound = new_model._compute_lower_bound(log_resp, log_prob_norm)
            change = lower_bound - prev_lower_bound
            if abs(change) < new_model.tol:
                new_model.converged_ = True
                print(f'n_iter: {i}')
                # break
            prev_lower_bound = lower_bound
            # # use m_step to update params: weights (i.e., mixing coefficients), means, and covariances with x and
            # # the previous params: log_resp (the log probability of each component), means and covariances
            # new_model._m_step(x_proj, new_model.log_resp,
            #                   new_model.n_samples)  # update mean, covariance and weight
            # sum_resp_pre = np.zeros((1,new_model.n_components)).reshape(new_model.sum_resp.shape)
            # new_model._m_step(new_model.X_train_proj, np.exp(log_resp), sum_resp_pre)
            _, m_time = func_running_time(new_model._m_step, self.acculumated_X_train_proj, np.exp(log_resp))
            _end = datetime.now()
            _tot = (_end - _st).total_seconds()
            if self.verbose > 5: print(f'{i + 1}th iterations takes {_tot} seconds, in which e_time: {e_time}, '
                                       f'and m_time: {m_time}')
            i += 1

        end_3 = datetime.now()
        online_iterations_train_time = (end_3 - start_3).total_seconds()

        start_4 = datetime.now()

        # Step 3:  update the abnormal threshold with all accumulated data
        self.model = new_model
        y_score, model_predict_time = func_running_time(self.model.decision_function, self.acculumated_X_train_proj)
        if self.verbose > 5: print(
            f'model_predict_time: {model_predict_time}, X_train_proj: {self.acculumated_X_train_proj.shape}')
        self.abnormal_thres = np.quantile(y_score, q=self.params['q_abnormal_thres'])  # abnormal threshold

        end_4 = datetime.now()
        online_recalculate_score_time = (end_4 - start_4).total_seconds()

        end_0 = datetime.now()
        model_online_train_time = (end_0 - start_0).total_seconds()
        print(f'Total batch time: {model_online_train_time} <=: preprocessing: {online_preprocessing_time},'
              f'first_train: {online_frist_train_time}, iterations_train: {online_iterations_train_time},'
              f'new_score: {online_recalculate_score_time}')

        self.train_time = {'train_time': model_online_train_time,
                           'preprocessing': online_preprocessing_time,
                           'first_train': online_frist_train_time,
                           'iteration_train': online_iterations_train_time,
                           'rescore': online_recalculate_score_time}

        return self

    def update_abnormal_thres(self, model, X_normal_proj):
        """Only use abnormal_score to update the abnormal_thres, in which,
            abnormal_score = y_score[y_score > abnormal_thres]

        Parameters
        ----------
        model
        abnormal_thres
        X_normal_proj

        Returns
        -------

        """

        y_score, model_predict_time = func_running_time(model.decision_function, X_normal_proj)
        abnormal_thres = np.quantile(y_score, q=self.params['q_abnormal_thres'])  # abnormal threshold
        model.y_score = y_score
        # abnormal_idx = np.where((y_score > abnormal_thres) == True)
        # y_score = y_score[abnormal_idx] if len(abnormal_idx[0]) > 0 else []
        #
        # n = model.n_samples
        # m = X_normal_proj.shape[0]
        #
        # if len(y_score) > 0:
        #     abnormal_thres = abnormal_thres + np.sum(y_score - abnormal_thres) / (n + m)
        # return abnormal_thres

        return abnormal_thres

    def _online_test(self, X_test, y_test):
        """Evaluate model on the test set

        Parameters
        ----------
        ----------
        model: model instance
            a fitted model on the init_set

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

        # Step 1.2: project time
        if 'kjl' in self.params.keys() and self.params['kjl']:
            X_test, proj_test_time = func_running_time(self.kjl_inst.transform, X_test)

        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            X_test, proj_test_time = func_running_time(self.nystrom_inst.transform, X_test)
        else:
            proj_test_time = 0
        self.test_time += proj_test_time

        ##########################################################################################
        # Step 2: evaluate GMM on the test set
        # print(X_test) # for dubeg
        y_score, model_test_time = func_running_time(self.model.decision_function, X_test)
        self.auc, model_auc_time = func_running_time(self.get_score, y_test, y_score)
        self.test_time += model_test_time + model_auc_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {std_test_time},'
            f', proj_test_time: {proj_test_time}, '
            f'model_test_time: {model_test_time}, model.paramters: {self.model.get_params()}')
        self.test_time = {'test_time': self.test_time,
                          'std_time': std_test_time,
                          'proj_time': proj_test_time,
                          'predict_time': model_test_time,
                          'auc_time': model_auc_time}

        return self


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

    def train_test_model(self, X_train, y_train, X_arrival, y_arrival, X_test, y_test):
        """ Train and test model

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
        self.n_samples, self.n_feats = X_train.shape

        ##########################################################################################
        # # Step 1. Get initial model (init_model) on initial set (init_set) and evaluate it on test set
        self.init_model = self.get_best_model(X_train, y_train, X_test, y_test)

        ##########################################################################################
        # Step 2. train the model on the batch data (previous+batch) and evaluate it.
        batch_train_times = [self.init_info['train_time']]
        batch_test_times = [self.init_info['test_time']]
        batch_aucs = [self.init_info['auc']]
        batch_novelty_threses = [self.init_info['novelty_thres']]
        batch_abnormal_threses = [self.init_info['abnormal_thres']]
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=self.params['batch_size'])):
            print(f'\n***batch: {i + 1}')
            if i == 0:
                self.model = copy.deepcopy(self.init_model)

            self._batch_train(X_batch, Y_batch)
            batch_train_times.append(self.train_time)
            batch_novelty_threses.append(self.init_info['novelty_thres'])
            batch_abnormal_threses.append(self.init_info['abnormal_thres'])

            # batch test model
            self._batch_test(X_test, y_test)
            batch_test_times.append(self.test_time)
            batch_aucs.append(self.auc)

            if self.verbose > 5:
                print(f'***batch: {i + 1}, train_time: {self.train_time}, test_time: {self.test_time}, auc: {self.auc}')

        self.info['train_times'] = batch_train_times
        self.info['test_times'] = batch_test_times
        self.info['aucs'] = batch_aucs
        self.info['novelty_threses'] = batch_novelty_threses
        self.info['abnormal_threses'] = batch_abnormal_threses

        return self

    def _batch_train(self, X_batch, y_batch=None):

        """batch train the model: using the (X_batch + previous data) to retrain a new model
        and use it to update the current model.

        Parameters
        ----------
        model: model instance
            a fitted model on the init_set, which is used to predict a new datatpoint is normal or not.

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
        # U the model to predict X_batch first, and according to the result,
        # only the normal data will be incorporated with previous data to train a new model instead of the current one.

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
        if self.verbose > 5: data_info(y_score.reshape(-1, 1), name='y_score')
        # print("i:{}, batch model prediction takes {} seconds, y_score: {}".format(0, testing_time, y_score))

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
        print(f'X_train: {self.init_X_train.shape}, drop {abnormal_cnt} abnormal flows.')

        # Step 3.2: train a new model and use it to instead of the current one.
        # we use model.n_compoents to initialize n_components or the value found by quickhsift++ or meanshift
        self.model_copy = copy.deepcopy(self.model)
        self.model = GMM(n_components=self.model.n_components)
        # update self.model: replace the current model with new_model
        self._init_train(self.init_X_train, self.init_y_train)  # self.train_time

        # the datapoint is predicted as a abnormal flow, so we should drop it.
        print(f'{abnormal_cnt} flows are predicted as abnormal, so we drop them.')

        end_0 = datetime.now()
        model_batch_train_time = (end_0 - start_0).total_seconds()
        print(f'Total batch time: {model_batch_train_time}')

        return self

    def _batch_test(self, X_batch, y_batch):
        self._init_test(X_batch, y_batch)


def get_normal_abnormal(normal_file, abnormal_file, random_state=42):
    """Get normal and abnormal data

    Parameters
    ----------
    normal_file
    abnormal_file
    random_state

    Returns
    -------
        normal data
        abnormal data

    """
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

                    line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs:{aucs_str}, train_times:' \
                           f'{train_times_str}, test_times:{test_times_str}, with params: {params}: {_suffex}'

                except Exception as e:
                    traceback.print_exc()
                    line = ''
                f.write(line + '\n')
                print(line)
            f.write('\n')


def plot_result(result, out_file):
    import matplotlib.pyplot as plt

    def plot_data(x, y, xlabel='range', ylabel='auc', ylim=[], title='', out_file=''):

        # with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        ax.plot(x, y, '*-', alpha=0.9)
        # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            plt.ylim(ylim)  # [0.0, 1.05]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        # plt.legend(loc='lower right')
        plt.title(title)

        # should use before plt.show()
        plt.savefig(out_file)

        plt.show()

    def plot_data2(xs, ys, xlabel='range', ylabel='auc', ylim=[], title='', out_file=''):

        # with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        for (x, y) in zip(xs, ys):
            ax.plot(x, y, '*-', alpha=0.9)
        # ax.plot(x, y, '*-', alpha=0.9)
        # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            plt.ylim(ylim)  # [0.0, 1.05]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        # plt.legend(loc='lower right')
        plt.title(title)

        # should use before plt.show()
        plt.savefig(out_file)

        plt.show()

    def plot_times(ys, xlabel='range', ylabel='auc', ylim=[], title='', out_file=''):

        fig, ax = plt.subplots()
        names = ys[-1].keys()
        for name in names:
            y = [v[name] for v in ys]
            x = range(len(y))
            # with plt.style.context(('ggplot')):
            ax.plot(x, y, '*-', alpha=0.9, label=name)
            # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            plt.ylim(ylim)  # [0.0, 1.05]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        plt.legend(loc='lower right')
        plt.title(title)

        # should use before plt.show()
        plt.savefig(out_file)

        plt.show()

    for (in_dir, case_str), (best_results, middle_results) in result.items():
        print(f'\n***{in_dir}, {case_str}')
        if 'online_False' in case_str:
            online = False
        else:
            online = True

        title = f'online:{online}'

        if not online:
            y = best_results['train_times']
            plot_data(range(len(y)), y, xlabel='batch data', ylabel='Training time (s)', title=title,
                      out_file=out_file.replace('.pdf', '-train_times.pdf'))
            y = best_results['test_times']
            plot_data(range(len(y)), y, xlabel='batch data', ylabel='Testing time (s)', title=title,
                      out_file=out_file.replace('.pdf', '-test_times.pdf'))

        else:
            ys = best_results['train_times']
            plot_times(ys, xlabel='batch data', ylabel='Training time (s)', title=title,
                       out_file=out_file.replace('.pdf', '-train_times.pdf'))

            ys = best_results['test_times']
            plot_times(ys, xlabel='batch data', ylabel='Testing time (s)', title=title,
                       out_file=out_file.replace('.pdf', '-test_times.pdf'))

            y = [v['n_components'] for v in best_results['model_params']]
            plot_data(range(len(y)), y, xlabel='batch data', ylabel='n_components time', title=title,
                      out_file=out_file.replace('.pdf', '-n_components.pdf'))

        y = best_results['aucs']
        plot_data(range(len(y)), y, xlabel='batch data', ylabel='AUC', ylim=[0.0, 1.05], title=title,
                  out_file=out_file.replace('.pdf', '-aucs.pdf'))

        y = best_results['novelty_threses']
        plot_data(range(len(y)), y, xlabel='batch data', ylabel='novelty threshold', title=title,
                  out_file=out_file.replace('.pdf', '-novelty.pdf'))

        y = best_results['abnormal_threses']
        plot_data(range(len(y)), y, xlabel='batch data', ylabel='abnormal threshold', title=title,
                  out_file=out_file.replace('.pdf', '-abnormal.pdf'))

        y1 = best_results['novelty_threses']
        y2 = best_results['abnormal_threses']
        xs = [range(len(y1)), range(len(y2))]
        ys = [y1, y2]
        plot_data2(xs, ys, xlabel='batch data', ylabel='Threshold', title=title,
                   out_file=out_file.replace('.pdf', '-threshold.pdf'))


def split_train_arrival_test(normal_arr, abnormal_arr, random_state=42):
    """

    Parameters
    ----------
    normal_arr
    abnormal_arr
    random_state

    Returns
    -------

    """
    from fractions import Fraction

    n_feats = min([data.shape[1] for data in normal_arr])
    for i in range(len(normal_arr)):
        normal_arr[i] = normal_arr[i][:, :n_feats]
        abnormal_arr[i] = abnormal_arr[i][:, :n_feats]

    ##########################################################################################
    # dataset1
    X_train1, y_train1, X_test1, y_test1 = split_train_test(normal_arr[0], abnormal_arr[0], train_size=0.8,
                                                            test_size=-1, random_state=random_state)
    # dataset2
    X_train2, y_train2, X_test2, y_test2 = split_train_test(normal_arr[1], abnormal_arr[1], train_size=0.8,
                                                            test_size=-1, random_state=random_state)

    X_test = np.concatenate([X_test1, X_test2], axis=0)
    y_test = np.concatenate([y_test1, y_test2], axis=0)

    ##########################################################################################
    # dataset1: Split train set into two subsets: initial set (init_set) and new arrival set (arvl_set)
    # with ratio 1:1.
    # in the init_set: X_train1 / X_train2= 9:1
    X_train1, X_arrival1, y_train1, y_arrival1 = train_test_split(X_train1, y_train1, train_size=0.9,
                                                                  random_state=random_state)
    # in the arrival set: X_arrival1 / X_arrival = 1:9
    X_train2, X_arrival2, y_train2, y_arrival2 = train_test_split(X_train2, y_train2, train_size=0.1,
                                                                  random_state=random_state)
    X_train = np.concatenate([X_train1, X_train2], axis=0)
    y_train = np.concatenate([y_train1, y_train2], axis=0)

    X_arrival = np.concatenate([X_arrival1, X_arrival2], axis=0)
    y_arrival = np.concatenate([y_arrival1, y_arrival2], axis=0)

    print(f'X_train: {X_train.shape}, in which, X_train1/X_train2 is {Fraction(X_train1.shape[0], X_train2.shape[0])}')
    print(f'X_arrival: {X_arrival.shape} in which, X_arrival1/X_arrival2 is '
          f'{Fraction(X_arrival1.shape[0], X_arrival2.shape[0])}')
    print(
        f'X_test: {X_test.shape},in which, X_test1/X_test2 {Fraction(X_test1.shape[0], X_test2.shape[0], _normalize=False)}')

    return X_train, y_train, X_arrival, y_arrival, X_test, y_test


@execute_time
def main(random_state, n_jobs=-1, n_repeats=1, online=True):
    """

    Parameters
    ----------
    random_state
    n_jobs
    n_repeats
    online: bool
        each case includes online and batch
        online = True  # online: True, otherwise, batch

    Returns
    -------

    """
    datasets = [
        #     # # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',
        'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
        # 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
        'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
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
        # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs, 'kjl': True},
        {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs, 'kjl': True},
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

    # All the results will be stored in the 'results'
    results = {}

    normal_arr = []
    abnormal_arr = []
    for i, data_name in enumerate(datasets):
        ##########################################################################################
        # Step 1: extract normal and abnormal data from input files.
        in_expand_dir = pth.join(in_dir, data_name, feat_set, f'header:{header}')
        out_expand_dir = pth.join(out_dir, data_name, feat_set, f'header:{header}')
        normal_file = f'{in_expand_dir}/normal.dat'
        abnormal_file = f'{in_expand_dir}/abnormal.dat'
        print(normal_file, abnormal_file)
        normal_data, abnormal_data = get_normal_abnormal(normal_file, abnormal_file, random_state=random_state)
        normal_arr.append(normal_data)
        abnormal_arr.append(abnormal_data)

    ##########################################################################################
    # Step 2: conduct experiments for each case
    for case in cases:
        case['random_state'] = random_state
        case['verbose'] = 5
        case['online'] = online  # online: True, otherwise, batch
        case['q_abnormal_thres'] = 1  # default 0.95
        case['batch_size'] = 1000  #

        keys = ['detector_name', 'covariance_type', 'gs', 'kjl', 'nystrom', 'quickshift',
                'meanshift', 'online']
        case_str = ''
        for k in keys:
            if k not in case.keys():
                case[k] = False
            case_str += f'{k}_{case[k]}-'

        # case_str = '-'.join([f'{k}_{v}' for k, v in case.items() if k in keys])
        try:
            # 3. get each result
            print(f"\n\n\n***{case}***, {case_str}")
            X_train, y_train, X_arrival, y_arrival, X_test, y_test = split_train_arrival_test(normal_arr, abnormal_arr,
                                                                                              random_state)

            if 'GMM' == case['detector_name']:
                if case['online']:
                    model = ONLINE_GMM_MAIN(case)
                else:  # 'batch', i.e., batch_GMM
                    model = BATCH_GMM_MAIN(case)
            else:
                raise NotImplementedError()
            model.train_test_model(X_train, y_train, X_arrival, y_arrival, X_test, y_test)
            _best_results = model.info  # model.info['train_times']
            _middle_results = {}

            # # save each result first
            # out_file = pth.abspath(f'{out_expand_dir}/{case_str}.csv')
            # print('+++', out_file)
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

    ##########################################################################################
    # # 5. plot results
    out_file = f'{out_dir}/online_{online}.pdf'
    print(f'\n\n***{out_file}***')
    # # Todo: format the results
    plot_result(results, out_file)
    print("\n\n---finish succeeded!")


if __name__ == '__main__':
    main(random_state=RANDOM_STATE, n_jobs=1, n_repeats=1, online=True)
