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

from sklearn.utils import shuffle

from generate_data import PCAP2FEATURES, artifical_data, mimic_data
from kjl.model.gmm import GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.model.kjl import KJL
from kjl.model.nystrom import NYSTROM
from kjl.model.online_gmm import ONLINE_GMM, quickshift_seek_modes, meanshift_seek_modes
from kjl.model.standardization import STD

from kjl.utils.data import split_train_test, load_data, extract_data, dump_data, save_result, batch, data_info
from kjl.utils.tool import execute_time, time_func
from fractions import Fraction
from collections import Counter
from report import plot_result

RANDOM_STATE = 43  # change random_state, the result will change too.
print('PYTHONPATH: ', os.environ['PYTHONPATH'])
np.random.seed(RANDOM_STATE)


class BASE_MODEL():

    def get_best_model(self, X_train, y_train, X_test, y_test):

        self.init_X_train = X_train
        self.init_y_train = y_train

        ##########################################################################################
        # Step 1: configure case
        # case = 'GMM_full-gs_True-kjl_True-nystrom_False-quickshift_False-meanshift_False'
        if self.params['detector_name'] == 'GMM' and self.params['covariance_type'] == 'diag' and \
                self.params['gs'] and not self.params['quickshift'] and \
                not self.params['meanshift']:  # self.params['kjl']  and
            params = {}
            # best params for the combination of UNB1 and UNB2
            params['n_components'] = [5]  # UNB2
            params['kjl_qs'] = [0.7]

            params['n_components'] = [20]  # mimic_GMM_dataset
            params['kjl_qs'] = [0.8]

            # # # # # grid search
            params['n_components'] =  [1,  5, 10, 15, 20, 25, 30, 35, 40, 45]
            params['kjl_qs'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

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
            self.model = GMM(n_components=n_components, random_state=self.params['random_state'],
                             covariance_type=self.params['covariance_type'])

            # Fit a model on the train set
            self._init_train(X_train, y_train)  # fit self.model with self.params on train set
            # Evaluate the model on the test set
            self._init_test(X_test, y_test)  # get self.auc
            print(f'auc: {self.auc}, {self.model.get_params()}, self.kjl_q: {kjl_q}')
            # for out in outs:
            if best_auc < self.auc:  # here should be <, not <=
                best_auc = self.auc
                best_model_params = copy.deepcopy(self.model.get_params())
                best_params = copy.deepcopy(self.params)

        ##########################################################################################
        print('\n\n******best model')
        # Step 3: get the best model with best_params
        self.params = best_params
        self.model = GMM(random_state=self.params['random_state'], covariance_type=self.params['covariance_type'])
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
        _, std_train_time = time_func(self.std_inst.fit, X_train)
        self.train_time += std_train_time
        X_train, std_train_time = time_func(self.std_inst.transform, X_train)
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
            _, kjl_train_time = time_func(self.kjl_inst.fit, X_train)
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

        print(self.std_inst.scaler.mean_, self.std_inst.scaler.scale_)
        data_info(X_train, name='without updating X_proj')
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
        self.init_X_train_proj = X_train  # self.model.X_train_proj = X_train

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

    def train_test_model(self, X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test,
                         y_test):
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
        self.n_samples, self.n_feats = X_init_train.shape
        ##########################################################################################
        # # Step 1. Get initial model (init_model) on initial set (init_set) and evaluate it on test set
        self.init_model = self.get_best_model(X_init_train, y_init_train, X_init_test, y_init_test)

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
                self.acculumated_X_train_proj = self.init_X_train_proj
                # self.acculumated_X_train = self.init_X_train

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
        self.info['params'] = self.params

        return self

    def _online_train(self, X_batch, y_batch=None):
        """Online train the model: using the X_batch to retrain and update the current model incrementally.

        Parameters
        ----------

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
        ##########################################################################################
        # Step 1: Predication.
        # Use the fitted model to predict X_batch first, and according to the result,
        # only the normal data will be used to train a new model, which will be used to replace the current model.

        start_1 = datetime.now()
        # Step 1.1: Preprocessing: std and projection
        X_batch_proj, info = self._preprocessing(X_batch, update=False)
        # Step 1.2: Obtain the abnormal score (note that a larger value is for outlier (positive))
        y_score, model_predict_time = func_running_time(self.model.decision_function, X_batch_proj)
        if self.verbose >= 8:
            print(f"online model prediction takes {model_predict_time} seconds.")
            data_info(y_score.reshape(-1, 1), name='y_score')
        end_1 = datetime.now()
        online_preprocessing_time = (end_1 - start_1).total_seconds()

        # data_info(X_batch_proj, name ='before updating X_proj')
        ##########################################################################################
        # Step 2: online train a new model on the batch data
        # 1) train on the normal data first (without needing to create any new component)
        # 2) train on the abnormal data (create new components)
        start_2 = datetime.now()

        # Step 2.1:  update std and proj, and get the new X_data
        X_batch_proj, info = self._preprocessing(X_batch, update=True)  # here must be X_batch, not X_batch_proj
        # data_info(X_batch_proj, name='after updating X_proj')
        if not self.params['kjl']:
            self.params['incorporated_points'] = 0
            self.params['fixed_U_size'] = False
        else:
            self.params['incorporated_points'] = self.kjl_inst.t
            self.params['fixed_U_size'] = self.kjl_inst.fixed_U_size

        # self.abnormal_thres = np.infty  # for debug

        normal_idx = np.where((y_score <= self.abnormal_thres) == True)
        abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
        X_batch_proj_normal = X_batch_proj[normal_idx] if len(normal_idx[0]) > 0 else []
        X_batch_proj_abnormal = X_batch_proj[abnormal_idx] if len(abnormal_idx[0]) > 0 else []
        # X_batch_normal = X_batch[normal_idx] if len(normal_idx[0]) > 0 else []
        # X_batch_abnormal = X_batch[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

        new_model = ONLINE_GMM(n_components=self.model.n_components,
                               covariance_type=self.model.covariance_type,
                               weights_init=self.model.weights_,
                               means_init=self.model.means_,
                               covariances_init=self.model.covariances_,
                               verbose=self.verbose,
                               random_state=self.params['random_state'],
                               warm_start=True)

        # print('self.model.weights_', self.model.weights_)
        # print('self.model.means_', self.model.means_)
        # print('self.model.covariances_', self.model.covariances_)
        # new_model.n_samples = self.model.n_samples
        new_model.sum_resp = self.model.sum_resp  # shape (1, d): sum of exp(log_resp)
        new_model._initialize()  # set up cholesky

        if self.verbose > 5: print(f'X_batch.shape: {X_batch.shape}')
        update_flg = False
        # The first update of GMM
        while (len(X_batch_proj_normal) > 0) or (len(X_batch_proj_abnormal) > 0):

            if len(X_batch_proj_normal) > 0:
                # 1) train on the normal data first (without needing to create any new component)
                _st = datetime.now()
                _, log_resp = new_model._e_step(X_batch_proj_normal)
                # data_info(np.exp(log_resp), name='resp')
                new_model._m_step_online(X_batch_proj_normal, log_resp, sum_resp_pre=new_model.sum_resp,
                                         n_samples_pre=self.acculumated_X_train_proj.shape[0])
                self.acculumated_X_train_proj = np.concatenate([self.acculumated_X_train_proj, X_batch_proj_normal],
                                                               axis=0)
                # self.acculumated_X_train = np.concatenate([self.acculumated_X_train, X_batch_normal],
                #                                                axis=0)
                _end = datetime.now()
                _tot = (_end - _st).total_seconds()
                print(f'processing {len(X_batch_proj_normal)} normal flows takes {_tot} seconds.')
                X_batch_proj_normal = []
            elif len(X_batch_proj_abnormal) > 0:
                update_flg = True
                # 2) train on the abnormal data (create new components)
                # each time only focuses on one abnormal datapoint
                _st = datetime.now()
                idx = 0
                x_proj = X_batch_proj_abnormal[idx].reshape(1, -1)
                X_batch_proj_abnormal = np.delete(X_batch_proj_abnormal, idx, axis=0)

                self.acculumated_X_train_proj = np.concatenate([self.acculumated_X_train_proj, x_proj], axis=0)
                new_model.add_new_component(x_proj, self.params['q_abnormal_thres'], self.acculumated_X_train_proj)
                # x_ = X_batch_abnormal[idx].reshape(1, -1)
                # X_batch_abnormal = np.delete(X_batch_abnormal, idx, axis=0)
                #
                # self.acculumated_X_train = np.concatenate([self.acculumated_X_train, x_], axis=0)
                # v, info = self._preprocessing(self.acculumated_X_train, update=False)
                # new_model.add_new_component(x_proj, self.params['q_abnormal_thres'], v)
                _end = datetime.now()
                _tot = (_end - _st).total_seconds()
                print(f'processing 1 abnormal flow takes {_tot} seconds.')
                print(new_model.get_params())
            else:
                break

            if len(X_batch_proj_abnormal) > 0:
                if not update_flg:
                    # v, info = self._preprocessing(self.acculumated_X_train, update=False)
                    self.abnormal_thres = self.update_abnormal_thres(new_model, self.acculumated_X_train_proj)

                y_score = new_model.decision_function(X_batch_proj_abnormal)
                # cond_normal = y_score < self.novelty_thres  # return bool values:  normal index
                normal_idx = np.where((y_score <= self.abnormal_thres) == True)
                abnormal_idx = np.where((y_score > self.abnormal_thres) == True)

                X_batch_proj_normal = X_batch_proj_abnormal[normal_idx] if len(normal_idx[0]) > 0 else []
                X_batch_proj_abnormal = X_batch_proj_abnormal[abnormal_idx] if len(abnormal_idx[0]) > 0 else []
                # X_batch_normal = X_batch_abnormal[normal_idx] if len(normal_idx[0]) > 0 else []
                # X_batch_abnormal = X_batch_abnormal[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

        end_2 = datetime.now()
        online_frist_train_time = (end_2 - start_2).total_seconds()

        start_3 = datetime.now()
        # Train the new model until it converges
        # X_batch_proj_normal = X_batch
        # if len(X_batch_proj_normal) > 0:  self.acculumated_X_train = np.concatenate([self.acculumated_X_train, X_batch_proj_normal], axis=0)
        # data_info(self.acculumated_X_train, name='self.acculumated_X_train_proj')
        # self.acculumated_X_train_proj, info = self._preprocessing(self.acculumated_X_train, update=False)
        # print(self.std_inst.scaler.mean_, self.std_inst.scaler.scale_)
        # data_info(self.acculumated_X_train_proj,name='self.acculumated_X_train_proj')
        i = 0
        new_model.converged_ = False
        if not new_model.converged_:  new_model.max_iter = 100
        prev_lower_bound = -np.infty
        if self.verbose > 1: print(f'acculumated_X_train_proj: {self.acculumated_X_train_proj.shape}')
        while (i < new_model.max_iter) and (not new_model.converged_):
            _st = datetime.now()
            # log_prob_norm, log_resp = new_model._e_step(self.acculumated_X_train_proj)
            (log_prob_norm, log_resp), e_time = func_running_time(new_model._e_step, self.acculumated_X_train_proj)

            # get the difference
            lower_bound = new_model._compute_lower_bound(log_resp, np.mean(log_prob_norm))
            change = lower_bound - prev_lower_bound
            if abs(change) < new_model.tol:
                new_model.converged_ = True
                print(f'n_iter: {i}')
                # break
            prev_lower_bound = lower_bound

            _, m_time = func_running_time(new_model._m_step, self.acculumated_X_train_proj, log_resp)

            _end = datetime.now()
            _tot = (_end - _st).total_seconds()
            if self.verbose > 5: print(f'{i + 1}th iterations takes {_tot} seconds, in which e_time: {e_time}, '
                                       f'and m_time: {m_time}')
            i += 1

        end_3 = datetime.now()
        online_iterations_train_time = (end_3 - start_3).total_seconds()

        ##########################################################################################
        # Step 3:  update the abnormal threshold with all accumulated data
        self.model = new_model
        print(new_model.get_params())
        if not new_model.converged_:
            self.abnormal_thres, online_recalculate_score_time = func_running_time(self.update_abnormal_thres,
                                                                                   self.model,
                                                                                   self.acculumated_X_train_proj)
        else:
            y_score = - log_prob_norm  # override the _e_step(), so  here is not mean(log_prob_norm)    #  return -1 * self.score_samples(X)
            # self.abnormal_thres = np.quantile(y_score, q=self.params['q_abnormal_thres'])  # abnormal threshold
            self.abnormal_thres, online_recalculate_score_time = func_running_time(np.quantile, y_score,
                                                                                   q=self.params['q_abnormal_thres'])

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
        # model.y_score = y_score

        return abnormal_thres

    def _preprocessing(self, X_test, update=False):

        info = {}
        if not update:
            ##########################################################################################
            # Step 1: preprocessing, which includes standardization, mode seeking and projection
            # Step 1.1: standardization
            std_X_test, std_test_time = func_running_time(self.std_inst.transform, X_test)

            # Step 1.2: project time
            if 'kjl' in self.params.keys() and self.params['kjl']:
                proj_X_test, proj_test_time = func_running_time(self.kjl_inst.transform, std_X_test)

            elif 'nystrom' in self.params.keys() and self.params['nystrom']:
                proj_X_test, proj_test_time = func_running_time(self.nystrom_inst.transform, std_X_test)
            else:
                proj_X_test = std_X_test
                proj_test_time = 0

            info = {'std_transform_time': std_test_time, 'proj_transform_time': proj_test_time}
        else:
            # update the mean and scaler of self.std_inst with 'x'
            _, std_update_time = func_running_time(self.std_inst.update,
                                                   X_test)  # use the original X_batch to update std_inst
            X_test_std, std_transform_time = func_running_time(self.std_inst.transform, X_test)
            # Step 2.3: update projection: kjl or nystrom
            if 'kjl' in self.params.keys() and self.params['kjl']:
                #  Update kjl: self.U_kjl, self.Xrow_kjl.
                # use the X_batch without projection to update kjl_inst
                _, proj_update_time = func_running_time(self.kjl_inst.update, X_test_std)
                proj_X_test, proj_transform_time = func_running_time(self.kjl_inst.transform, X_test_std)

            elif 'nystrom' in self.params.keys() and self.params['nystrom']:
                # Update nystrom_inst
                _, proj_update_time = func_running_time(self.nystrom_inst.update, X_test_std)
                proj_X_test, proj_transform_time = func_running_time(self.nystrom_inst.transform, X_test_std)
            else:
                proj_X_test = X_test_std
                proj_update_time = 0

        return proj_X_test, info

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
        X_test, info = self._preprocessing(X_test, update=False)
        std_test_time = info['std_transform_time']
        proj_test_time = info['proj_transform_time']
        self.test_time += std_test_time + proj_test_time
        # data_info(X_test, name='X_test')

        ##########################################################################################
        # Step 2: evaluate GMM on the test set
        # print(X_test) # for dubeg
        # print('self.model.weights_', self.model.weights_)
        # print('self.model.means_', self.model.means_)
        # print('self.model.covariances_', self.model.covariances_)
        y_score, model_test_time = func_running_time(self.model.decision_function, X_test)
        self.auc, model_auc_time = func_running_time(self.get_score, y_test, y_score)
        self.test_time += model_test_time + model_auc_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {std_test_time},'
            f', proj_test_time: {proj_test_time}, '
            f'model_test_time: {model_test_time}, model.paramters: {self.model.get_params()}, auc: {self.auc}')
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

    def train_test_model(self, X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test,
                         y_test):
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
        self.n_samples, self.n_feats = X_init_train.shape

        ##########################################################################################
        # # Step 1. Get initial model (init_model) on initial set (init_set) and evaluate it on test set
        self.init_model = self.get_best_model(X_init_train, y_init_train, X_init_test, y_init_test)

        ##########################################################################################
        # Step 2. train the model on the batch data (previous+batch) and evaluate it.
        batch_train_times = [self.init_info['train_time']]
        batch_test_times = [self.init_info['test_time']]
        batch_aucs = [self.init_info['auc']]
        batch_novelty_threses = [self.init_info['novelty_thres']]
        batch_abnormal_threses = [self.init_info['abnormal_thres']]
        batch_model_params = [self.init_info['model_params']]
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=self.params['batch_size'])):
            print(f'\n***batch: {i + 1}')
            if i == 0:
                self.model = copy.deepcopy(self.init_model)

            self._batch_train(X_batch, Y_batch)
            batch_train_times.append(self.train_time)
            batch_novelty_threses.append(self.init_info['novelty_thres'])
            batch_abnormal_threses.append(self.init_info['abnormal_thres'])
            batch_model_params.append(self.init_info['model_params'])

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
        self.info['model_params'] = batch_model_params
        self.info['params'] = self.params

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
        self.model = GMM(n_components=self.model.n_components, covariance_type= self.params['covariance_type'], random_state=self.params['random_state'])
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


def split_train_arrival_test(normal_X, normal_y, abnormal_X, abnormal_y, one_data_flg=False, random_state=42):
    """

    Parameters
    ----------
    normal_arr
    abnormal_arr
    random_state

    Returns
    -------

    """
    init_train_n = 5000
    init_test_n = 100

    arrival_n = 5000
    # test_n = 5000

    normal_y = np.asarray(normal_y)
    abnormal_y = np.asarray(abnormal_y)

    def random_select(X, y, n=100, random_state=100):
        X, y = shuffle(X, y, random_state=random_state)

        X0 = X[:n, :]
        y0 = y[:n]

        rest_X = X[n:, :]
        rest_y = y[n:]
        return X0, y0, rest_X, rest_y

    normal_X, normal_y = shuffle(normal_X, normal_y, random_state=random_state)

    ##################################################################################################################
    # dataset 1
    idx = normal_y == 'normal_0'
    normal_X_1 = normal_X[idx]
    normal_y_1 = normal_y[idx]
    idx = abnormal_y == 'abnormal_0'
    abnormal_X_1 = abnormal_X[idx]
    abnormal_y_1 = abnormal_y[idx]

    # init_set: train and test set
    # init_train_set: data1/data2=9:1
    # init_test_set: data1/data=9:1
    init_test_abnormal1_size = 50
    test_abnormal1_size = 50
    init_train_normal_X_1, init_train_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1, normal_y_1,
                                                                                         n=4500,
                                                                                         random_state=random_state)

    init_test_normal_X_1, init_test_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1, normal_y_1,
                                                                                       n=init_test_abnormal1_size,
                                                                                       random_state=random_state)
    init_test_abnormal_X_1, init_test_abnormal_y_1, abnormal_X_1, abnormal_y_1 = random_select(abnormal_X_1,
                                                                                               abnormal_y_1,
                                                                                               n=init_test_abnormal1_size,
                                                                                               random_state=random_state)

    # arrival set: data1/data2 = 1:9
    arrival_train_normal_X_1, arrival_train_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1,
                                                                                               normal_y_1,
                                                                                               n=500,
                                                                                               random_state=random_state)

    # test set: data1/data2 = 1:1
    test_normal_X_1, test_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1, normal_y_1,
                                                                             n=test_abnormal1_size,
                                                                             random_state=random_state)

    test_abnormal_X_1, test_abnormal_y_1, abnormal_X_1, abnormal_y_1 = random_select(abnormal_X_1, abnormal_y_1,
                                                                                     n=test_abnormal1_size,
                                                                                     random_state=random_state)

    ##################################################################################################################
    # dataset 2
    # arrival set: data1/data2 = 1:9
    if one_data_flg:  # use the same dataset
        normal_X_2 = normal_X_1
        normal_y_2 = normal_y_1
        abnormal_X_2 = abnormal_X_1
        abnormal_y_2 = abnormal_y_1

    else:
        idx = normal_y == 'normal_1'
        normal_X_2 = normal_X[idx]
        normal_y_2 = normal_y[idx]
        idx = abnormal_y == 'abnormal_1'
        abnormal_X_2 = abnormal_X[idx]
        abnormal_y_2 = abnormal_y[idx]

    # init_set: train and test set
    init_train_normal_X_2, init_train_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
                                                                                         n=init_train_n - 4500,
                                                                                         random_state=random_state)

    init_test_normal_X_2, init_test_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
                                                                                       n=init_test_abnormal1_size,
                                                                                       random_state=random_state)
    init_test_abnormal_X_2, init_test_abnormal_y_2, abnormal_X_2, abnormal_y_2 = random_select(abnormal_X_2,
                                                                                               abnormal_y_2,
                                                                                               n=init_test_abnormal1_size,
                                                                                               random_state=random_state)

    arrival_train_normal_X_2, arrival_train_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
                                                                                               n=init_train_n - 500,
                                                                                               random_state=random_state)

    test_normal_X_2, test_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
                                                                             n=test_abnormal1_size,
                                                                             random_state=random_state)

    test_abnormal_X_2, test_abnormal_y_2, abnormal_X_2, abnormal_y_2 = random_select(abnormal_X_2, abnormal_y_2,
                                                                                     n=test_abnormal1_size,
                                                                                     random_state=random_state)

    # init_set: train and test set
    X_init_train = np.concatenate([init_train_normal_X_1, init_train_normal_X_2], axis=0)
    y_init_train = np.concatenate([init_train_normal_y_1, init_train_normal_y_2], axis=0)

    X_init_test = np.concatenate(
        [init_test_normal_X_1, init_test_normal_X_2, init_test_abnormal_X_1, init_test_abnormal_X_2], axis=0)
    y_init_test = np.concatenate(
        [init_test_normal_y_1, init_test_normal_y_2, init_test_abnormal_y_1, init_test_abnormal_y_2], axis=0)

    # arrival set: only train set
    X_arrival = np.concatenate([arrival_train_normal_X_1, arrival_train_normal_X_2], axis=0)
    y_arrival = np.concatenate([arrival_train_normal_y_1, arrival_train_normal_y_2], axis=0)

    # test set
    X_test = np.concatenate([test_normal_X_1, test_normal_X_2, test_abnormal_X_1, test_abnormal_X_2], axis=0)
    y_test = np.concatenate([test_normal_y_1, test_normal_y_2, test_abnormal_y_1, test_abnormal_y_2], axis=0)

    print(f'X_init_train: {X_init_train.shape}, in which, y_init_train is {Counter(y_init_train)}')
    print(f'X_init_test: {X_init_test.shape}, in which, y_init_test is {Counter(y_init_test)}')
    print(f'X_arrival_train: {X_arrival.shape}, in which, y_arrival is {Counter(y_arrival)}')
    print(f'X_test: {X_test.shape}, in which, y_test is {Counter(y_test)}')

    return X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test, y_test


#
# def _get_feats(subdatasets, out_dir, random_state=100):
#     obs_dir = '../../IoT_feature_sets_comparison_20190822/examples/'
#     in_dir = f'{obs_dir}data/data_reprst/pcaps'
#     out_dir = 'out/'
#     combined_feats, Xy_file = get_feats(subdatasets, in_dir, out_dir, q_interval=0.9,
#                                                         feat_type='IAT_SIZE', fft=False, header=False,
#                                                         random_state=random_state, verbose=10, single=False)
#
#     return combined_feats, Xy_file


def get_path(subdatasets, in_dir='', name='Xy-normal-abnormal.dat'):
    if type(subdatasets) == tuple:
        data_name = '-'.join(subdatasets).replace('/', '-')
    else:
        data_name = subdatasets

    if 'DS10_UNB' in data_name:
        data_name += '/AGMT-WorkingHours'

    return os.path.join(in_dir, data_name, name)


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
    datasets = [  # 'DEMO_IDS/DS-srcIP_192.168.10.5',
        'mimic_GMM_dataset',
        # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',  # data_name is unique
        # ('DS10_UNB_IDS/DS13-srcIP_192.168.10.8', 'DS10_UNB_IDS/DS14-srcIP_192.168.10.9'),   # demo
        # ('DS10_UNB_IDS/DS12-srcIP_192.168.10.8', 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9'),
        # ('DS10_UNB_IDS/DS13-srcIP_192.168.10.9', 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9'),
        # # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
        # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
        # #
        # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
        # #
        # # # 'DS30_OCS_IoT/DS31-srcIP_192.168.0.13',
        #
        # 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
        # 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196',
        # #
        # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        # 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
        # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
        # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
        # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
        # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
        #
        # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
        # 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
        # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
        # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'

        # 'WRCCDC/2020-03-20',
        # 'DEFCON/ctf26',
        # 'ISTS/2015',
        # 'MACCDC/2012',
        # 'CTU_IOT23/CTU-IoT-Malware-Capture-7-1',

    ]

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

    overwrite = 1
    # All the results will be stored in the 'results'
    results = {}
    for i, subdatasets in enumerate(datasets):
        Xy_file = get_path(subdatasets, 'out/data/data_reprst/pcaps', name='Xy-normal-abnormal.dat')
        in_dir = os.path.dirname(Xy_file)
        out_dir = os.path.dirname(Xy_file)
        print(f'\n***i:{i}, {Xy_file}')
        single_device = False
        if overwrite:
            if os.path.exists(Xy_file): os.remove(Xy_file)
        if not os.path.exists(Xy_file):
            if type(subdatasets) != tuple and 'mimic' in subdatasets:
                data, Xy_file = mimic_data(name=subdatasets, single_device=single_device, random_state=random_state)
            else:
                # pcaps and flows directory
                in_dir = f'../../IoT_feature_sets_comparison_20190822/examples/data/data_reprst/pcaps'
                pf = PCAP2FEATURES(out_dir=os.path.dirname(Xy_file), random_state=random_state)
                normal_files, abnormal_files = pf.get_path(subdatasets, in_dir)
                pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
                data, Xy_file = pf.data, pf.Xy_file
        else:
            print('load data')
            data, load_time = func_running_time(load_data, Xy_file)
            print(f'load {Xy_file} takes {load_time} s.')

        normal_X, normal_y = data['normal']
        abnormal_X, abnormal_y = data['abnormal']
        data_info(normal_X, name='normal')
        data_info(abnormal_X, name='abnormal')

        print(f'normal_X: {normal_X.shape}, normal_y: {Counter(normal_y)}')
        print(f'abnormal_X: {abnormal_X.shape}, abnormal_y: {Counter(abnormal_y)}')

        ##########################################################################################
        # Step 2: conduct experiments for each case
        for case in cases:
            case['random_state'] = random_state
            case['verbose'] = 5
            case['online'] = online  # online: True, otherwise, batch
            case['q_abnormal_thres'] = 0.9  # default 0.95
            case['batch_size'] = 100  #

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
                X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test, y_test = split_train_arrival_test(
                    normal_X, normal_y,
                    abnormal_X, abnormal_y,
                    one_data_flg=single_device, random_state=random_state)
                data_info(X_init_train, name='X_init_train')
                data_info(X_init_test, name='X_init_test')
                data_info(X_arrival, name='X_arrival')
                data_info(X_test, name='X_test')
                # change multi-labels to binary (normal_i -> normal(0), abnormal_i-> abnormal(1))
                y_init_train = [0 if v.startswith('normal') else 1 for v in y_init_train]
                y_init_test = [0 if v.startswith('normal') else 1 for v in y_init_test]
                y_arrival = [0 if v.startswith('normal') else 1 for v in y_arrival]
                y_test = [0 if v.startswith('normal') else 1 for v in y_test]

                if 'GMM' == case['detector_name']:
                    if case['online']:
                        model = ONLINE_GMM_MAIN(case)
                    else:  # 'batch', i.e., batch_GMM
                        model = BATCH_GMM_MAIN(case)
                else:
                    raise NotImplementedError()
                model.train_test_model(X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival,
                                       X_test, y_test)
                _best_results = model.info  # model.info['train_times']
                _middle_results = {}

                # # save each result first
                # out_file = pth.abspath(f'{out_expand_dir}/{case_str}.csv')
                # print('+++', out_file)
                # save_each_result(_best_results, case_str, out_file)
                #
                # dump_data(_middle_results, out_file + '-middle_results.dat')
                #
                results[(in_dir, case_str)] = (_best_results, _middle_results)
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
