"""Main entrance of batch GMM and online GMM experiments

Run the below command under "examples/"
    PYTHONPATH=../:./ python3.7 main_gmm.py > out/batch_online_gmm.txt 2>&1 &
"""

import copy
import itertools
import os
import os.path as pth
import traceback
from collections import Counter, OrderedDict
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.metrics import roc_curve, pairwise_distances
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters

from online.config import *
from online.generate_data import generate_data, split_train_arrival_test
from kjl.model.gmm import GMM
from kjl.model.kjl import KJL
from kjl.model.nystrom import NYSTROM
from kjl.model.online_gmm import ONLINE_GMM
from kjl.model.seek_mode import MODESEEKING
from kjl.preprocessing.standardization import STD
from kjl.utils.data import load_data, batch, data_info
from kjl.utils.parameters import PARAM
from kjl.utils.tool import execute_time, time_func, mprint
from online.report import imgs2xlsx


class SINGLE_CASE:

    def __init__(self, random_state=42, **kwargs):
        self.random_state = random_state

        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_data(self, data_file, params):
        if self.overwrite:
            if pth.exists(data_file): os.remove(data_file)
        if not pth.exists(data_file):
            data_file = generate_data(params.data_name, params.data_type, out_file=data_file,
                                      overwrite=params.overwrite, random_state=params.random_state)
        X, y = load_data(data_file)
        return X, y

    def run(self, params):
        info = {}

        params.data_type = 'two_datasets'
        X, y = self.load_data(params.data_file, params)
        params.X_init_train, params.y_init_train, params.X_init_test, params.y_init_test, \
        params.X_arrival, params.y_arrival, params.X_test, params.y_test = \
            split_train_arrival_test(X, y, params)
        for is_online in [True, False]:
            params.is_online = is_online
            _info = {}
            for i in range(params.n_repeats):
                if params.is_online:
                    model = ONLINE_GMM_MAIN(params)
                    # model = BATCH_GMM_MAIN(params)
                else:
                    model = BATCH_GMM_MAIN(params)
                params.random_state = (i + 1) * self.random_state
                model.random_state = params.random_state
                model.params.random_state = params.random_state
                model.init_train_test(params.X_init_train, params.y_init_train, params.X_init_test, params.y_init_test)
                _info[i] = model.batch_train_test(params.X_arrival, params.y_arrival, params.X_test, params.y_test)
            info['online' if is_online else 'batch'] = _info
        self.info = info

    def display(self, info, out_file, key=(), is_show=False):

        online_info = info['online']
        batch_info = info['batch']
        dataset_name, data_file, ratio, experiment_case = key

        def get_values_of_key(info, name='train_times'):
            vs = []
            for i_repeat, res in info.items():
                # train_times = res['train_times']
                # test_times = res['test_times']
                # acus = res['aucs']
                # n_components = res['n_components']
                if name == 'train_times':
                    v = [_v['train_time'] for _v in res.info[name]]
                elif name == 'test_times':
                    v = [_v['test_time'] for _v in res.info[name]]
                elif name == 'n_components':
                    v = [_v[name] for _v in res.info['model_params']]
                else:
                    v = res.info[name]
                vs.append(np.asarray(v))

            return np.asarray(vs)

        def _plot(ax, online_train_times, batch_train_times, xlabel, ylabel, title, out_file, ylim=[],
                  legend_position='upper right'):

            y = batch_train_times
            x = range(y.shape[1])
            yerr = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            ax.errorbar(x, y, yerr, ecolor='r', capsize=2, linestyle='-', marker='.', color='green',
                        markeredgecolor='green', markerfacecolor='green', label=f'Batch', alpha=0.9)  # marker='*',

            y = online_train_times
            yerr = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            ax.errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                        markeredgecolor='blue', markerfacecolor='blue', label=f'Online', alpha=0.9)  # marker='*',

            # plt.xlim([0.0, 1.0])
            if len(ylim) == 2:
                ax.set_ylim(ylim)  # [0.0, 1.05]
            # ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # plt.xticks(x)
            # plt.yticks(y)
            ax.legend(loc=legend_position)
            # ax.set_title(title)

        params = online_info[0].info['params']
        print(f'\n***{dataset_name}, {experiment_case}')
        if 'online:False' in experiment_case:
            online = False
            params.incorporated_points = 0
            params.fixed_U_size = False
        else:
            online = True

        n_point = params.incorporated_points
        n_point = f'{n_point}' if n_point > 1 else f'{n_point}'

        fixed_kjl = params.fixed_kjl
        fixed_U_size = params.fixed_U_size
        n_components_init = params.n_components
        covariance_type = params.covariance_type
        q_kjl = params.q_kjl
        n_kjl = params.n_kjl
        d_kjl = params.d_kjl
        std = params.std
        kjl = params.kjl
        n_repeats = params.n_repeats
        # dataset_name, data_file = k_dataset
        dataset_name = f'{dataset_name} (init_set={int(params.percent_first_init * 100)}:{int(round((1 - params.percent_first_init) * 100))}-{params.X_init_train.shape})'

        if kjl:
            title = f'n_comp={n_components_init}, {covariance_type}; std={std}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}; n_repeats={n_repeats}'
            if fixed_kjl:
                title = f'Batch vs. Online GMM with fixed KJL on {dataset_name}\n{title}'
            elif fixed_U_size:
                # (replace {n_point} cols and rows of U)
                title = f'Batch vs. Online GMM with fixed U size on {dataset_name}\n{title}'
            else:  # increased_U
                title = f'Batch vs. Online GMM with increased U size on {dataset_name}\n{title}'
        else:
            title = f'n_comps={n_components_init}, {covariance_type}; KJL={kjl}; n_repeats={n_repeats}'
            title = f'Batch vs. Online GMM on {dataset_name}\n({title})'

        batch_size = params.batch_size
        xlabel = f'The ith batch: i * batch_size({batch_size}) datapoints'

        fig, ax = plt.subplots(nrows=2, ncols=2)
        # train_times
        online_train_times = get_values_of_key(online_info, name='train_times')
        batch_train_times = get_values_of_key(batch_info, name='train_times')

        _plot(ax[0, 0], online_train_times, batch_train_times, xlabel=xlabel, ylabel='Training time (s)', title=title,
              out_file=out_file.replace('.pdf', '-train_times.pdf'))
        # test times
        online_test_times = get_values_of_key(online_info, name='test_times')
        batch_test_times = get_values_of_key(batch_info, name='test_times')
        _plot(ax[0, 1], online_test_times, batch_test_times, xlabel=xlabel, ylabel='Testing time (s)', title=title,
              out_file=out_file.replace('.pdf', '-test_times.pdf'))

        # aucs
        online_aucs = get_values_of_key(online_info, name='aucs')
        batch_aucs = get_values_of_key(batch_info, name='aucs')
        _plot(ax[1, 0], online_aucs, batch_aucs, xlabel=xlabel, ylabel='AUCs', title=title,
              legend_position='lower right', ylim=[0.0, 1.05],
              out_file=out_file.replace('.pdf', '-aucs.pdf'))

        # n_components
        online_n_components = get_values_of_key(online_info, name='n_components')
        batch_n_components = get_values_of_key(batch_info, name='n_components')
        _plot(ax[1, 1], online_n_components, batch_n_components, xlabel=xlabel, ylabel='n_components', title=title,
              legend_position='lower right',
              out_file=out_file.replace('.pdf', '-n_components.pdf'))

        fig.suptitle(title, fontsize=11)

        plt.tight_layout()  # rect=[0, 0, 1, 0.95]
        try:
            plt.subplots_adjust(top=0.9, bottom=0.1, right=0.975, left=0.12)
        except Warning as e:
            raise ValueError(e)
        #
        # fig.text(.5, 15, "total label", ha='center')
        plt.figtext(0.5, 0.01, f'X-axis:({xlabel})', fontsize=11, va="bottom", ha="center")
        print(out_file)
        if not pth.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
        fig.savefig(out_file, format='pdf', dpi=300)
        fig.savefig(out_file + '.png', format='png', dpi=300)
        if is_show: plt.show()
        plt.close(fig)

        return out_file


def add_offset(data, name='means', offset=1e-3):
    if name == 'weights':
        if np.any(data <= offset):
            new_data = (data + offset) / np.sum(data)
        else:
            new_data = data
    elif name == 'means':
        new_data = []
        for vs in data:
            if np.any(abs(data) <= offset):
                new_data.append(np.array([v + offset if v > 0 else v - offset for v in vs]))
            else:
                new_data.append(vs)
        new_data = np.array(new_data, dtype=float)
    elif name == 'covariances':
        new_data = []
        for vs in data:
            if np.any(abs(data) <= offset):
                new_data.append(np.array([v + offset if v > 0 else v - offset for v in vs]))
            else:
                new_data.append(vs)
        new_data = np.array(new_data, dtype=float)
    else:
        pass

    return new_data


class BASE_MODEL:

    def init_train_test(self, X_init_train, y_init_train, X_init_test, y_init_test):

        ##########################################################################################
        # Step 1. configure parameters for tuning
        # case = 'GMM_full-gs_True-kjl_True-nystrom_False-quickshift_False-meanshift_False'
        if self.params.detector_name == 'GMM' and self.params.covariance_type == 'diag' and \
                self.params.gs:  # self.params['kjl']  and not self.params.quickshift  and not self.params.meanshift
            params = {}
            # best params for the combination of UNB1 and UNB2
            # params['n_components'] = [5]  # UNB2
            # params['qs_kjl'] = [0.7]

            # params['n_components'] = [2]  # mimic_GMM_dataset
            params['qs_kjl'] = [0.3]

            # # # # # # grid search
            params['n_components'] = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
            # params['qs_kjl'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

            # fixed n and d
            # params['ns_kjl'] = [100]
            # params['ds_kjl'] = [10]

            params['ns_kjl'] = [self.params.n_kjl]
            params['ds_kjl'] = [self.params.d_kjl]

        else:
            raise ValueError(self.params)

        ##########################################################################################
        # Step 2. Find the best parameters
        best_auc = -1
        for n_components, d_kjl, n_kjl, q_kjl in list(itertools.product(params['n_components'],
                                                                        params['ds_kjl'], params['ns_kjl'],
                                                                        params['qs_kjl'])):
            self.params.n_components = n_components
            self.params.d_kjl = d_kjl
            self.params.n_kjl = n_kjl
            self.params.q_kjl = q_kjl
            self.params.ratio_kjl = self.params.n_kjl / X_init_train.shape[0]  # for updating kjl when U increases
            self.model = GMM(n_components=n_components, covariance_type=self.params.covariance_type,
                             random_state=self.params.random_state)

            # Fit a model on the init train set with self.params
            self._init_train(X_init_train, y_init_train)
            # Evaluate the model on the test set and get AUC
            self._init_test(X_init_test, y_init_test)
            mprint(f'AUC: {self.auc} with {self.model.get_params()}, and self.q_kjl: {q_kjl}', self.verbose, DEBUG)
            # find the best one
            if best_auc < self.auc:  # here should be <, not <=
                best_auc = self.auc
                best_model_params = copy.deepcopy(self.model.get_params())  # only GMM parameters
                best_params = copy.deepcopy(self.params)  # inculdes all params, but part of model paramters.

        ##########################################################################################
        # Step 3. To get the best model with best_params, best_auc, best train_time and test_time
        mprint(f'\n***The best params: model_params: {best_model_params}, params: {best_params}')
        self.params = best_params
        self.model = GMM()
        model_params = {'n_components': best_model_params['n_components'],
                        'covariance_type': best_model_params['covariance_type'],
                        'means_init': None, 'random_state': best_model_params['random_state']}
        self.model.set_params(**model_params)

        # Fit self.model on the init set
        self._init_train(X_init_train, y_init_train)
        # Evaluate the model on the test set
        self._init_test(X_init_test, y_init_test)
        mprint(f'***init result: train_time: {self.train_time}, abnormal_thres: {self.abnormal_thres}, '
               f'test_time: {self.test_time}, AUC: {self.auc},', self.verbose, DEBUG)

        ##########################################################################################
        # Step 4. Store all important results
        self.init_info = {'train_time': self.train_time, 'abnormal_thres': self.abnormal_thres,
                          'test_time': self.test_time, 'auc_time': self.auc,
                          'X_train_shape': X_init_train.shape, 'X_test_shape': X_init_test.shape,
                          'params': self.params, 'model_params': self.model.get_params()}
        return self.model

    def _init_train(self, X_init_train, y_init_train=None):
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

        train_time = 0.0

        self.X_init_train = X_init_train
        self.y_init_train = y_init_train

        ##########################################################################################
        # Step 1: Preprocessing the data, which includes standardization, mode seeking, and kernel projection.
        if self.params.std:
            # Step 1.1: Standardize the data first
            # fit std_inst
            self.std_inst = STD()
            _, std_fitting_time = time_func(self.std_inst.fit, X_init_train)
            # transform X_train
            X_init_train, std_time = time_func(self.std_inst.transform, X_init_train)
            # mprint(f'mu: {self.std_inst.scaler.mean_},std_var: {self.std_inst.scaler.scale_}', self.verbose, DEBUG)
            std_time += std_fitting_time
        else:
            std_time = 0.0
        train_time += std_time

        # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
        time_seeking = 0.0
        # self.thres_n = 100  # used to filter clusters which have less than 100 datapoints
        # if 'meanshift' in self.params.keys() and self.params['meanshift']:
        #     dists = pairwise_distances(X_train)
        #     self.sigma = np.quantile(dists, self.params['q_kjl'])  # also used for kjl
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
        train_time += time_seeking

        # Step 1.3. Project the data onto a lower space with KJL or Nystrom
        if self.params.kjl:
            # Fit a kjl_inst on the X_train
            self.kjl_inst = KJL(self.params)
            _, kjl_time = time_func(self.kjl_inst.fit, X_init_train, y_init_train, self.X_init_train, self.y_init_train)
            proj_time = kjl_time
            # Transform X_train
            X_init_train, kjl_time = time_func(self.kjl_inst.transform, X_init_train)
            proj_time += kjl_time
        elif self.params.nystrom:
            # Fit a nystrom on the X_train
            self.nystrom_inst = NYSTROM(self.params)
            _, nystrom_time = time_func(self.nystrom_inst.fit, X_init_train, y_init_train)
            proj_time = nystrom_time
            # Transform X_train
            X_init_train, nystrom_time = time_func(self.nystrom_inst.transform, X_init_train)
            proj_time += nystrom_time
        else:
            proj_time = 0.0
        train_time += proj_time
        if self.verbose > DEBUG: data_info(X_init_train, name='X_proj before updating KJL')

        ##########################################################################################
        # Step 2. Setting self.model'params (e.g., means_init, and n_components when use meanshift or quickshift)
        model_params = {'n_components': self.params.n_components,
                        'covariance_type': self.params.covariance_type,
                        'means_init': None, 'random_state': self.random_state}
        self.model.set_params(**model_params)
        # mprint(self.model.get_params(), self.verbose, DEBUG)
        # Fit self.model on the X_train
        _, model_fitting_time = time_func(self.model.fit, X_init_train)
        train_time += model_fitting_time

        ##########################################################################################
        # Step 3. Get the threshold used to decide if a new flow is normal
        # the following values will be used in the online update phase
        y_score, _ = time_func(self.model.decision_function, X_init_train)
        self.abnormal_thres = np.quantile(y_score, q=self.params.q_abnormal_thres)  # abnormal threshold
        _, log_resp = self.model._e_step(X_init_train)
        self.model.sum_resp = np.sum(np.exp(log_resp), axis=0)
        self.model.y_score = y_score
        # self.X_init_train_proj = X_init_train

        self.train_time = {'train_time': train_time,
                           'preprocessing_time': std_time + proj_time,
                           # 'mode_seeking': time_seeking,
                           'model_fitting_time': model_fitting_time,
                           'rescore_time': 0}

        return self

    def _init_test(self, X_init_test, y_init_test):
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

        test_time = 0.0
        print(f'y_test: {Counter(y_init_test)}')
        ##########################################################################################
        # Step 1: Preprocessing
        X_init_test, preprocessing_time = time_func(self._preprocessing, X_init_test)
        test_time += preprocessing_time
        # if self.verbose >= WARNING:
        #     # data_info(X_init_test, name='X_init_test')
        #     # print('self.kjl_inst.U', self.kjl_inst.U)
        #     print(preprocessing_time)

        ##########################################################################################
        # Step 2: Evaluate GMM on the test set
        y_score, prediction_time = time_func(self.model.decision_function, X_init_test)
        self.auc, auc_time = time_func(self.get_score, y_init_test, y_score)
        if self.verbose >= DEBUG: data_info(np.asarray(y_score).reshape(-1, 1), name='y_score')
        test_time += prediction_time + auc_time

        mprint(f'Total test time: {test_time} <= preprocessing_time: {preprocessing_time}, '
               f'prediction_time: {prediction_time}, auc_time: {auc_time}, AUC: {self.auc}', self.verbose, DEBUG)

        self.test_time = {'test_time': test_time,
                          'preprocessing_time': preprocessing_time,
                          'prediction_time': prediction_time,
                          'auc_time': auc_time}

        return self

    def get_score(self, y_test, y_score):
        y_test = [0 if v.startswith('normal') else 1 for v in y_test]
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return auc

    def _preprocessing(self, X):
        # Step 1.1: std time
        if self.params.std:
            X, std_time = time_func(self.std_inst.transform, X)
        else:
            std_time = 0.0
        # Step 1.2: project time
        if self.params.kjl:
            X, proj_time = time_func(self.kjl_inst.transform, X)
        elif self.params.nystrom:
            X, proj_time = time_func(self.nystrom_inst.transform, X)
        else:
            proj_time = 0

        print(f'std_time: {std_time}, proj_time: {proj_time}')
        return X


class ONLINE_GMM_MAIN(BASE_MODEL, ONLINE_GMM):

    def __init__(self, params):
        """Main class of online GMM experiment

        Parameters
        ----------
        params
        """
        self.params = params
        self.random_state = params.random_state
        self.verbose = params.verbose
        # stores all results
        self.info = {}

    def batch_train_test(self, X_arrival, y_arrival, X_test,
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
        ##########################################################################################
        # # # Step 1. Get best initial model (init_model) on initial set (init_set) and evaluate it on init test set
        # self.init_model = self.init_train_test(X_init_train, y_init_train, X_init_test, y_init_test)
        self.init_model = self.model

        ##########################################################################################
        # Step 2. Online train and evaluate model
        self.info['train_times'] = [self.init_info['train_time']]
        self.info['abnormal_threses'] = [self.init_info['abnormal_thres']]
        self.info['model_params'] = [self.init_info['model_params']]
        self.info['test_times'] = [self.init_info['test_time']]
        self.info['aucs'] = [self.init_info['auc_time']]

        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=self.params.batch_size)):
            if i == 0:
                # copy means_, covariances, and other params from the init_model
                self.model = copy.deepcopy(self.init_model)
                # self.X_acculumated_train_proj = self.X_init_train_proj
                self.X_acculumated_train = self.X_init_train
                self.y_acculumated_train = self.y_init_train

            # online train model (update GMM model values, such as, means, covariances, kjl_U, and n_components)
            self.online_train(X_batch, Y_batch)  # update self.model
            self.info['train_times'].append(self.train_time)
            self.info['model_params'].append(self.model.get_params())
            self.info['abnormal_threses'].append(self.abnormal_thres)

            # online test model
            self.online_test(X_test, y_test)
            self.info['test_times'].append(self.test_time)
            self.info['aucs'].append(self.auc)

            mprint(f'batch_{i + 1}: train_time: {self.train_time}, '
                   f'test_time: {self.test_time}, auc: {self.auc}', self.verbose, WARNING)

        self.info['params'] = self.params

        return self

    def online_train(self, X_batch, y_batch=None):
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
        mprint(f'y_batch: {Counter(y_batch)}', self.verbose, DEBUG)
        train_time = 0.0
        X_batch_raw = X_batch
        y_batch_raw = y_batch

        ##########################################################################################
        start = datetime.now()
        # Step 1: Predict on the arrival data  first.
        # Use the fitted model to predict X_batch first, and according to the result,
        # only the normal data will be used to train a new model, which will be used to replace the current model.
        # Step 1.1: Preprocessing: std and projection
        _X_batch, preprocessing_time = time_func(self._preprocessing, X_batch)
        train_time += preprocessing_time
        # Step 1.2: Obtain the abnormal score (note that a larger value is for outlier (positive))
        y_score, prediction_time = time_func(self.model.decision_function, _X_batch)

        ##########################################################################################
        # Step 2.1. Preprocessing
        # Update std and proj, and get the new X_data
        if self.params.fixed_kjl:  # fixed KJL, U, and STD
            # use the self.kjl_inst obtained on the init_set
            std_time = 0.0
            proj_time = 0.0
            X_batch = _X_batch
        else:
            if self.params.std:
                # update the mean and scaler of self.std_inst with 'x'
                # use the original X_batch to update std_inst
                _, time_std_update = time_func(self.std_inst.update, X_batch_raw)
                X_batch, std_time = time_func(self.std_inst.transform, X_batch_raw)
                std_time += time_std_update
            else:
                self.std_inst = None
                X_batch = X_batch_raw
                std_time = 0.0
            # Step 2.2: update projection: kjl or nystrom
            if self.params.kjl:
                #  Update kjl: self.U_kjl, self.Xrow_kjl.
                self.kjl_inst.n_samples = self.X_acculumated_train.shape[0]
                _, time_proj_update = time_func(self.kjl_inst.update, X_batch, y_batch, X_batch_raw, y_batch_raw,
                                                self.std_inst)
                X_batch, proj_time = time_func(self.kjl_inst.transform, X_batch)
                proj_time += time_proj_update
            elif self.params.nystrom:
                # Update nystrom_inst
                _, time_proj_update = time_func(self.nystrom_inst.update, X_batch)
                X_batch, proj_time = time_func(self.nystrom_inst.transform, X_batch)
                proj_time += time_proj_update
            else:
                proj_time = 0

        if not self.params.kjl:
            self.params.incorporated_points = 0
            self.params.fixed_U_size = False
        else:
            self.params.incorporated_points = self.kjl_inst.t
            self.params.fixed_U_size = self.kjl_inst.fixed_U_size

        self.abnormal_thres = np.infty  # for debug

        normal_idx = np.where((y_score <= self.abnormal_thres) == True)
        abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
        X_batch_normal = X_batch[normal_idx] if len(normal_idx[0]) > 0 else []
        X_batch_abnormal = X_batch[abnormal_idx] if len(abnormal_idx[0]) > 0 else []
        # without std and kjl
        X_batch_normal_raw = X_batch_raw[normal_idx] if len(normal_idx[0]) > 0 else []
        X_batch_abnormal_raw = X_batch_raw[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

        # reproject the previous X_acculumated_train_proj with the updated std and kjl
        X_acculumated_train_proj = self._preprocessing(self.X_acculumated_train)

        end = datetime.now()
        preprocessing_time = (end - start).total_seconds()
        train_time += preprocessing_time

        start = datetime.now()
        # # Step 2.3: Seek modes of the data by quickshift++ or meanshift for initialize GMM
        n_thres = 0  # used to filter clusters which have less than 100 datapoints
        if self.params.meanshift:
            if self.params.kjl:  # use the same sigma of kjl
                self.sigma = self.kjl_inst.sigma_kjl
            else:
                dists = pairwise_distances(X_acculumated_train_proj)
                self.sigma = np.quantile(dists, q=0.3)
            ms = MODESEEKING(method_name='meanshift', bandwidth=None,
                             random_state=self.random_state, verbose=self.verbose)
            _, ms_fitting_time = time_func(ms.fit, X_acculumated_train_proj, n_thres=n_thres)
            n_components = ms.n_clusters_
            mprint(f'mode_seeking_time: {ms_fitting_time}s, n_clusters: {n_components}', self.verbose, DEBUG)
            n_samples, _ = X_acculumated_train_proj.shape
            resp = np.zeros((n_samples, n_components))
            # label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
            #                        random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), ms.labels_] = 1

            weights, means, covariances = _estimate_gaussian_parameters(
                X_acculumated_train_proj, resp, self.model.reg_covar, self.model.covariance_type)
            weights /= n_samples
            new_model = ONLINE_GMM(n_components=n_components,
                                   covariance_type=self.model.covariance_type,
                                   weights_init=weights,
                                   means_init=means,
                                   covariances_init=covariances,
                                   verbose=self.verbose,
                                   random_state=self.params.random_state,
                                   warm_start=True)

        elif self.params.quickshift:
            ms = MODESEEKING(method_name='quickshift', k=100, beta=0.9,
                             random_state=self.random_state, verbose=self.verbose)
            _, ms_fitting_time = time_func(ms.fit, X_acculumated_train_proj, n_thres=n_thres)
            n_components = ms.n_clusters_
            mprint(f'mode_seeking_time: {ms_fitting_time}s, n_clusters: {n_components}', self.verbose, DEBUG)
            n_samples, _ = X_acculumated_train_proj.shape
            resp = np.zeros((n_samples, self.model.n_components))
            # label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
            #                        random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), ms.labels_] = 1

            weights, means, covariances = _estimate_gaussian_parameters(
                X_acculumated_train_proj, resp, self.model.reg_covar, self.model.covariance_type)
            weights /= n_samples
            new_model = ONLINE_GMM(n_components=n_components,
                                   covariance_type=self.model.covariance_type,
                                   weights_init=weights,
                                   means_init=means,
                                   covariances_init=covariances,
                                   verbose=self.verbose,
                                   random_state=self.params.random_state,
                                   warm_start=True)
        else:
            # print(f'before online, self.model.weights_: {self.model.weights_}')
            # print(f'before online, self.model.means_: {self.model.means_}')
            # print(f'before online, self.model.covariances_: {self.model.covariances_}')
            new_model = ONLINE_GMM(n_components=self.model.n_components,
                                   covariance_type=self.model.covariance_type,
                                   weights_init=self.model.weights_,
                                   means_init=self.model.means_,
                                   covariances_init=self.model.covariances_,
                                   verbose=self.verbose,
                                   random_state=self.params.random_state,
                                   warm_start=True)

        # Step 2.4: online train a new model on the batch data
        # 1) train on the normal data first (without needing to create any new component)
        # 2) train on the abnormal data (create new components)
        # set the default params of ONLINE_GMM
        # new_model.sum_resp = self.model.sum_resp  # shape (1, d): sum of exp(log_resp)
        # new_model._initialize()  # set up cholesky

        update_flg = False
        # The first update of GMM
        # X_batch_used = np.zeros()
        while (len(X_batch_normal) > 0) or (len(X_batch_abnormal) > 0):
            if len(X_batch_normal) > 0:
                # # 1) train on the normal data first (without needing to create any new component)
                # _, log_resp = new_model._e_step(X_batch_normal)
                # # data_info(np.exp(log_resp), name='resp')
                # new_model._m_step_online(X_batch_normal, log_resp, sum_resp_pre=new_model.sum_resp,
                #                          n_samples_pre=self.X_acculumated_train_proj.shape[0])
                self.X_acculumated_train = np.concatenate([self.X_acculumated_train, X_batch_normal_raw], axis=0)
                X_batch_normal_raw = []
                X_acculumated_train_proj = np.concatenate([X_acculumated_train_proj, X_batch_normal], axis=0)
                X_batch_normal = []

            # elif len(X_batch_abnormal) > 0:
            #     update_flg = True
            #     # 2) train on the abnormal data (create new components)
            #     # each time only focuses on one abnormal datapoint
            #     idx = 0
            #     # add a new data with std and kjl into self.X_acculumated_train_proj
            #     x_proj = X_batch_abnormal[idx].reshape(1, -1)
            #     X_batch_abnormal = np.delete(X_batch_abnormal, idx, axis=0)
            #     X_acculumated_train_proj = np.concatenate([X_acculumated_train_proj, x_proj], axis=0)
            #     # add a new data without std and kjl into self.X_acculumated_train
            #     x_batch_raw = X_batch_abnormal_raw[idx].reshape(1, -1)
            #     X_batch_abnormal_raw = np.delete(X_batch_abnormal_raw, idx, axis=0)
            #     self.X_acculumated_train = np.concatenate([self.X_acculumated_train, x_batch_raw], axis=0)
            #
            #     # add new components and update threshold
            #     new_model.add_new_component(x_proj, self.params.q_abnormal_thres, X_acculumated_train_proj)
            # else:
            #     break
            #
            # if len(X_batch_abnormal) > 0:
            #     if not update_flg:  # update threshold
            #         self.abnormal_thres = self.update_abnormal_thres(new_model, X_acculumated_train_proj)
            #
            #     y_score = new_model.decision_function(X_batch_abnormal)
            #     # cond_normal = y_score < self.novelty_thres  # return bool values:  normal index
            #     normal_idx = np.where((y_score <= self.abnormal_thres) == True)
            #     abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
            #     X_batch_normal = X_batch_abnormal[normal_idx] if len(normal_idx[0]) > 0 else []
            #     X_batch_abnormal = X_batch_abnormal[abnormal_idx] if len(abnormal_idx[0]) > 0 else []
            #     # without std and proj
            #     X_batch_normal_raw = X_batch_abnormal_raw[normal_idx] if len(normal_idx[0]) > 0 else []
            #     X_batch_abnormal_raw = X_batch_abnormal_raw[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

        end = datetime.now()
        first_train_time = (end - start).total_seconds()

        start = datetime.now()

        new_model = ONLINE_GMM(n_components=self.model.n_components,
                               covariance_type=self.model.covariance_type,
                               verbose=self.verbose,
                               random_state=self.params.random_state,
                               warm_start=True)
        new_model._initialize_parameters(X_acculumated_train_proj, self.random_state)
        # Train the new model until it converges
        i = 0
        new_model.converged_ = False
        if not new_model.converged_:  new_model.max_iter = 100
        prev_lower_bound = -np.infty
        if self.verbose >= 1: mprint(f'X_acculumated_train_proj: {X_acculumated_train_proj.shape}')
        best_n_iter_ = 0
        while (i < new_model.max_iter) and (not new_model.converged_):
            _st = datetime.now()
            # new_model.weights_ = add_offset(new_model.weights_, name='weights')
            # new_model.means_ = add_offset(new_model.means_, name='means')
            # new_model.covariances_ = add_offset(new_model.covariances_, name='covariances')
            # log_prob_norm, log_resp = new_model._e_step(self.X_acculumated_train_proj)
            (log_prob_norm, log_resp), e_time = time_func(new_model._e_step, X_acculumated_train_proj)

            # get the difference
            lower_bound = new_model._compute_lower_bound(log_resp, np.mean(log_prob_norm))
            # print(f'i: {i}, 1. / np.sqrt(self.model.covariances_): ', 1 / np.sqrt(new_model.covariances_), log_resp,
            #       new_model.means_, new_model.weights_)
            change = lower_bound - prev_lower_bound
            if abs(change) < new_model.tol:
                # if np.all(new_model.means_[0] == 0.0):
                #     new_model.converged_ = False
                #     # new_model.weights_ = self.model.weights_
                #     # new_model.means_ = self.model.means_
                #     # new_model.covariances_ = self.model.covariances_
                #     # new_model._initialize()  # set up cholesky
                # else:
                #     new_model.converged_ = True
                new_model.converged_ = True
                best_n_iter_ = i + 1
                mprint(f'best_n_iter: {i + 1}, new_model.tol:{new_model.tol},abs(change):{abs(change)}')
                # break
            prev_lower_bound = lower_bound
            _, m_time = time_func(new_model._m_step, X_acculumated_train_proj, log_resp)
            # print(
            #     f'i: {i},abs(change):{abs(change)}, prev_lower_bound: {prev_lower_bound}, log_resp: {log_resp}, log_prob_norm: {log_prob_norm}, new_model.weights_ {new_model.weights_}, new_model.means_ {new_model.means_},new_model.covariances_ {new_model.covariances_}')
            _end = datetime.now()
            _tot = (_end - _st).total_seconds()
            if self.verbose > 5: mprint(f'{i + 1}th iterations takes {_tot} seconds, in which e_time: {e_time}, '
                                        f'and m_time: {m_time}', self.verbose, DEBUG)
            i += 1
        new_model.n_iter_ = best_n_iter_
        end = datetime.now()
        iteration_time = (end - start).total_seconds()
        train_time += first_train_time + iteration_time
        # print(f'1. / np.sqrt(self.model.covariances_): ', 1 / np.sqrt(new_model.covariances_), log_resp,
        #       new_model.means_, new_model.weights_)

        # self.model = GMM(n_components=self.model.n_components, covariance_type=self.params.covariance_type,
        #                  random_state=self.params.random_state,  weights_init=self.model.weights_,
        #                            means_init=self.model.means_, precisions_init= 1. / np.sqrt(self.model.covariances_),
        #                  )
        # print(f'1. / np.sqrt(self.model.covariances_): ', self.model.precisions_init)
        # _, model_fitting_time = time_func(self.model.fit, X_acculumated_train_proj)
        # log_prob_norm=0
        # first_train_time=0
        # iteration_time = model_fitting_time
        # self.model.sum_resp=0  # shape (1, d): sum of exp(log_resp)

        ##########################################################################################
        # Step 3:  update the abnormal threshold with all accumulated data
        self.model = new_model
        mprint(new_model.get_params(), self.verbose, DEBUG)
        if not new_model.converged_:
            self.abnormal_thres, rescore_time = time_func(self.update_abnormal_thres,
                                                          self.model,
                                                          X_acculumated_train_proj)
        else:
            # override the _e_step(), so  here is not mean(log_prob_norm)    #  return -1 * self.score_samples(X)
            y_score = - log_prob_norm
            self.abnormal_thres, rescore_time = time_func(np.quantile, y_score, q=self.params.q_abnormal_thres)
        train_time += rescore_time

        mprint(f'Batch time: {train_time} <=: preprocessing_time: {preprocessing_time},'
               f'first_train_time: {first_train_time}, iteration_time: {iteration_time},'
               f'rescore_time: {rescore_time}')

        self.train_time = {'train_time': train_time,
                           'preprocessing_time': preprocessing_time,
                           'model_fitting_time': first_train_time + iteration_time,
                           'rescore_time': rescore_time}

        mprint(f'n_iter: {self.model.n_iter_}', self.verbose, DEBUG)

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

        y_score, model_predict_time = time_func(model.decision_function, X_normal_proj)
        abnormal_thres = np.quantile(y_score, q=self.params.q_abnormal_thres)  # abnormal threshold
        # model.y_score = y_score

        return abnormal_thres

    def online_test(self, X_test, y_test):
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
        self._init_test(X_test, y_test)


class BATCH_GMM_MAIN(BASE_MODEL):

    def __init__(self, params):
        """Main class of online GMM experiment

        Parameters
        ----------
        params: dict
            the parameters for this experiment
        """
        self.params = params

        self.random_state = params.random_state
        self.verbose = params.verbose

        # stores important results
        self.info = {}

    def batch_train_test(self, X_arrival, y_arrival, X_test,
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
        # ##########################################################################################
        # # # Step 1. Get initial model (init_model) on initial set (init_set) and evaluate it on test set
        # self.init_model = self.init_train_test(X_init_train, y_init_train, X_init_test, y_init_test)
        self.init_model = self.model

        ##########################################################################################
        # Step 2. train the model on the batch data (previous+batch) and evaluate it.
        self.info['train_times'] = [self.init_info['train_time']]
        self.info['abnormal_threses'] = [self.init_info['abnormal_thres']]
        self.info['model_params'] = [self.init_info['model_params']]
        self.info['test_times'] = [self.init_info['test_time']]
        self.info['aucs'] = [self.init_info['auc_time']]
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=self.params.batch_size)):
            if i == 0:
                self.model = copy.deepcopy(self.init_model)
                self.X_acculumated_train = self.X_init_train
                self.y_acculumated_train = self.y_init_train

            self.batch_train(X_batch, Y_batch)
            self.info['train_times'].append(self.train_time)
            self.info['model_params'].append(self.model.get_params())
            self.info['abnormal_threses'].append(self.abnormal_thres)

            # batch test model
            self.batch_test(X_test, y_test)
            self.info['test_times'].append(self.test_time)
            self.info['aucs'].append(self.auc)

            mprint(f'batch_{i + 1}: train_time: {self.train_time}, '
                   f'test_time: {self.test_time}, auc: {self.auc}', self.verbose, WARNING)

        self.info['params'] = self.params

        return self

    def batch_train(self, X_batch, y_batch=None):

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
        start = datetime.now()

        mprint(f'y_batch: {Counter(y_batch)}', self.verbose, DEBUG)
        train_time = 0.0
        X_batch_raw = X_batch

        ##########################################################################################
        # Step 1: Preprocessing
        _X_batch, preprocessing_time = time_func(self._preprocessing, X_batch)
        train_time += preprocessing_time
        ##########################################################################################
        # Step 2: Evaluate GMM on the test set
        y_score, prediction_time = time_func(self.model.decision_function, _X_batch)
        train_time += prediction_time

        ##########################################################################################
        # Step 3. Batch train a GMM from scratch
        # and use it to replace the previous one.
        # Step 3.1: concatenate previous data and X_batch (only normal data)
        self.abnormal_thres = np.infty  # for debug

        normal_idx = np.where((y_score <= self.abnormal_thres) == True)
        abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
        if len(normal_idx[0]) > 0:
            X_normal = X_batch_raw[normal_idx]
            y_normal = y_batch[normal_idx]
            self.X_acculumated_train = np.concatenate([self.X_acculumated_train, X_normal], axis=0)
            self.y_acculumated_train = np.concatenate([self.y_acculumated_train, y_normal], axis=0)
            abnormal_cnt = len(abnormal_idx[0])
        else:
            abnormal_cnt = len(abnormal_idx[0])
        mprint(f'X_train: {self.X_acculumated_train.shape}, drop {abnormal_cnt} abnormal flows.')
        # the datapoint is predicted as a abnormal flow, so we should drop it.
        mprint(f'{abnormal_cnt} flows are predicted as abnormal, so we drop them.', self.verbose, DEBUG)

        # Step 3.2: train a new model and use it to instead of the current one.
        # we use model.n_compoents to initialize n_components or the value found by quickhsift++ or meanshift
        # self.model_copy = copy.deepcopy(self.model)
        self.model = GMM(n_components=self.model.n_components, covariance_type=self.params.covariance_type,
                         random_state=self.params.random_state)
        # update self.model: replace the current model with new_model
        if self.params.fixed_kjl:  # (fixed KJL, U, and STD)
            # use the init self.kjl to transform X_batch
            # Step 1: Preprocessing
            X_batch, prepro_time = time_func(self._preprocessing, self.X_acculumated_train)
            preprocessing_time += prepro_time
            ##########################################################################################
            # # Step 2: fit a model
            _, model_fitting_time = time_func(self.model.fit, X_batch)
            train_time += model_fitting_time
            ##########################################################################################
            # Step 3: get the threshold used to decide if a new flow is normal
            # the following values will be used in the online update phase
            y_score, prediction_time = time_func(self.model.decision_function, X_batch)
            self.abnormal_thres, rescore_time = time_func(np.quantile, y_score, q=self.params.q_abnormal_thres)
            train_time += prediction_time

            end = datetime.now()
            train_time = (end - start).total_seconds()
            self.train_time = {'train_time': train_time,
                               'preprocessing_time': preprocessing_time,
                               'model_fitting_time': model_fitting_time,
                               'rescore_time': rescore_time}
        else:
            end = datetime.now()
            train_time = (end - start).total_seconds()
            self._init_train(self.X_acculumated_train, self.y_acculumated_train)  # self.train_time
            self.train_time = {'train_time': train_time + self.train_time['train_time'],
                               'preprocessing_time': preprocessing_time + self.train_time['preprocessing_time'],
                               'model_fitting_time': self.train_time['model_fitting_time'],
                               'rescore_time': self.train_time['rescore_time']}

        mprint(f'n_iter: {self.model.n_iter_}', self.verbose, DEBUG)
        mprint(f'self.train_time: {self.train_time}', self.verbose, 0)

        return self

    def batch_test(self, X_batch, y_batch):
        self._init_test(X_batch, y_batch)


def _main(X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test,
          y_test, params):
    results = []
    random_state = params.random_state
    for i_repeat_idx in range(params.n_repeats):
        # changing random_state affects KJL
        params.random_state = random_state * (i_repeat_idx + 1)
        if params.detector_name == 'GMM':
            if params.online:
                md = ONLINE_GMM_MAIN(params)
            else:
                md = BATCH_GMM_MAIN(params)
        else:
            raise NotImplementedError()

        md.train_test_model(X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test,
                            y_test)
        results.append((md.info, {}))

    return results


def generate_experiment_cases(online=True, gs=True, n_repeats=5, q_abnormal_thres=1.0, fixed_kjl=False,
                              verbose=10, batch_size=100, std=True, random_state=42):
    TEMPLATE = {'detector_name': '', 'gs': False, 'std': std, 'kjl': False, 'nystrom': False, 'quickshift': False,
                'meanshift': False, 'online': online, 'random_state': random_state, 'n_repeats': n_repeats,
                'q_abnormal_thres': q_abnormal_thres, 'verbose': verbose, 'batch_size': batch_size,
                'fixed_kjl': fixed_kjl}

    def create_case(template=TEMPLATE, **kwargs):
        experiment_case = copy.deepcopy(template)
        for k, v in kwargs.items():
            experiment_case[k] = v

        return experiment_case

    experiment_cases = [
        # case 1: OCSVM-gs:True
        # {'detector_name': 'OCSVM', 'gs': gs},

        # # GMM-gs:True
        # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs},
        # {'detector_name': 'GMM', 'covariance_type': 'diag', 'gs': gs},
        #
        # # # GMM-gs:True-kjl:True
        # create_case(template=TEMPLATE, detector_name='GMM', covariance_type='full', gs=True, kjl=True)
        create_case(template=TEMPLATE, detector_name='GMM', covariance_type='diag', gs=gs, kjl=True)
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
        # create_case(template=TEMPLATE, detector_name='GMM', covariance_type='diag', gs=gs, kjl=True, meanshift=True)
    ]

    return experiment_cases


@execute_time
def main():
    ##########################################################################################################
    # Step 0. All datasets
    n_init_train = 1000
    in_dir = f'data/feats/n_init_train_{n_init_train}'
    out_dir = '../examples/online/out'
    data_path_mappings = {
        # 'DEMO_IDS': 'DEMO_IDS/DS-srcIP_192.168.10.5',
        'mimic_GMM': f'{in_dir}/mimic_GMM_dataset/Xy-normal-abnormal.dat',

        'UNB1_UNB2': f'{in_dir}/UNB1_UNB2/Xy-normal-abnormal.dat',
        'UNB1_UNB3': f'{in_dir}/UNB1_UNB3/Xy-normal-abnormal.dat',
        'UNB1_UNB4': f'{in_dir}/UNB1_UNB4/Xy-normal-abnormal.dat',
        'UNB1_UNB5': f'{in_dir}/UNB1_UNB5/Xy-normal-abnormal.dat',
        'UNB2_UNB3': f'{in_dir}/UNB2_UNB3/Xy-normal-abnormal.dat',
        'UNB1_CTU1': f'{in_dir}/UNB1_CTU1/Xy-normal-abnormal.dat',
        'UNB1_MAWI1': f'{in_dir}/UNB1_MAWI1/Xy-normal-abnormal.dat',
        'UNB2_CTU1': f'{in_dir}/UNB2_CTU1/Xy-normal-abnormal.dat',
        'UNB2_MAWI1': f'{in_dir}/UNB2_MAWI1/Xy-normal-abnormal.dat',
        'UNB2_FRIG1': f'{in_dir}/UNB2_FRIG1/Xy-normal-abnormal.dat',
        # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
        'UNB2_FRIG2': f'{in_dir}/UNB_FRIG2/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)

        'CTU1_UNB1': f'{in_dir}/CTU1_UNB1/Xy-normal-abnormal.dat',
        'CTU1_MAWI1': f'{in_dir}/CTU1_MAWI1/Xy-normal-abnormal.dat',
        #
        'UChi_FRIG1': f'{in_dir}/UChi_FRI1/Xy-normal-abnormal.dat',
        # Fridge: (normal: idle, and idle1) abnormal: (open_shut, browse)
        'UChi_FRIG2': f'{in_dir}/UChi_FRIG2/Xy-normal-abnormal.dat',

        'MAWI1_UNB1': f'{in_dir}/MAWI1_UNB1/Xy-normal-abnormal.dat',
        'MAWI1_CTU1': f'{in_dir}/MAWI1_CTU1/Xy-normal-abnormal.dat',  # works
        'MAWI1_UNB2': f'{in_dir}/MAWI1_UNB2/Xy-normal-abnormal.dat',
        'CTU1_UNB2': f'{in_dir}/CTU1_UNB2/Xy-normal-abnormal.dat',
        #
        'UNB1_FRIG1': f'{in_dir}/UNB1_FRIG1/Xy-normal-abnormal.dat',
        # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
        'CTU1_FRIG1': f'{in_dir}/CTU1_FRIG1/Xy-normal-abnormal.dat',
        # # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
        'MAWI1_FRIG1': f'{in_dir}/MAWI1_FRIG1/Xy-normal-abnormal.dat',
        # # MAWI1+Fridge: (normal: idle) abnormal: (open_shut)
        #
        'FRIG1_UNB1': f'{in_dir}/FRIG1_UNB1/Xy-normal-abnormal.dat',
        # # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
        'FRIG1_CTU1': f'{in_dir}/FRIG1_CTU1/Xy-normal-abnormal.dat',
        # # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
        'FRIG1_MAWI1': f'{in_dir}/FRIG1_MAWI1/Xy-normal-abnormal.dat',
        #
        'UNB1_FRIG2': f'{in_dir}/UNB1_FRIG2/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
        'CTU1_FRIG2': f'{in_dir}/CTU1_FRIG2/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (browse)
        'MAWI1_FRIG2': f'{in_dir}/MAWI1_FRIG2/Xy-normal-abnormal.dat',
        # # MAWI1+Fridge: (normal: idle) abnormal: (browse)
        #
        'FRIG2_UNB1': f'{in_dir}/FRIG2_UNB1/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
        'FRIG2_CTU1': f'{in_dir}/FRIG2_CTU1/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (browse)
        'FRIG2_MAWI1': f'{in_dir}/FRIG2_MAWI1/Xy-normal-abnormal.dat',
        # # MAWI1+Fridge: (normal: idle) abnormal: (browse)
        # # #
        'UNB1_SCAM1': f'{in_dir}/UNB1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        'CTU1_SCAM1': f'{in_dir}/CTU1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        'MAWI1_SCAM1': f'{in_dir}/MAWI1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        'FRIG1_SCAM1': f'{in_dir}/FRIG1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        'FRIG2_SCAM1': f'{in_dir}/FRIG2_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        #
        'MACCDC1_UNB1': f'{in_dir}/MACCDC1_UNB1/Xy-normal-abnormal.dat',
        'MACCDC1_CTU1': f'{in_dir}/MACCDC1_CTU1/Xy-normal-abnormal.dat',
        'MACCDC1_MAWI1': f'{in_dir}/MACCDC1_MAWI1/Xy-normal-abnormal.dat',
        #
        # # less flows of wshr1
        'UNB1_DRYER1': f'{in_dir}/UNB1_DRYER1/Xy-normal-abnormal.dat',
        'DRYER1_UNB1': f'{in_dir}/DRYER1_UNB1/Xy-normal-abnormal.dat',
        #
        # # it works
        'UNB1_DWSHR1': f'{in_dir}/UNB1_DWSHR1/Xy-normal-abnormal.dat',
        'DWSHR1_UNB1': f'{in_dir}/DWSHR1_UNB1/Xy-normal-abnormal.dat',
        #
        'FRIG1_DWSHR1': f'{in_dir}/FRIG1_DWSHR1/Xy-normal-abnormal.dat',
        'FRIG2_DWSHR1': f'{in_dir}/FRIG2_DWSHR1/Xy-normal-abnormal.dat',
        'CTU1_DWSHR1': f'{in_dir}/CTU1_DWSHR1/Xy-normal-abnormal.dat',
        'MAWI1_DWSHR1': f'{in_dir}/MAWI1_DWSHR1/Xy-normal-abnormal.dat',
        'MACCDC1_DWSHR1': f'{in_dir}/MACCDC1_DWSHR1/Xy-normal-abnormal.dat',
        #
        # # less flows of wshr1
        'UNB1_WSHR1': f'{in_dir}/UNB1_WSHR1/Xy-normal-abnormal.dat',
        'WSHR1_UNB1': f'{in_dir}/WSHR1_UNB1/Xy-normal-abnormal.dat',

    }
    print(f'len(data_path_mappings): {len(data_path_mappings)}')
    random_state = RANDOM_STATE
    overwrite = False
    verbose = VERBOSE

    ratios = [0.5, 0.8, 0.9, 0.95, 1.0] # [0.5, 0.8, 0.9, 0.95, 1.0]
    experiment_cases = generate_experiment_cases(n_repeats=5, fixed_kjl=False,
                                                 q_abnormal_thres=1, std=True, batch_size=100,
                                                 verbose=verbose, random_state=random_state)
    n_kjl = 100
    d_kjl = 10

    def single_case(data_name, data_file, percent_first_init, experiment_case):
        try:
            info = OrderedDict()

            sc = SINGLE_CASE(data_file=data_file, percent_first_init=percent_first_init, random_state=RANDOM_STATE,
                             overwrite=overwrite, verbose=VERBOSE)
            params = PARAM(data_file=data_file, data_name=data_name, percent_first_init=percent_first_init,
                           random_state=RANDOM_STATE, n_init_train=n_init_train, n_kjl=n_kjl, d_kjl=d_kjl,
                           overwrite=overwrite, verbose=VERBOSE)
            params.add_param(**experiment_case)
            params_str = str(params).replace('-', '\n\t')
            sc.run(params)

            out_file = f'out/{data_file}-case0-ratio_{percent_first_init}.dat'
            # dump_data(sc.info, out_file)  # pickle cannot be used in mulitprocessing, try dill (still not work)
            sc.display(sc.info, out_file + '.pdf', key=(data_name, data_file, percent_first_init, params_str),
                       is_show=True)  # sc.info = {'online_GMM': , 'batch_GMM': ''}
            print('+++online:')
            _aucs = np.asarray([np.asarray(v.info['aucs']) for i, v in sc.info['online'].items()])
            print(_aucs)
            # print([(f'repeat_{i}', v) for i, v in enumerate(_aucs)])
            print(f'mean: {np.mean(_aucs, axis=0)}\nstd:{np.std(_aucs, axis=0)}')
            print('+++batch:')
            _aucs = np.asarray([np.asarray(v.info['aucs']) for i, v in sc.info['batch'].items()])
            # print([(f'repeat_{i}', v) for i, v in enumerate(_aucs)])
            print(_aucs)
            print(f'mean: {np.mean(_aucs, axis=0)}\nstd:{np.std(_aucs, axis=0)}')

            info = {(data_name, data_file): {percent_first_init: {params_str: {}}}}
            info[(data_name, data_file)][percent_first_init][params_str] = copy.copy(
                sc.info)  # contains 'online' and 'batch'
        except:
            traceback.print_exc()

        return info

    # in parallel
    # get the number of cores. Note that, one cpu might have more than one core.
    n_jobs = int(joblib.cpu_count() // 4)
    print(f'n_job: {n_jobs}')
    parallel = Parallel(n_jobs=n_jobs, verbose=30)
    with parallel:
        outs = parallel(delayed(single_case)(data_name, data_file, percent_first_init, experiment_case)
                        for ((data_name, data_file), percent_first_init, experiment_case) in \
                        itertools.product(data_path_mappings.items(), ratios, experiment_cases))

    out_file = f'out/{in_dir}/init={n_init_train}-d_kjl={d_kjl}-n_kjl={n_kjl}.dat'
    # print(out_file)
    # dump_data(outs, out_file=out_file)
    out_file += '.xlsx'
    print(out_file)
    imgs2xlsx(data_path_mappings, out_file)
    mprint("\nFinish!")


if __name__ == '__main__':
    main()
