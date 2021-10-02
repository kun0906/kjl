"""Main entrance of batch GMM and online GMM experiments

Run the below command under "applications/"
    PYTHONPATH=../:./ python3.7 online/main_gmm_meanshift.py > online/out/online/src_dst/batch_online_gmm.txt 2>&1 &
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, pairwise_distances

from kjl.models.gmm import GMM
from kjl.models.kjl import KJL
from kjl.models.nystrom import NYSTROM
from kjl.models.online_gmm import ONLINE_GMM
from kjl.models.seek_mode import meanshift_seek_modes
from kjl.preprocessing.standardization import STD
from kjl.utils.data import load_data, batch, data_info
from kjl.utils.parameters import PARAM
from kjl.utils.tool import execute_time, time_func, mprint, dump_data
from online.config import *
from online.generate_data import generate_data, split_train_arrival_test, plot_data, split_left_test
from online.report import imgs2xlsx, individual_img


def sub_plot(batch_train_times, online_train_times, xlabel='', ylabel='auc', title='', ylim=[],
              out_file='res.png', legend_position='upper right'):

    n_cols = 2
    fig, ax = plt.subplots(nrows=3, ncols=n_cols)

    print(f'++batch: {batch_train_times}, ylabel: {ylabel}, title: {title}')
    print(f'++online: {online_train_times}, ylabel: {ylabel}, title: {title}')
    r = 0
    c = 0
    for i in range(len(batch_train_times)):
        if i % n_cols == 0 and i > 0:
            r += 1
            c = 0

        y = batch_train_times[i, :].reshape(1, -1)
        x = range(y.shape[1])
        yerr = np.std(y, axis=0)
        y = np.mean(y, axis=0)
        ax[r, c].errorbar(x, y, yerr, ecolor='r', capsize=2, linestyle='-', marker='.', color='green',
                    markeredgecolor='green', markerfacecolor='green', label=f'Batch', alpha=0.9)  # marker='*',

        y = online_train_times[i, :].reshape(1, -1)

        yerr = np.std(y, axis=0)
        y = np.mean(y, axis=0)
        ax[r, c].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                    markeredgecolor='blue', markerfacecolor='blue', label=f'Online', alpha=0.9)  # marker='*',

        if i % n_cols == 0:
            # plt.xlim([0.0, 1.0])
            if len(ylim) == 2:
                ax[r, c].set_ylim(ylim)  # [0.0, 1.05]
            # ax.set_xlabel(xlabel)
            ax[r, c].set_ylabel(ylabel)
            # plt.xticks(x)
            # plt.yticks(y)
            ax[r, c].legend(loc=legend_position)
        if 'component' in ylabel:
            ax[r, c].set_ylim([0, 50])  # [0.0, 1.05]
        c += 1
    # ax.set_title(title)
    # ax.figure.savefig(out_file, format='png', dpi=300)
    fig.suptitle(title)
    fig.savefig(out_file, format='png', dpi=300)

class SINGLE_CASE:

    def __init__(self, random_state=42, **kwargs):
        self.overwrite = None
        self.random_state = random_state

        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_data(self, data_file, params):
        if self.overwrite:
            if pth.exists(data_file): os.remove(data_file)
        if not pth.exists(data_file):
            data_file = generate_data(params.data_name, params.data_type, out_file=data_file,
                                      direction= params.direction,
                                      overwrite=params.overwrite, random_state=params.random_state)
        X, y = load_data(data_file)
        return X, y

    def visual_tsne(self, X, y, params=None):

        n_components = 2
        if not (params is None):
            data_name = params.data_name
            init_set = f'(init_set={int(params.percent_first_init * 100)}:{int(round((1 - params.percent_first_init) * 100))}-{params.X_init_train.shape})'
        else:
            data_name = 'Data'
            init_set = ''

        # PCA
        X_embedded = PCA(n_components=n_components, random_state=100).fit_transform(X)
        plot_data(X_embedded, y, title=f'pca on {data_name}, {init_set}')

        # TSNE
        X_embedded = TSNE(n_components=n_components, random_state=100).fit_transform(X)
        plot_data(X_embedded, y, title=f'tsne on {data_name}, {init_set}')

    # def run(self, params):
    #     info = {}
    #
    #     params.data_type = 'two_datasets'
    #     X, y = self.load_data(params.data_file, params)
    #
    #     # params.X_init_train, params.y_init_train, params.X_init_test, params.y_init_test, \
    #     # params.X_arrival, params.y_arrival, params.X_test, params.y_test = split_train_arrival_test(X, y, params)
    #     # X_tmp = np.concatenate([params.X_init_train, params.X_init_test, params.X_arrival, params.X_test],
    #     #                        axis=0)
    #     # y_tmp = np.concatenate([params.y_init_train, params.y_init_test, params.y_arrival, params.y_test],
    #     #                        axis=0)
    #     # self.visual_tsne(X_tmp, y_tmp, params)
    #     n_repeats = params.n_repeats
    #     for is_online in [True, False]:
    #         # params_copy.is_online = is_online
    #         _info = {}
    #         for i in range(n_repeats):
    #             params_copy = copy.deepcopy(params)
    #             params_copy.is_online = is_online
    #             if params_copy.is_online:
    #                 models = ONLINE_GMM_MAIN(params_copy)
    #                 # models = BATCH_GMM_MAIN(params_copy)
    #             else:
    #                 models = BATCH_GMM_MAIN(params_copy)
    #
    #             params_copy.random_state = (i + 1) * self.random_state
    #             # params_copy.random_state = (i + 1) * 100
    #             print(f'i: {i}, ---random_state: {params_copy.random_state}')
    #             models.random_state = params_copy.random_state
    #             models.params.random_state = params_copy.random_state
    #
    #             params_copy.X_init_train, params_copy.y_init_train, params_copy.X_init_test, params_copy.y_init_test, \
    #             params_copy.X_arrival, params_copy.y_arrival, params_copy.X_test, params_copy.y_test = split_train_arrival_test(
    #                 X, y, params_copy)
    #             if i == 0 and is_online:
    #                 X_tmp = np.concatenate(
    #                     [params_copy.X_init_train, params_copy.X_init_test, params_copy.X_arrival, params_copy.X_test],
    #                     axis=0)
    #                 y_tmp = np.concatenate(
    #                     [params_copy.y_init_train, params_copy.y_init_test, params_copy.y_arrival, params_copy.y_test],
    #                     axis=0)
    #                 # self.visual_tsne(X_tmp, y_tmp, params_copy)
    #                 data_info(X_tmp, name=f'X_tmp: {i}, random_state: {params_copy.random_state}')
    #
    #             models.init_train_test(params_copy.X_init_train, params_copy.y_init_train, params_copy.X_test,
    #                                   params_copy.y_test)
    #             _info[i] = models.batch_train_test(params_copy.X_arrival, params_copy.y_arrival, params_copy.X_test,
    #                                               params_copy.y_test)
    #         info['online' if is_online else 'batch'] = _info
    #     self.info = info

    def run(self, params):
        info = {}

        params.data_type = 'two_datasets'
        X, y = self.load_data(params.data_file, params)

        X_left, y_left, X_test, y_test = split_left_test(X, y, params)

        # params.X_init_train, params.y_init_train, params.X_init_test, params.y_init_test, \
        # params.X_arrival, params.y_arrival, params.X_test, params.y_test = split_train_arrival_test(X, y, params)
        # X_tmp = np.concatenate([params.X_init_train, params.X_init_test, params.X_arrival, params.X_test],
        #                        axis=0)
        # y_tmp = np.concatenate([params.y_init_train, params.y_init_test, params.y_arrival, params.y_test],
        #                        axis=0)
        # self.visual_tsne(X_tmp, y_tmp, params)
        n_repeats = params.n_repeats
        for is_online in [True, False]:  # means if we use online_gmm or batch_gmm
            # params_copy.is_online = is_online
            _info = {}
            for i in range(n_repeats):
                params_copy = copy.deepcopy(params)
                params_copy.is_online = is_online
                if params_copy.is_online:
                    model = ONLINE_GMM_MAIN(params_copy)
                    # models = BATCH_GMM_MAIN(params_copy)
                else:
                    model = BATCH_GMM_MAIN(params_copy)

                params_copy.random_state = (i + 1) * self.random_state
                # params_copy.random_state = (i + 1) * 100
                print(f'i: {i}, ---random_state: {params_copy.random_state}')
                model.random_state = params_copy.random_state
                model.params.random_state = params_copy.random_state

                params_copy.X_init_train, params_copy.y_init_train, params_copy.X_init_test, params_copy.y_init_test, \
                params_copy.X_arrival, params_copy.y_arrival, params_copy.X_test, params_copy.y_test = split_train_arrival_test(
                    X_left, y_left, params_copy)
                if not params_copy.gs:  # when we don't use grid search: X_init_test = X_test
                    params_copy.X_init_test, params_copy.y_init_test = X_test, y_test
                # params_copy.X_init_test, params_copy.y_init_test = X_test, y_test
                # use the unique test set
                params_copy.X_test, params_copy.y_test = X_test, y_test
                if i == 0:
                    X_tmp = np.concatenate(
                        [params_copy.X_init_train, params_copy.X_init_test, params_copy.X_arrival, params_copy.X_test],
                        axis=0)
                    y_tmp = np.concatenate(
                        [params_copy.y_init_train, params_copy.y_init_test, params_copy.y_arrival, params_copy.y_test],
                        axis=0)
                    # self.visual_tsne(X_tmp, y_tmp, params_copy)
                    data_info(X_tmp, name=f'X_tmp: {i}, random_state: {params_copy.random_state}')

                # models.init_train_test(params_copy.X_init_train, params_copy.y_init_train, params_copy.X_test,
                #                       params_copy.y_test)
                print(f'---gs: {params_copy.gs}, ratio: {params_copy.percent_first_init}\n'
                      f'X_init_train: {params_copy.X_init_train.shape}, y_init_train: {Counter(params_copy.y_init_train)},\n'
                      f'X_init_test: {params_copy.X_init_test.shape}, y_init_test: {Counter(params_copy.y_init_test)}\n'
                      f'X_arrival: {params_copy.X_arrival.shape}, y_arrival: {Counter(params_copy.y_arrival)},\n'
                      f'X_test: {params_copy.X_test.shape}, y_test: {Counter(params_copy.y_test)}')
                model.init_train_test(params_copy.X_init_train, params_copy.y_init_train, params_copy.X_init_test,
                                      params_copy.y_init_test, params_copy.X_test, params_copy.y_test)
                model.batch_train_test(params_copy.X_arrival, params_copy.y_arrival, params_copy.X_test,
                                                  params_copy.y_test)
                _info[i] = model.info
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
                    v = [_v['train_time'] for _v in res[name]]
                elif name == 'test_times':
                    v = [_v['test_time'] for _v in res[name]]
                elif name == 'n_components':
                    v = [_v[name] for _v in res['model_params']]
                else:
                    v = res[name]
                vs.append(np.asarray(v))

            return np.asarray(vs)

        def _plot(ax, online_train_times, batch_train_times, xlabel, ylabel, title, out_file, ylim=[],
                  legend_position='upper right'):

            sub_plot(online_train_times, batch_train_times, xlabel=xlabel, ylabel=ylabel, title=title,
                     out_file=out_file + '-ind.png')

            y = batch_train_times
            print(f'++batch: {batch_train_times}, ylabel: {ylabel}, title: {title}')
            x = range(y.shape[1])
            yerr = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            ax.errorbar(x, y, yerr, ecolor='r', capsize=2, linestyle='-', marker='.', color='green',
                        markeredgecolor='green', markerfacecolor='green', label=f'Batch', alpha=0.9)  # marker='*',

            y = online_train_times
            print(f'++online: {online_train_times}, ylabel: {ylabel}, title: {title}')
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
            # ax.figure.savefig(out_file, format='png', dpi=300)

        params = online_info[0]['params']
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
        n_components_init = int(
            np.mean([online_info[i]['model_params'][0]['n_components'] for i in range(params.n_repeats)]))
        gs = params.gs
        covariance_type = params.covariance_type
        meanshift = params.meanshift
        q_kjl = params.q_kjl
        n_kjl = params.n_kjl
        d_kjl = params.d_kjl
        c_kjl = params.centering_kjl
        std = params.std
        random_state = self.random_state
        with_means_std = params.with_means_std
        kjl = params.kjl
        n_repeats = params.n_repeats
        # dataset_name, data_file = k_dataset
        dataset_name = f'{dataset_name} (init_set={int(params.percent_first_init * 100)}:{int(round((1 - params.percent_first_init) * 100))}-{params.X_init_train.shape})'
        if gs:
            init_val_set = f'{params.X_init_test.shape[0]}'
        else:
            init_val_set = f'=X_test'
        arrival_set = f'{params.X_arrival.shape[0]}'
        test_set = f'{params.X_test.shape[0]}'
        if kjl:
            # {covariance_type};
            title = f'n_cp={n_components_init}, {covariance_type}; gs={gs}; std={std}_c={with_means_std}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}, c={c_kjl}; ms={meanshift}'
            if fixed_kjl:
                title = f'Fixed KJL on {dataset_name};\ninit_val(test)={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n{title}'
            elif fixed_U_size:
                # (replace {n_point} cols and rows of U)
                title = f'Fixed U size on {dataset_name};\ninit_val(test)={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n{title}'
            else:  # increased_U
                title = f'Increased U size on {dataset_name};\ninit_val(test)={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n{title}'
        else:
            #  {covariance_type};
            title = f'n_cp={n_components_init}, {covariance_type}; gs={gs}; std={std}_ctr={with_means_std}; KJL={kjl},  c={c_kjl}; ms={meanshift}'
            title = f'{dataset_name};\ninit_val={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n({title})'

        title = title.replace('False', 'F')
        title = title.replace('True', 'T')
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
        print(f'online_aucs: {online_aucs}')
        print(f'batch_aucs: {batch_aucs}')
        print(f'online_aucs-batch_aucs: {online_aucs - batch_aucs}')
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
            plt.subplots_adjust(top=0.85, bottom=0.1, right=0.975, left=0.12)
        except Warning as e:
            raise ValueError(e)

        # fig.text(.5, 15, "total label", ha='center')
        plt.figtext(0.5, 0.01, f'X-axis:({xlabel})', fontsize=11, va="bottom", ha="center")
        print(out_file)
        if not pth.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
        if pth.exists(out_file): os.remove(out_file)
        fig.savefig(out_file, format='pdf', dpi=300)
        out_file += '.png'
        if pth.exists(out_file): os.remove(out_file)
        fig.savefig(out_file, format='png', dpi=300)
        if is_show: plt.show()
        plt.close(fig)

        return out_file


class BASE_MODEL:

    def __init__(self):
        self.verbose = 1

    def init_train_test(self, X_init_train, y_init_train, X_init_test, y_init_test, X_test, y_test):

        ##########################################################################################
        # Step 1. configure parameters for tuning
        # case = 'GMM_full-gs_True-kjl_True-nystrom_False-quickshift_False-meanshift_False'
        if self.params.gs:
            if self.params.detector_name == 'GMM' and self.params.covariance_type == 'diag':  # self.params['kjl']  and not self.params.quickshift  and not self.params.meanshift
                params = {}
                # # # # # # grid search
                params['n_components'] = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
                params['qs_kjl'] = [0.3]
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

                # Fit a models on the init train set with self.params
                self._init_train(X_init_train, y_init_train)
                # Evaluate the models on the test set and get AUC
                self._init_test(X_init_test, y_init_test)
                mprint(f'AUC: {self.auc} with {self.model.get_params()}, and self.q_kjl: {q_kjl}', self.verbose, DEBUG)
                # find the best one
                if best_auc < self.auc:  # here should be <, not <=
                    best_auc = self.auc
                    best_model_params = copy.deepcopy(self.model.get_params())  # only GMM parameters
                    best_params = copy.deepcopy(self.params)  # inculdes all params, but part of models paramters.

            ##########################################################################################
            # Step 3. To get the best models with best_params, best_auc, best train_time and test_time
            mprint(
                f'\n***The best params: model_params: {best_model_params}, q_kjl: {best_params.q_kjl}, d_kjl: {best_params.d_kjl}, n_kjl: {best_params.n_kjl}')
            self.params = best_params
            self.model = GMM()
            model_params = {'n_components': best_model_params['n_components'],
                            'covariance_type': best_model_params['covariance_type'],
                            'means_init': None, 'random_state': best_model_params['random_state']}
            self.model.set_params(**model_params)

            # Fit self.models on the init set
            self._init_train(X_init_train, y_init_train)
            # Evaluate the models on the test set
            # self._init_test(X_init_test, y_init_test)
            self._init_test(X_test, y_test)
            self.abnormal_thres = np.infty
            mprint(f'***init result: train_time: {self.train_time}, abnormal_thres: {self.abnormal_thres}, '
                   f'test_time: {self.test_time}, AUC: {self.auc},', self.verbose, DEBUG)
        else:  # default params: use meanshift
            # best params for the combination of UNB1 and UNB2
            # params['n_components'] = [5]  # UNB2
            # params['qs_kjl'] = [0.7]
            #
            # params['n_components'] = [2]  # mimic_GMM_dataset
            # params['qs_kjl'] = [0.3]

            # self.params.meanshift = True  # ms = False if gs else True
            if not self.params.meanshift: self.params.n_components = 10
            self.params.q_kjl = 0.3
            self.params.q_ms = 0.3

            self.model = GMM()
            model_params = {
                'covariance_type': self.params.covariance_type,
                'means_init': None, 'random_state': self.params.random_state}
            self.model.set_params(**model_params)

            # Fit self.models on the init set
            self._init_train(X_init_train, y_init_train, is_batch_train=True)  # use meanshift to get the n_components
            # Evaluate the models on the test set
            self._init_test(X_init_test, y_init_test)
            # self.params.meanshift = False    # only use meanshift on the init set
            self.abnormal_thres = np.infty
            mprint(
                f'\n***The best params: model_params: {self.model.get_params()}, q_kjl: {self.params.q_kjl}, d_kjl: {self.params.d_kjl}, n_kjl: {self.params.n_kjl}')
            mprint(f'***init result: train_time: {self.train_time}, abnormal_thres: {self.abnormal_thres}, '
                   f'test_time: {self.test_time}, AUC: {self.auc},', self.verbose, DEBUG)

        ##########################################################################################
        # Step 4. Store all important results
        self.init_info = {'train_time': self.train_time, 'abnormal_thres': self.abnormal_thres,
                          'test_time': self.test_time, 'auc_time': self.auc,
                          'X_train_shape': X_init_train.shape, 'X_test_shape': X_init_test.shape,
                          'params': self.params, 'model_params': self.model.get_params()}

        # self.params.meanshift = False

        return self.model

    def _init_train(self, X_init_train, y_init_train=None, is_batch_train=False):
        """Train models on the initial set (init_set)

        Parameters
        ----------
        models: models instance

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

        # self.X_init_train = X_init_train
        # self.y_init_train = y_init_train

        ##########################################################################################
        # Step 1: Preprocessing the data, which includes standardization, mode seeking, and kernel projection.
        if self.params.std:
            # Step 1.1: Standardize the data first
            # fit std_inst
            self.std_inst = STD(with_means=self.params.with_means_std)
            _, std_fitting_time = time_func(self.std_inst.fit, X_init_train)
            # transform X_train
            X_init_train, std_time = time_func(self.std_inst.transform, X_init_train)
            # mprint(f'mu: {self.std_inst.scaler.mean_},std_var: {self.std_inst.scaler.scale_}', self.verbose, DEBUG)
            std_time += std_fitting_time
        else:
            self.std_inst = None
            std_time = 0.0
        train_time += std_time

        self.X_init_train = X_init_train
        self.y_init_train = y_init_train

        seeking_time = 0.0
        if self.params.before_kjl_meanshift:
            # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
            start = datetime.now()
            if is_batch_train:
                self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
                if self.params.meanshift:
                    dists = pairwise_distances(X_init_train)
                    if self.params.kjl: self.params.q_ms = self.params.q_kjl
                    if not hasattr(self.params, 'q_ms'): self.params.q_ms = 0.3

                    if 0 < self.params.q_ms < 1:
                        sigma = np.quantile(dists, self.params.q_ms)  # also used for kjl
                    else:
                        sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl

                    means_init, n_components, seeking_time, all_n_clusters = meanshift_seek_modes(
                        X_init_train, bandwidth=sigma, thres_n=self.thres_n)
                    self.params.n_components = n_components
                # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
                #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                #         thres_n=self.thres_n)
                #     self.params['n_components'] = self.n_components
                else:
                    seeking_time = 0
            else:
                seeking_time = 0.0
            end = datetime.now()
            seeking_time = (end - start).total_seconds()
            train_time += seeking_time

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
        if self.verbose >= DEBUG: data_info(X_init_train, name='X_proj')

        seeking_time = 0.0
        if self.params.after_kjl_meanshift:
            # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
            start = datetime.now()
            if is_batch_train:
                self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
                if self.params.meanshift:
                    dists = pairwise_distances(X_init_train)
                    if self.params.kjl: self.params.q_ms = self.params.q_kjl
                    if not hasattr(self.params, 'q_ms'): self.params.q_ms = 0.3
                    if 0 < self.params.q_ms < 1:
                        sigma = np.quantile(dists, self.params.q_ms)  # also used for kjl
                    else:
                        sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl

                    means_init, n_components, seeking_time, all_n_clusters = meanshift_seek_modes(
                        X_init_train, bandwidth=sigma, thres_n=self.thres_n)
                    self.params.n_components = n_components
                # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
                #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                #         thres_n=self.thres_n)
                #     self.params['n_components'] = self.n_components
                else:
                    seeking_time = 0
            else:
                seeking_time = 0.0
            end = datetime.now()
            seeking_time = (end - start).total_seconds()
            train_time += seeking_time

        ##########################################################################################
        # Step 2. Setting self.models'params (e.g., means_init, and n_components when use meanshift or quickshift)
        model_params = {'n_components': self.params.n_components,
                        'covariance_type': self.params.covariance_type,
                        'means_init': None, 'random_state': self.random_state}
        self.model.set_params(**model_params)
        # mprint(self.models.get_params(), self.verbose, DEBUG)
        # Fit self.models on the X_train
        _, model_fitting_time = time_func(self.model.fit, X_init_train)
        train_time += model_fitting_time

        # ##########################################################################################
        # # # Step 3. Get the threshold used to decide if a new flow is normal
        # # # the following values will be used in the online update phase
        # y_score, _ = time_func(self.models.decision_function, X_init_train)
        # self.abnormal_thres = np.quantile(y_score, q=self.params.q_abnormal_thres)  # abnormal threshold
        # _, log_resp = self.models._e_step(X_init_train)
        # self.models.sum_resp = np.sum(np.exp(log_resp), axis=0)
        # self.models.y_score = y_score
        # # self.X_init_train_proj = X_init_train

        self.train_time = {'train_time': train_time,
                           'preprocessing_time': std_time + proj_time,
                           'seeking_time': seeking_time,
                           'model_fitting_time': model_fitting_time,
                           'rescore_time': 0}

        return self

    def _init_test(self, X_init_test, y_init_test):
        """Evaluate the models on the set set

        Parameters
        ----------
        models:
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

        test_time = 0.0
        mprint(f'y_test: {Counter(y_init_test)}', self.verbose, INFO)
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

    def batch_train_test(self, X_arrival, y_arrival, X_test, y_test):
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
        # # # Step 1. Get best initial models (init_model) on initial set (init_set) and evaluate it on init test set
        # self.init_model = self.init_train_test(X_init_train, y_init_train, X_init_test, y_init_test)
        self.init_model = self.model

        ##########################################################################################
        # Step 2. Online train and evaluate models
        self.info['train_times'] = [self.init_info['train_time']]
        self.info['abnormal_threses'] = [self.init_info['abnormal_thres']]
        self.info['model_params'] = [self.init_info['model_params']]
        self.info['test_times'] = [self.init_info['test_time']]
        self.info['aucs'] = [self.init_info['auc_time']]

        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=self.params.batch_size)):
            if i == 0:
                # copy means_, covariances, and other params from the init_model
                self.model = copy.deepcopy(self.init_model)
                # self.std_inst = copy.deepcopy(self.std_inst)
                # self.kjl_inst = copy.deepcopy(self.kjl_inst)
                self.X_acculumated_train = self.X_init_train  # (std data)
                self.y_acculumated_train = self.y_init_train

            # online train models (update GMM models values, such as, means, covariances, kjl_U, and n_components)
            self.online_train(X_batch, Y_batch)  # update self.models
            self.info['train_times'].append(self.train_time)
            self.info['model_params'].append(self.model.get_params())
            self.info['abnormal_threses'].append(self.abnormal_thres)

            # online test models
            self.online_test(X_test, y_test)
            self.info['test_times'].append(self.test_time)
            self.info['aucs'].append(self.auc)

            mprint(f'batch_{i + 1}: train_time: {self.train_time}, '
                   f'test_time: {self.test_time}, auc: {self.auc}', self.verbose, WARNING)

        self.info['params'] = self.params

        return self

    def online_train(self, X_batch, y_batch=None):
        """Online train the models: using the X_batch to retrain and update the current models incrementally.

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
        # if self.verbose >= DEBUG: data_info(np.asarray(X_batch), name='X_batch')
        if self.verbose >= DEBUG: data_info(np.asarray(self.X_acculumated_train), name='X_acculumated_train')
        mprint(f'y_batch: {Counter(y_batch)}', self.verbose, DEBUG)
        train_time = 0.0
        X_batch_raw = X_batch
        y_batch_raw = y_batch

        ##########################################################################################
        start = datetime.now()
        # Step 1: Predict on the arrival data  first.
        # Use the fitted models to predict X_batch first, and according to the result,
        # only the normal data will be used to train a new models, which will be used to replace the current models.
        # Step 1.1: Preprocessing: std and projection
        _X_batch, preprocessing_time = time_func(self._preprocessing, X_batch)
        # train_time += preprocessing_time
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
                # _, time_std_update = time_func(self.std_inst.update, X_batch_raw)
                X_batch, std_time = time_func(self.std_inst.transform, X_batch_raw)
                # std_time += time_std_update
            else:
                self.std_inst = None
                X_batch = X_batch_raw
                std_time = 0.0

            self.abnormal_thres = np.infty  # for debug
            normal_idx = np.where((y_score <= self.abnormal_thres) == True)
            abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
            X_batch_normal = X_batch[normal_idx] if len(normal_idx[0]) > 0 else []
            y_batch_normal = y_batch[normal_idx] if len(normal_idx[0]) > 0 else []
            X_batch_abnormal = X_batch[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

            # # without std and kjl
            # X_batch_normal_raw = X_batch_raw[normal_idx] if len(normal_idx[0]) > 0 else []
            # y_batch_normal_raw = y_batch_raw[normal_idx] if len(normal_idx[0]) > 0 else []
            # X_batch_abnormal_raw = X_batch_raw[abnormal_idx] if len(abnormal_idx[0]) > 0 else []

            # reproject the previous X_acculumated_train_proj with the updated std and kjl
            self.X_acculumated_train = np.concatenate([self.X_acculumated_train, X_batch_normal], axis=0)  # std_data
            self.y_acculumated_train = np.concatenate([self.y_acculumated_train, y_batch_normal])  # std_data

            seeking_time = 0.0
            if self.params.before_kjl_meanshift:
                # before kjl, using meanshift
                start = datetime.now()
                # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
                seeking_time = 0
                self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
                if self.params.meanshift:
                    dists = pairwise_distances(self.X_acculumated_train)
                    if self.params.kjl: self.params.q_ms = self.params.q_kjl
                    if not hasattr(self.params, 'q_ms'): self.params.q_ms = 0.3
                    if 0 < self.params.q_ms < 1:
                        sigma = np.quantile(dists, self.params.q_ms)  # also used for kjl
                    else:
                        sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl
                    means_init, n_components, seeking_time, all_n_clusters = meanshift_seek_modes(
                        self.X_acculumated_train, bandwidth=sigma, thres_n=self.thres_n)
                    self.params.n_components = n_components
                    self.model.n_components = n_components
                    self.model.means_ = means_init
                # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
                #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                #         thres_n=self.thres_n)
                #     self.params['n_components'] = self.n_components
                else:
                    seeking_time = 0
                end = datetime.now()
                seeking_time = (end - start).total_seconds()
                train_time += seeking_time

            # Step 2.2: update projection: kjl or nystrom
            if self.params.kjl:
                # if self.verbose > 100:  data_info(self.X_acculumated_train, name='self.X_acculumated_train')  # if will take train time
                # # compute sigma
                # dists = pairwise_distances(self.X_acculumated_train)
                # self.kjl_inst.sigma_kjl = np.quantile(dists, q=self.params.q_kjl)
                # # print(f'self.sigma_kjl: {self.kjl_inst.sigma_kjl}')
                self.kjl_inst.n_samples = self.X_acculumated_train.shape[0] - len(X_batch_normal)

                #  Update kjl: self.U_kjl, self.Xrow_kjl.
                # self.kjl_inst.n_samples = self.X_acculumated_train.shape[0]
                _, time_proj_update = time_func(self.kjl_inst.update, X_batch, y_batch, None, None,
                                                None)
                # # X_batch, proj_time = time_func(self.kjl_inst.transform, X_batch)
                # proj_time += time_proj_update
                proj_time = time_proj_update
            elif self.params.nystrom:
                # Update nystrom_inst
                _, time_proj_update = time_func(self.nystrom_inst.update, X_batch)
                # X_batch, proj_time = time_func(self.nystrom_inst.transform, X_batch)
                # proj_time += time_proj_update
            else:
                proj_time = 0

        if not self.params.kjl:
            self.params.incorporated_points = 0
            self.params.fixed_U_size = False
        else:
            self.params.incorporated_points = self.kjl_inst.t
            self.params.fixed_U_size = self.kjl_inst.fixed_U_size

        # self.abnormal_thres = np.infty  # for debug

        # normal_idx = np.where((y_score <= self.abnormal_thres) == True)
        # abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
        # X_batch_normal = X_batch[normal_idx] if len(normal_idx[0]) > 0 else []
        # y_batch_normal = y_batch[normal_idx] if len(normal_idx[0]) > 0 else []
        # X_batch_abnormal = X_batch[abnormal_idx] if len(abnormal_idx[0]) > 0 else []
        #
        # # # without std and kjl
        # # X_batch_normal_raw = X_batch_raw[normal_idx] if len(normal_idx[0]) > 0 else []
        # # y_batch_normal_raw = y_batch_raw[normal_idx] if len(normal_idx[0]) > 0 else []
        # # X_batch_abnormal_raw = X_batch_raw[abnormal_idx] if len(abnormal_idx[0]) > 0 else []
        #
        # # reproject the previous X_acculumated_train_proj with the updated std and kjl
        # self.X_acculumated_train = np.concatenate([self.X_acculumated_train, X_batch_normal], axis=0)  # std_data
        # self.y_acculumated_train = np.concatenate([self.y_acculumated_train, y_batch_normal])  # std_data
        if self.params.kjl:
            X_acculumated_train_proj, _ = time_func(self.kjl_inst.transform, self.X_acculumated_train)
            # X_acculumated_train_proj = self._preprocessing(self.X_acculumated_train)
        else:
            X_acculumated_train_proj = self.X_acculumated_train

        end = datetime.now()
        preprocessing_time = (end - start).total_seconds()
        train_time += preprocessing_time

        start = datetime.now()
        # # # Step 2.3: Seek modes of the data by quickshift++ or meanshift for initialize GMM
        # n_thres = 0  # used to filter clusters which have less than 100 datapoints
        # if self.params.meanshift:
        #     if self.params.kjl:  # use the same sigma of kjl
        #         self.sigma = self.kjl_inst.sigma_kjl
        #     else:
        #         dists = pairwise_distances(X_acculumated_train_proj)
        #         self.sigma = np.quantile(dists, q=0.3)
        #     ms = MODESEEKING(method_name='meanshift', bandwidth=None,
        #                      random_state=self.random_state, verbose=self.verbose)
        #     _, ms_fitting_time = time_func(ms.fit, X_acculumated_train_proj, n_thres=n_thres)
        #     n_components = ms.n_clusters_
        #     mprint(f'mode_seeking_time: {ms_fitting_time}s, n_clusters: {n_components}', self.verbose, DEBUG)
        #     n_samples, _ = X_acculumated_train_proj.shape
        #     resp = np.zeros((n_samples, n_components))
        #     # label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
        #     #                        random_state=random_state).fit(X).labels_
        #     resp[np.arange(n_samples), ms.labels_] = 1
        #
        #     weights, means, covariances = _estimate_gaussian_parameters(
        #         X_acculumated_train_proj, resp, self.models.reg_covar, self.models.covariance_type)
        #     weights /= n_samples
        #     new_model = ONLINE_GMM(n_components=n_components,
        #                            covariance_type=self.models.covariance_type,
        #                            weights_init=weights,
        #                            means_init=means,
        #                            covariances_init=covariances,
        #                            verbose=self.verbose,
        #                            random_state=self.params.random_state,
        #                            warm_start=True)
        #
        # elif self.params.quickshift:
        #     ms = MODESEEKING(method_name='quickshift', k=100, beta=0.9,
        #                      random_state=self.random_state, verbose=self.verbose)
        #     _, ms_fitting_time = time_func(ms.fit, X_acculumated_train_proj, n_thres=n_thres)
        #     n_components = ms.n_clusters_
        #     mprint(f'mode_seeking_time: {ms_fitting_time}s, n_clusters: {n_components}', self.verbose, DEBUG)
        #     n_samples, _ = X_acculumated_train_proj.shape
        #     resp = np.zeros((n_samples, self.models.n_components))
        #     # label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
        #     #                        random_state=random_state).fit(X).labels_
        #     resp[np.arange(n_samples), ms.labels_] = 1
        #
        #     weights, means, covariances = _estimate_gaussian_parameters(
        #         X_acculumated_train_proj, resp, self.models.reg_covar, self.models.covariance_type)
        #     weights /= n_samples
        #     new_model = ONLINE_GMM(n_components=n_components,
        #                            covariance_type=self.models.covariance_type,
        #                            weights_init=weights,
        #                            means_init=means,
        #                            covariances_init=covariances,
        #                            verbose=self.verbose,
        #                            random_state=self.params.random_state,
        #                            warm_start=True)
        # else:
        #     # print(f'before online, self.models.weights_: {self.models.weights_}')
        #     # print(f'before online, self.models.means_: {self.models.means_}')
        #     # print(f'before online, self.models.covariances_: {self.models.covariances_}')
        #     new_model = ONLINE_GMM(n_components=self.models.n_components,
        #                            covariance_type=self.models.covariance_type,
        #                            weights_init=self.models.weights_,
        #                            means_init=self.models.means_,
        #                            covariances_init=self.models.covariances_,
        #                            verbose=self.verbose,
        #                            random_state=self.params.random_state,
        #                            warm_start=True)

        # Step 2.4: online train a new models on the batch data
        # 1) train on the normal data first (without needing to create any new component)
        # 2) train on the abnormal data (create new components)
        # set the default params of ONLINE_GMM
        # new_model.sum_resp = self.models.sum_resp  # shape (1, d): sum of exp(log_resp)
        # new_model._initialize()  # set up cholesky

        # update_flg = False
        # # The first update of GMM
        # # X_batch_used = np.zeros()
        # while (len(X_batch_normal) > 0) or (len(X_batch_abnormal) > 0):
        #     if len(X_batch_normal) > 0:
        #         # # 1) train on the normal data first (without needing to create any new component)
        #         # _, log_resp = new_model._e_step(X_batch_normal)
        #         # # data_info(np.exp(log_resp), name='resp')
        #         # new_model._m_step_online(X_batch_normal, log_resp, sum_resp_pre=new_model.sum_resp,
        #         #                          n_samples_pre=self.X_acculumated_train_proj.shape[0])
        #         self.X_acculumated_train = np.concatenate([self.X_acculumated_train, X_batch_normal_raw], axis=0)
        #         X_batch_normal_raw = []
        #         X_acculumated_train_proj = np.concatenate([X_acculumated_train_proj, X_batch_normal], axis=0)
        #         X_batch_normal = []
        #         self.y_acculumated_train = np.concatenate([self.y_acculumated_train, y_batch_normal_raw], axis=0)
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

        # SINGLE_CASE.visual_tsne(X_acculumated_train_proj, self.y_acculumated_train, self.params)
        seeking_time = 0.0
        if self.params.after_kjl_meanshift:
            # seek after kjl
            start = datetime.now()
            # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
            seeking_time = 0
            self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
            if self.params.meanshift:
                dists = pairwise_distances(X_acculumated_train_proj)
                if self.params.kjl: self.params.q_ms = self.params.q_kjl
                if not hasattr(self.params, 'q_ms'): self.params.q_ms = 0.3
                if 0 < self.params.q_ms < 1:
                    sigma = np.quantile(dists, self.params.q_ms)  # also used for kjl
                else:
                    sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl
                means_init, n_components, seeking_time, all_n_clusters = meanshift_seek_modes(
                    X_acculumated_train_proj, bandwidth=sigma, thres_n=self.thres_n)
                self.params.n_components = n_components
                self.model.n_components = n_components
                self.model.means_ = means_init
            # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
            #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
            #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
            #         thres_n=self.thres_n)
            #     self.params['n_components'] = self.n_components
            else:
                seeking_time = 0
            end = datetime.now()
            seeking_time = (end - start).total_seconds()

            train_time += seeking_time
            start = datetime.now()

        new_model = ONLINE_GMM(n_components=self.model.n_components,
                               covariance_type=self.model.covariance_type,
                               # means_init=self.models.means_,
                               verbose=self.verbose,
                               random_state=self.params.random_state,
                               warm_start=True)
        new_model._initialize_parameters(X_acculumated_train_proj, self.random_state,
                                         init='k-means++')  # k-means++ ,self.models.means_
        # Train the new models until it converges
        i = 0
        new_model.converged_ = False
        if not new_model.converged_:  new_model.max_iter = 100
        prev_lower_bound = -np.infty
        if self.verbose >= 1:
            # if self.verbose >= DEBUG: data_info(X_acculumated_train_proj, name='X_proj')
            mprint(
                f'X_acculumated_train_proj: {X_acculumated_train_proj.shape}, y_acculumated_train: {Counter(self.y_acculumated_train)}')
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
            # print(f'i: {i}, 1. / np.sqrt(self.models.covariances_): ', 1 / np.sqrt(new_model.covariances_), log_resp,
            #       new_model.means_, new_model.weights_)
            change = lower_bound - prev_lower_bound
            if abs(change) < new_model.tol:
                # if np.all(new_model.means_[0] == 0.0):
                #     new_model.converged_ = False
                #     # new_model.weights_ = self.models.weights_
                #     # new_model.means_ = self.models.means_
                #     # new_model.covariances_ = self.models.covariances_
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
        # print(f'1. / np.sqrt(self.models.covariances_): ', 1 / np.sqrt(new_model.covariances_), log_resp,
        #       new_model.means_, new_model.weights_)

        # self.models = GMM(n_components=self.models.n_components, covariance_type=self.params.covariance_type,
        #                  random_state=self.params.random_state,  weights_init=self.models.weights_,
        #                            means_init=self.models.means_, precisions_init= 1. / np.sqrt(self.models.covariances_),
        #                  )
        # print(f'1. / np.sqrt(self.models.covariances_): ', self.models.precisions_init)
        # _, model_fitting_time = time_func(self.models.fit, X_acculumated_train_proj)
        # log_prob_norm=0
        # first_train_time=0
        # iteration_time = model_fitting_time
        # self.models.sum_resp=0  # shape (1, d): sum of exp(log_resp)

        ##########################################################################################
        # Step 3:  update the abnormal threshold with all accumulated data
        self.model = new_model
        # # mprint(new_model.get_params(), self.verbose, DEBUG)
        rescore_time = 0.0
        # if not new_model.converged_:
        #     self.abnormal_thres, rescore_time = time_func(self.update_abnormal_thres,
        #                                                   self.models,
        #                                                   X_acculumated_train_proj)
        # else:
        #     # override the _e_step(), so  here is not mean(log_prob_norm)    #  return -1 * self.score_samples(X)
        #     y_score = - log_prob_norm
        #     self.abnormal_thres, rescore_time = time_func(np.quantile, y_score, q=self.params.q_abnormal_thres)

        train_time += rescore_time

        mprint(f'Batch time: {train_time} <=: preprocessing_time: {preprocessing_time}, seeking_time: {seeking_time}'
               f'first_train_time: {first_train_time}, iteration_time: {iteration_time},'
               f'rescore_time: {rescore_time}')

        self.train_time = {'train_time': train_time,
                           'preprocessing_time': preprocessing_time,
                           'seeking_time': seeking_time,
                           'model_fitting_time': first_train_time + iteration_time,
                           'rescore_time': rescore_time}

        mprint(f'n_iter: {self.model.n_iter_}', self.verbose, DEBUG)

        return self

    # def update_abnormal_thres(self, models, X_normal_proj):
    #     """Only use abnormal_score to update the abnormal_thres, in which,
    #         abnormal_score = y_score[y_score > abnormal_thres]
    #
    #     Parameters
    #     ----------
    #     models
    #     abnormal_thres
    #     X_normal_proj
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     y_score, model_predict_time = time_func(models.decision_function, X_normal_proj)
    #     abnormal_thres = np.quantile(y_score, q=self.params.q_abnormal_thres)  # abnormal threshold
    #     # models.y_score = y_score
    #
    #     return abnormal_thres

    def online_test(self, X_test, y_test):
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
        # if self.verbose >= DEBUG: data_info(np.asarray(X_test), name='X_test')
        test_time = 0.0
        X_test, preprocessing_time = time_func(self._preprocessing, X_test)
        test_time += preprocessing_time

        ##########################################################################################
        # Step 2: Evaluate GMM on the test set
        y_score, prediction_time = time_func(self.model.decision_function, X_test)
        self.auc, auc_time = time_func(self.get_score, y_test, y_score)
        if self.verbose >= DEBUG: data_info(np.asarray(y_score).reshape(-1, 1), name='y_score')
        test_time += prediction_time + auc_time

        mprint(f'Total test time: {test_time} <= preprocessing_time: {preprocessing_time}, '
               f'prediction_time: {prediction_time}, auc_time: {auc_time}, AUC: {self.auc}', self.verbose, DEBUG)

        self.test_time = {'test_time': test_time,
                          'preprocessing_time': preprocessing_time,
                          'prediction_time': prediction_time,
                          'auc_time': auc_time}
        # self._init_test(X_test, y_test)

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

        self.random_state = params.random_state
        self.verbose = params.verbose

        # stores important results
        self.info = {}

    def batch_train_test(self, X_arrival, y_arrival, X_test,
                         y_test):
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
        # ##########################################################################################
        # # # Step 1. Get initial models (init_model) on initial set (init_set) and evaluate it on test set
        # self.init_model = self.init_train_test(X_init_train, y_init_train, X_init_test, y_init_test)
        self.init_model = self.model
        # X_test, preprocessing_time = time_func(self._preprocessing, X_test)

        ##########################################################################################
        # Step 2. train the models on the batch data (previous+batch) and evaluate it.
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
                # if self.params.kjl:
                #     self.params.sigma_kjl= copy.deepcopy(self.kjl_inst.sigma_kjl)

            self.batch_train(X_batch, Y_batch)
            self.info['train_times'].append(self.train_time)
            self.info['model_params'].append(self.model.get_params())
            self.info['abnormal_threses'].append(self.abnormal_thres)

            # batch test models
            self.batch_test(X_test, y_test)
            self.info['test_times'].append(self.test_time)
            self.info['aucs'].append(self.auc)

            mprint(f'batch_{i + 1}: train_time: {self.train_time}, '
                   f'test_time: {self.test_time}, auc: {self.auc}', self.verbose, WARNING)

        self.info['params'] = self.params

        return self

    def batch_train(self, X_batch, y_batch=None):

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
        # if self.verbose >= DEBUG: data_info(np.asarray(X_batch), name='X_batch')
        if self.verbose >= DEBUG: data_info(np.asarray(self.X_acculumated_train), name='X_acculumated_train')

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

        if self.params.std:
            X_batch, _ = time_func(self.std_inst.transform, X_batch_raw)
        else:
            X_batch = X_batch_raw

        normal_idx = np.where((y_score <= self.abnormal_thres) == True)
        abnormal_idx = np.where((y_score > self.abnormal_thres) == True)
        if len(normal_idx[0]) > 0:
            X_normal = X_batch[normal_idx]
            y_normal = y_batch[normal_idx]
            self.X_acculumated_train = np.concatenate([self.X_acculumated_train, X_normal], axis=0)
            self.y_acculumated_train = np.concatenate([self.y_acculumated_train, y_normal], axis=0)
            abnormal_cnt = len(abnormal_idx[0])
        else:
            abnormal_cnt = len(abnormal_idx[0])
        mprint(f'X_train: {self.X_acculumated_train.shape}, drop {abnormal_cnt} abnormal flows.')
        # the datapoint is predicted as a abnormal flow, so we should drop it.
        mprint(f'{abnormal_cnt} flows are predicted as abnormal, so we drop them.', self.verbose, DEBUG)

        # Step 3.2: train a new models and use it to instead of the current one.
        # we use models.n_compoents to initialize n_components or the value found by quickhsift++ or meanshift
        # self.model_copy = copy.deepcopy(self.models)

        # update self.models: replace the current models with new_model
        if self.params.fixed_kjl:  # (fixed KJL, U, and STD)
            # use the init self.kjl to transform X_batch
            # Step 1: Preprocessing
            X_batch, prepro_time = time_func(self._preprocessing, self.X_acculumated_train)
            preprocessing_time += prepro_time

            # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
            self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
            if self.params.meanshift:
                dists = pairwise_distances(X_batch)
                if 0 < self.params.q_ms < 1:
                    sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl
                else:
                    sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl
                means_init, n_components, seeking_time, all_n_clusters = meanshift_seek_modes(
                    X_batch, bandwidth=sigma, thres_n=self.thres_n)
                self.params.n_components = n_components
            # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
            #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
            #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
            #         thres_n=self.thres_n)
            #     self.params['n_components'] = self.n_components
            else:
                seeking_time = 0
            train_time += seeking_time

            self.model = GMM(n_components=self.params.n_components, covariance_type=self.params.covariance_type,
                             random_state=self.params.random_state)
            ##########################################################################################
            # # Step 2: fit a models
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
                               'seeking_time': seeking_time,
                               'model_fitting_time': model_fitting_time,
                               'rescore_time': rescore_time}
        else:
            # data_info(self.X_acculumated_train, name='self.X_acculumated_train')
            seeking_time = 0.0
            if self.params.before_kjl_meanshift:
                # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
                start = datetime.now()
                self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
                if self.params.meanshift:
                    dists = pairwise_distances(self.X_acculumated_train)
                    if self.params.kjl: self.params.q_ms = self.params.q_kjl
                    if not hasattr(self.params, 'q_ms'): self.params.q_ms = 0.3
                    if 0 < self.params.q_ms < 1:
                        sigma = np.quantile(dists, self.params.q_ms)  # also used for kjl
                    else:
                        sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl
                    print(f'sigma: {sigma}, self.params.q_ms = {self.params.q_ms}')
                    means_init, n_components, seeking_time, all_n_clusters = meanshift_seek_modes(
                        self.X_acculumated_train, bandwidth=sigma, thres_n=self.thres_n)
                    self.params.n_components = n_components
                    self.model.n_components = n_components
                # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
                #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                #         thres_n=self.thres_n)
                #     self.params['n_components'] = self.n_components
                else:
                    seeking_time = 0.0
                end = datetime.now()
                seeking_time = (end - start).total_seconds()
                train_time += seeking_time

            # Step 1: Preprocessing
            if self.params.kjl:
                # Fit a kjl_inst on the X_train
                self.kjl_inst = KJL(self.params)
                _, kjl_time = time_func(self.kjl_inst.fit, self.X_acculumated_train, self.y_acculumated_train,
                                        self.X_acculumated_train, self.y_acculumated_train)
                # print(f'self.Xrow: {list(self.kjl_inst.Xrow)}, self.sigma_kjl: {self.kjl_inst.sigma_kjl}')
                proj_time = kjl_time
                # Transform X_train
                X_batch, kjl_time = time_func(self.kjl_inst.transform, self.X_acculumated_train)
                proj_time += kjl_time
            else:
                X_batch = self.X_acculumated_train
                proj_time = 0.0
            # preprocessing_time += proj_time
            if self.verbose >= DEBUG: data_info(X_batch, name='X_proj')

            end = datetime.now()
            preprocessing_time = (end - start).total_seconds()
            train_time += preprocessing_time

            seeking_time = 0.0
            if self.params.after_kjl_meanshift:
                # # Step 1.2: Seek modes of the data by quickshift++ or meanshift
                start = datetime.now()
                self.thres_n = 10  # used to filter clusters which have less than 100 datapoints
                if self.params.meanshift:
                    dists = pairwise_distances(X_batch)
                    if self.params.kjl: self.params.q_ms = self.params.q_kjl
                    if not hasattr(self.params, 'q_ms'): self.params.q_ms = 0.3
                    if 0 < self.params.q_ms < 1:
                        sigma = np.quantile(dists, self.params.q_ms)  # also used for kjl
                    else:
                        sigma = np.quantile(dists, self.params.q_kjl)  # also used for kjl
                    means_init, n_components, seeking_time, all_n_clusters = meanshift_seek_modes(
                        X_batch, bandwidth=sigma, thres_n=self.thres_n)
                    self.params.n_components = n_components
                    self.model.n_components = n_components
                # elif 'quickshift' in self.params.keys() and self.params['quickshift']:
                #     self.means_init, self.n_components, self.seek_train_time, self.all_n_clusters = quickshift_seek_modes(
                #         X_train, k=self.params['quickshift_k'], beta=self.params['quickshift_beta'],
                #         thres_n=self.thres_n)
                #     self.params['n_components'] = self.n_components
                else:
                    seeking_time = 0.0
                end = datetime.now()
                seeking_time = (end - start).total_seconds()
                train_time += seeking_time

            self.model = GMM(n_components=self.model.n_components, covariance_type=self.params.covariance_type,
                             random_state=self.params.random_state)
            ##########################################################################################
            # Step 2. Setting self.models'params (e.g., means_init, and n_components when use meanshift or quickshift)
            model_params = {'n_components': self.params.n_components,
                            'covariance_type': self.params.covariance_type,
                            'means_init': None, 'random_state': self.random_state}
            self.model.set_params(**model_params)
            # mprint(self.models.get_params(), self.verbose, DEBUG)
            # Fit self.models on the X_train
            _, model_fitting_time = time_func(self.model.fit, X_batch)
            train_time += model_fitting_time

            rescore_time = 0

            self.train_time = {'train_time': train_time,
                               'preprocessing_time': preprocessing_time,
                               'seeking_time': seeking_time,
                               'model_fitting_time': model_fitting_time,
                               'rescore_time': rescore_time}

        mprint(f'n_iter: {self.model.n_iter_}', self.verbose, DEBUG)
        mprint(f'self.train_time: {self.train_time}', self.verbose, 0)

        return self

    def batch_test(self, X_test, y_test):
        # if self.verbose >= DEBUG: data_info(np.asarray(X_test), name='X_test')
        test_time = 0.0
        X_test, preprocessing_time = time_func(self._preprocessing, X_test)
        test_time += preprocessing_time

        ##########################################################################################
        # Step 2: Evaluate GMM on the test set
        y_score, prediction_time = time_func(self.model.decision_function, X_test)
        self.auc, auc_time = time_func(self.get_score, y_test, y_score)
        if self.verbose >= DEBUG: data_info(np.asarray(y_score).reshape(-1, 1), name='y_score')
        test_time += prediction_time + auc_time

        mprint(f'Total test time: {test_time} <= preprocessing_time: {preprocessing_time}, '
               f'prediction_time: {prediction_time}, auc_time: {auc_time}, AUC: {self.auc}', self.verbose, DEBUG)

        self.test_time = {'test_time': test_time,
                          'preprocessing_time': preprocessing_time,
                          'prediction_time': prediction_time,
                          'auc_time': auc_time}
        # self._init_test(X_test, y_test)

        return self


def generate_experiment_cases(online=True, gs=True, n_repeats=5, q_abnormal_thres=1.0, fixed_kjl=False,
                              meanshift=False, before_kjl_meanshift=False, after_kjl_meanshift=False,
                              kjl=True, std=True, with_means_std=True, direction = 'src_dst',
                              verbose=10, batch_size=100, random_state=42):
    TEMPLATE = {'detector_name': '', 'gs': False, 'std': std, 'with_means_std': with_means_std,
                'kjl': False, 'nystrom': False, 'quickshift': False, 'direction':direction,
                'before_kjl_meanshift': before_kjl_meanshift, 'after_kjl_meanshift': after_kjl_meanshift,
                'meanshift': meanshift, 'online': online, 'random_state': random_state, 'n_repeats': n_repeats,
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
        create_case(template=TEMPLATE, detector_name='GMM', covariance_type='diag', gs=gs, kjl=kjl)
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
def main(n_init_train=500, *, gs=True, kjl=True, centering_kjl=True, std=True, with_means_std=True, n_kjl=100, d_kjl=10,
         random_state=42, meanshift=True):
    ##########################################################################################################
    # Step 0. All datasets
    # n_init_train = 500
    root_dir = 'online'
    direction = 'src_dst'
    in_dir = f'data/{direction}/iat_size/n_init_train_{n_init_train}'
    data_path_mappings = {
        # # 'DEMO_IDS': 'DEMO_IDS/DS-srcIP_192.168.10.5',
        'mimic_GMM': f'{in_dir}/mimic_GMM/Xy-normal-abnormal.dat',

        # 'UNB1_UNB2': f'{in_dir}/UNB1_UNB2/Xy-normal-abnormal.dat',  # UNB1 (abnorma1 = 280) , UNB2(abnormal=360)
        # 'UNB1_UNB2_UNB3': f'{in_dir}/UNB1_UNB2_UNB3/Xy-normal-abnormal.dat',    # UNB3 as abnormal
        # 'UNB1_UNB3_UNB2': f'{in_dir}/UNB1_UNB3_UNB2/Xy-normal-abnormal.dat',    # UNB2 as abnormal
        # 'UNB2_UNB3_UNB1': f'{in_dir}/UNB2_UNB3_UNB1/Xy-normal-abnormal.dat',    # UNB1 as abnormal
        # # 'UNB1_UNB4_UNB5': f'{in_dir}/UNB1_UNB4_UNB5/Xy-normal-abnormal.dat',
        # # 'UNB1_UNB3': f'{in_dir}/UNB1_UNB3/Xy-normal-abnormal.dat', #  UNB3(abnormal=367)
        # # 'UNB1_UNB4': f'{in_dir}/UNB1_UNB4/Xy-normal-abnormal.dat', # UNB4(abnormal=348)
        # # # # 'UNB1_UNB5': f'{in_dir}/UNB1_UNB5/Xy-normal-abnormal.dat', #  UNB2(abnormal=)
        # # 'UNB2_UNB3': f'{in_dir}/UNB2_UNB3/Xy-normal-abnormal.dat',
        # # 'UNB2_UNB4': f'{in_dir}/UNB2_UNB4/Xy-normal-abnormal.dat',
        # 'UNB1_CTU1': f'{in_dir}/UNB1_CTU1/Xy-normal-abnormal.dat',
        # 'UNB1_MAWI1': f'{in_dir}/UNB1_MAWI1/Xy-normal-abnormal.dat',
        # #
        # 'CTU1_UNB1': f'{in_dir}/CTU1_UNB1/Xy-normal-abnormal.dat',
        # 'CTU1_MAWI1': f'{in_dir}/CTU1_MAWI1/Xy-normal-abnormal.dat',
        # 'MAWI1_CTU1': f'{in_dir}/MAWI1_CTU1/Xy-normal-abnormal.dat',
        # #
        # 'UNB1_MAWI1_CTU1': f'{in_dir}/UNB1_MAWI1_CTU1/Xy-normal-abnormal.dat',  # CTU1 as abnormal
        # 'MAWI1_CTU1_UNB1': f'{in_dir}/MAWI1_CTU1_UNB1/Xy-normal-abnormal.dat',  # UNB1 as abnormal
        # 'UNB1_CTU1_MAWI1': f'{in_dir}/UNB1_CTU1_MAWI1/Xy-normal-abnormal.dat',  # MAWI as abnormal
        # #
        # 'UNB1_FRIGIDLE_OPEN':  f'{in_dir}/UNB1_FRIGIDLE_OPEN/Xy-normal-abnormal.dat',   # UNB(normal), Frig(idle + open_shut): open_shut as abnormal
        # 'FRIGIDLE_OPEN_UNB1':  f'{in_dir}/FRIGIDLE_OPEN_UNB1/Xy-normal-abnormal.dat',   # UNB(normal) as abnormal, Frig(idle + open_shut)
        # 'UNB1_FRIGIDLE_BROWSE': f'{in_dir}/UNB1_FRIGIDLE_BROWSE/Xy-normal-abnormal.dat',
        # 'FRIGIDLE_BROWSE_UNB1': f'{in_dir}/FRIGIDLE_BROWSE_UNB1/Xy-normal-abnormal.dat',
        # 'UNB1_FRIGOPEN_BROWSE': f'{in_dir}/UNB1_FRIGOPEN_BROWSE/Xy-normal-abnormal.dat',
        # 'FRIGOPEN_BROWSE_UNB1': f'{in_dir}/FRIGOPEN_BROWSE_UNB1/Xy-normal-abnormal.dat',
        #
        #
        # 'UNB1_AECHOIDLE_SHOP':  f'{in_dir}/UNB1_AECHOIDLE_SHOP/Xy-normal-abnormal.dat',
        # # 'AECHOIDLE_SHOP_UNB1':  f'{in_dir}/AECHOIDLE_SHOP_UNB1/Xy-normal-abnormal.dat',
        # # 'UNB1_AECHOIDLE_SONG': f'{in_dir}/UNB1_AECHOIDLE_SONG/Xy-normal-abnormal.dat',
        # # 'AECHOIDLE_SONG_UNB1': f'{in_dir}/AECHOIDLE_SHOP_UNB1/Xy-normal-abnormal.dat',
        # 'UNB1_AECHOSHOP_SONG': f'{in_dir}/UNB1_AECHOSHOP_SONG/Xy-normal-abnormal.dat',
        # 'AECHOSHOP_SONG_UNB1': f'{in_dir}/AECHOSHOP_SONG_UNB1/Xy-normal-abnormal.dat',
        #
        # ## -----------------------------------------------------------------
        #
        # # # # AECHO:
        # # subdatasets1 = (normal1, abnormal1)  # normal(idle) + abnormal(shop)
        # # subdatasets2 = (abnormal2, None)  # normal(song)
        # 'AECHO_IDLE_SHOP': f'{in_dir}/AECHO_IDLE_SHOP/Xy-normal-abnormal.dat',
        #
        # # # # # AECHO:
        # # # subdatasets1 = (normal1, abnormal2)  # normal(idle) + abnormal (song)
        # # # subdatasets2 = (abnormal1, None)  # normal(shop)
        # 'AECHO_IDLE_SONG': f'{in_dir}/AECHO_IDLE_SONG/Xy-normal-abnormal.dat',
        # #
        # # # # subdatasets1 = (abnormal1, abnormal2)  # normal(shop) + abnormal (song)
        # # # # subdatasets2 = (normal2, None)  # normal(idle2)
        # 'AECHO_SHOP_SONG': f'{in_dir}/AECHO_SHOP_SONG/Xy-normal-abnormal.dat',
        # # #
        # # # # subdatasets1 = (abnormal2, abnormal1)  # normal(song) + abnormal (shop)
        # # # # subdatasets2 = (normal2, None)  # normal(idle2)
        # 'AECHO_SONG_SHOP': f'{in_dir}/AECHO_SONG_SHOP/Xy-normal-abnormal.dat',
        #
        # # # # # Fridge:
        # # subdatasets1 = (abnormal2, normal1)  # normal(browse) + abnormal(idle1)
        # # # subdatasets2 = (abnormal1, None)  # normal(open_shut)
        # 'FRIG_BROWSE_OPEN': f'{in_dir}/FRIG_BROWSE_OPEN/Xy-normal-abnormal.dat',
        # #
        # # # # # Fridge:
        # # subdatasets1 = (abnormal2, normal2)  # normal(browse) + abnormal (idle2)
        # # subdatasets2 = (abnormal1, normal1)  # normal(open_shut) + abnormal(idle1)
        # 'FRIG_IDLE12': f'{in_dir}/FRIG_IDLE12/Xy-normal-abnormal.dat',
        # #
        # # # # # # # Fridge:
        # # # # subdatasets1 = (normal1, abnormal2)  # normal(idle1) + abnormal(browse)
        # # # # subdatasets2 = (abnormal1,None)  # normal(open_shut)
        # 'FRIG_IDLE1_OPEN': f'{in_dir}/FRIG_IDLE1_OPEN/Xy-normal-abnormal.dat',  # open and idle have much similar flows
        # # # #
        # # # # # # # # # Fridge:
        # # # # # # subdatasets1 = (abnormal1, abnormal2)  # normal(open_shut) + abnormal(browse)
        # # # # # # subdatasets2 = (normal1,None)  # normal(idle1)
        # 'FRIG_OPEN_IDLE1': f'{in_dir}/FRIG_OPEN_IDLE1/Xy-normal-abnormal.dat',
        # # # #
        # # # # # # # # Fridge:
        # # # # # subdatasets1 = (normal1, abnormal1)  # normal(idle1) + abnormal (open_shut)
        # # # # # subdatasets2 = (abnormal2,None)  # normal(browse)
        # 'FRIG_IDLE1_BROWSE': f'{in_dir}/FRIG_IDLE1_BROWSE/Xy-normal-abnormal.dat',
        # # # # # # # # Fridge:
        # # # # # subdatasets1 = (abnormal2, abnormal1)  # normal(browse) + abnormal (open_shut)
        # # # # # subdatasets2 = (normal1,None)  # normal(idle1)
        # 'FRIG_BROWSE_IDLE1': f'{in_dir}/FRIG_BROWSE_IDLE1/Xy-normal-abnormal.dat',
        # # #
        # # # # # # # Fridge:
        # # # # # subdatasets1 = (normal2, abnormal2)  # normal(idle2) + abnormal(browse)
        # # # # # subdatasets2 = (abnormal1,None)  # normal(open_shut)
        # 'FRIG_IDLE2_OPEN': f'{in_dir}/FRIG_IDLE2_OPEN/Xy-normal-abnormal.dat',
        # # #
        # # # # # # # # Fridge:
        # # # # # subdatasets1 = (abnormal1, abnormal2)  # normal(open_shut) + abnormal(browse)
        # # # # # subdatasets2 = (normal2,None)  # normal(idle2)
        # 'FRIG_OPEN_IDLE2': f'{in_dir}/FRIG_OPEN_IDLE2/Xy-normal-abnormal.dat',
        # # # #
        # # #
        # # # # # # # # # Fridge:
        # # # # # # subdatasets1 = (normal2, abnormal1)  # normal(idle2) + abnormal (open_shut)
        # # # # # # subdatasets2 = (abnormal2,None)  # normal(browse)
        # 'FRIG_IDLE2_BROWSE': f'{in_dir}/FRIG_IDLE2_BROWSE/Xy-normal-abnormal.dat',
        # # # # # # # # # Fridge:
        # # # # # # subdatasets1 = (abnormal2, abnormal1)  # normal(browse) + abnormal (open_shut)
        # # # # # # subdatasets2 = (normal2,None)  # normal(idle2)
        # 'FRIG_BROWSE_IDLE2': f'{in_dir}/FRIG_BROWSE_IDLE2/Xy-normal-abnormal.dat',
        # #
        # -----------------------------------------------------------------
        #
        # 'UNB2_CTU1': f'{in_dir}/UNB2_CTU1/Xy-normal-abnormal.dat',
        # 'UNB2_MAWI1': f'{in_dir}/UNB2_MAWI1/Xy-normal-abnormal.dat',
        # 'UNB2_FRIG1': f'{in_dir}/UNB2_FRIG1/Xy-normal-abnormal.dat',
        # # # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
        # 'UNB2_FRIG2': f'{in_dir}/UNB_FRIG2/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
        # # #
        # # 'MAWI1_UNB1': f'{in_dir}/MAWI1_UNB1/Xy-normal-abnormal.dat',
        # 'MAWI1_CTU1': f'{in_dir}/MAWI1_CTU1/Xy-normal-abnormal.dat',  # works
        # 'MAWI1_UNB2': f'{in_dir}/MAWI1_UNB2/Xy-normal-abnormal.dat',
        # 'CTU1_UNB2': f'{in_dir}/CTU1_UNB2/Xy-normal-abnormal.dat',
        #
        # # 'UNB1_FRIG1': f'{in_dir}/UNB1_FRIG1/Xy-normal-abnormal.dat',
        # # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
        # 'CTU1_FRIG1': f'{in_dir}/CTU1_FRIG1/Xy-normal-abnormal.dat',
        # # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
        # 'MAWI1_FRIG1': f'{in_dir}/MAWI1_FRIG1/Xy-normal-abnormal.dat',
        # MAWI1+Fridge: (normal: idle) abnormal: (open_shut)

        # 'FRIG1_UNB1': f'{in_dir}/FRIG1_UNB1/Xy-normal-abnormal.dat',
        # # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
        # 'FRIG1_CTU1': f'{in_dir}/FRIG1_CTU1/Xy-normal-abnormal.dat',
        # # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
        # 'FRIG1_MAWI1': f'{in_dir}/FRIG1_MAWI1/Xy-normal-abnormal.dat',
        #
        # # 'UNB1_FRIG2': f'{in_dir}/UNB1_FRIG2/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
        # # 'CTU1_FRIG2': f'{in_dir}/CTU1_FRIG2/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (browse)
        # # 'MAWI1_FRIG2': f'{in_dir}/MAWI1_FRIG2/Xy-normal-abnormal.dat',
        # # MAWI1+Fridge: (normal: idle) abnormal: (browse)

        # 'FRIG2_UNB1': f'{in_dir}/FRIG2_UNB1/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
        # 'FRIG2_CTU1': f'{in_dir}/FRIG2_CTU1/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (browse)
        # 'FRIG2_MAWI1': f'{in_dir}/FRIG2_MAWI1/Xy-normal-abnormal.dat',
        # MAWI1+Fridge: (normal: idle) abnormal: (browse)
        # #

        # SCAM has less than 100 abnormal flows, so it cannot be used
        # 'UNB1_SCAM1': f'{in_dir}/UNB1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        # 'CTU1_SCAM1': f'{in_dir}/CTU1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        # 'MAWI1_SCAM1': f'{in_dir}/MAWI1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        # 'FRIG1_SCAM1': f'{in_dir}/FRIG1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        # 'FRIG2_SCAM1': f'{in_dir}/FRIG2_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
        #
        # 'MACCDC1_UNB1': f'{in_dir}/MACCDC1_UNB1/Xy-normal-abnormal.dat',
        # 'MACCDC1_CTU1': f'{in_dir}/MACCDC1_CTU1/Xy-normal-abnormal.dat',
        # 'MACCDC1_MAWI1': f'{in_dir}/MACCDC1_MAWI1/Xy-normal-abnormal.dat',
        #
        # # less flows of wshr1
        'UNB1_DRYER1': f'{in_dir}/UNB1_DRYER1/Xy-normal-abnormal.dat',
        # 'DRYER1_UNB1': f'{in_dir}/DRYER1_UNB1/Xy-normal-abnormal.dat',
        #
        # # it works
        # 'UNB1_DWSHR1': f'{in_dir}/UNB1_DWSHR1/Xy-normal-abnormal.dat',
        # 'DWSHR1_UNB1': f'{in_dir}/DWSHR1_UNB1/Xy-normal-abnormal.dat',
        #
        # 'FRIG1_DWSHR1': f'{in_dir}/FRIG1_DWSHR1/Xy-normal-abnormal.dat',
        # 'FRIG2_DWSHR1': f'{in_dir}/FRIG2_DWSHR1/Xy-normal-abnormal.dat',
        # 'CTU1_DWSHR1': f'{in_dir}/CTU1_DWSHR1/Xy-normal-abnormal.dat',
        # 'MAWI1_DWSHR1': f'{in_dir}/MAWI1_DWSHR1/Xy-normal-abnormal.dat',
        # 'MACCDC1_DWSHR1': f'{in_dir}/MACCDC1_DWSHR1/Xy-normal-abnormal.dat',
        #
        # less flows of wshr1
        # 'UNB1_WSHR1': f'{in_dir}/UNB1_WSHR1/Xy-normal-abnormal.dat',
        # 'WSHR1_UNB1': f'{in_dir}/WSHR1_UNB1/Xy-normal-abnormal.dat',

    }
    print(f'len(data_path_mappings): {len(data_path_mappings)}, {data_path_mappings}')
    # random_state = RANDOM_STATE
    overwrite = 0
    verbose = VERBOSE

    ratios =  [0.5, 0.8, 0.9, 0.95, 1.0] # [0.5, 0.8, 0.9, 0.95, 1.0]
    # gs=True
    ms = meanshift  # if gs else True
    if ms:
        before_kjl_meanshift = True
        after_kjl_meanshift = False if before_kjl_meanshift else True
    else:
        before_kjl_meanshift = False
        after_kjl_meanshift = False
    fixed_kjl = False
    # kjl = True
    # std = True
    with_means_std = with_means_std
    if not std:
        with_means_std = False
    experiment_cases = generate_experiment_cases(n_repeats=5, fixed_kjl=fixed_kjl,
                                                 meanshift=ms, before_kjl_meanshift=before_kjl_meanshift,
                                                 after_kjl_meanshift=after_kjl_meanshift,
                                                 gs=gs, direction = direction,
                                                 q_abnormal_thres=1, std=std, with_means_std=with_means_std,
                                                 batch_size=50, kjl=kjl,
                                                 verbose=verbose, random_state=random_state)
    if not kjl:
        n_kjl = 0
        d_kjl = 0
        centering_kjl = False
    else:
        # n_kjl = 100
        # d_kjl = 10
        centering_kjl = centering_kjl

    def single_case(data_name, data_file, percent_first_init, experiment_case):
        try:
            info = OrderedDict()

            sc = SINGLE_CASE(data_file=data_file, percent_first_init=percent_first_init,
                             random_state=experiment_case['random_state'],
                             overwrite=overwrite, verbose=VERBOSE)
            params = PARAM(data_file=data_file, data_name=data_name, percent_first_init=percent_first_init,
                           random_state=experiment_case['random_state'], n_init_train=n_init_train, n_kjl=n_kjl,
                           d_kjl=d_kjl,
                           centering_kjl=centering_kjl,
                           overwrite=overwrite, verbose=VERBOSE)
            params.add_param(**experiment_case)
            params_str = str(params).replace('-', '\n\t')
            sc.run(params)

            # out_file = f'{root_dir}/out/{data_file}-case0-ratio_{percent_first_init}.dat'
            out_file = f'{root_dir}/out/{pth.dirname(data_file)}/gs={gs}-std={std}_center={with_means_std}-' \
                       f'kjl={kjl}-d_kjl={d_kjl}-n_kjl={n_kjl}-c_kjl={centering_kjl}-' \
                       f'ms={ms}-before_kjl={before_kjl_meanshift}-fixed_kjl={fixed_kjl}-seed={random_state}/case0-ratio_{percent_first_init}.dat'
            dump_data(sc.info, out_file)  # pickle cannot be used in mulitprocessing, try dill (still not work)
            sc.display(sc.info, out_file + '.pdf', key=(data_name, data_file, percent_first_init, params_str),
                       is_show=True)  # sc.info = {'online_GMM': , 'batch_GMM': ''}
            print('+++online:')
            _aucs = np.asarray([np.asarray(v['aucs']) for i, v in sc.info['online'].items()])
            print(_aucs)
            # print([(f'repeat_{i}', v) for i, v in enumerate(_aucs)])
            print(f'mean: {np.mean(_aucs, axis=0)}\nstd:{np.std(_aucs, axis=0)}')
            print('+++batch:')
            _aucs = np.asarray([np.asarray(v['aucs']) for i, v in sc.info['batch'].items()])
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
    n_jobs = int(joblib.cpu_count() // 3) or 1
    print(f'n_job: {n_jobs}')
    parallel = Parallel(n_jobs=n_jobs, verbose=30)
    with parallel:
        outs = parallel(
            delayed(single_case)(data_name, os.path.join(root_dir, data_file), percent_first_init, experiment_case)
            for ((data_name, data_file), percent_first_init, experiment_case) in \
            itertools.product(data_path_mappings.items(), ratios, experiment_cases))

    out_file = f'{root_dir}/out/{root_dir}/{in_dir}/gs={gs}-std={std}_center={with_means_std}-' \
               f'kjl={kjl}-d_kjl={d_kjl}-n_kjl={n_kjl}-c_kjl={centering_kjl}-' \
               f'ms={ms}-before_kjl={before_kjl_meanshift}-fixed_kjl={fixed_kjl}-seed={random_state}/res.dat'
    print(f'***final_results: {out_file}')
    dump_data(outs, out_file=out_file) # _pickle.PicklingError: Can't pickle <class 'abc.ONLINE_GMM_MAIN'>: attribute lookup ONLINE_GMM_MAIN on abc failed
    individual_img(in_file=out_file,root_dir = 'online')  # from report import individual_img
    in_file = out_file
    out_file += '.xlsx'
    imgs2xlsx(in_file, data_path_mappings,  out_file, root_dir)
    print(out_file)
    mprint("\nFinish!")

    return out_file


if __name__ == '__main__':
    demo = True
    if demo:
        init_sizes = [500]  # [100, 200, 500, 700]
        gses = [False] # [False, True]
        kjls = [True] # [False, True]
        centering_kjls = [False]  # [True, False]
        stds =  [False]
        with_means_stds = [False]
        n_kjls =[100] #[100, 50, 200]
        d_kjls =[5]  #[5, 20]
        meanshifts=[False]  #  [False, True]
        random_states = [np.power(10, i) for i in range(1, 1 + 1)]
        # random_states = [42]
    else:
        init_sizes = [500]  # [100, 200, 500, 700]
        gses = [True, False]
        kjls = [True]  # [True, False]
        centering_kjls = [False]  # [True, False]
        stds = [False]
        with_means_stds = [False]
        n_kjls = [100, 200, 300, 400, 500]
        d_kjls = [5, 10, 50, 100, 200]
        meanshifts = [False, True]
        random_states = [np.power(10, i) for i in range(1, 3 + 1)]
    i = 0
    out_files = []
    for init_size, gs, kjl, centering_kjl, std, with_means_std, n_kjl, d_kjl, random_state, ms in itertools.product(
            init_sizes, gses, kjls, centering_kjls, stds, with_means_stds, n_kjls, d_kjls, random_states, meanshifts):
        if not kjl:  # and i ==0:
            # for kjl=False, only need process one time
            i += 1
            n_kjl = 0
            d_kjl = 0
            print(
                f'***n_init_train={init_size}, gs={gs}, kjl={kjl}, centering_kjl={centering_kjl}, std={std}, with_means_std={with_means_std}, n_kjl={n_kjl}, d_kjl={d_kjl}, random_state={random_state}, meanshift={ms}')
            out_file = main(n_init_train=init_size, gs=gs, kjl=kjl, centering_kjl=centering_kjl, std=std, meanshift=ms,
                            with_means_std=with_means_std, n_kjl=n_kjl, d_kjl=d_kjl, random_state=random_state)
        if kjl:
            print(
                f'***n_init_train={init_size}, gs={gs}, kjl={kjl}, centering_kjl={centering_kjl}, std={std}, with_means_std={with_means_std}, n_kjl={n_kjl}, d_kjl={d_kjl}, random_state={random_state}, meanshift={ms}')
            out_file = main(n_init_train=init_size, gs=gs, kjl=kjl, centering_kjl=centering_kjl, std=std, meanshift=ms,
                            with_means_std=with_means_std, n_kjl=n_kjl, d_kjl=d_kjl, random_state=random_state)

        out_files.append(out_file)
    ##### merge multi-xlsx into one
    # import os
    # import pandas as pd
    #
    # # cwd = os.path.abspath('')
    # # files = os.listdir(cwd)
    # df = pd.DataFrame()
    # for out_file in out_files:
    #     if out_file.endswith('.xlsx'):
    #         df = df.append(pd.read_excel(out_file), ignore_index=True)
    # df.head()
    # df.to_excel(out_file + '-total.xlsx')

    # import xlwt
    # import xlrd
    #
    # wkbk = xlwt.Workbook()
    # outsheet = wkbk.add_sheet('Sheet1')
    #
    # xlsfiles = out_files
    #
    # outrow_idx = 0
    # for f in xlsfiles:
    #     # This is all untested; essentially just pseudocode for concept!
    #     insheet = xlrd.open_workbook(f).sheets()[0]
    #     for row_idx in range(insheet.nrows):
    #         for col_idx in range(insheet.ncols):
    #             outsheet.write(outrow_idx, col_idx,
    #                            insheet.cell_value(row_idx, col_idx))
    #         outrow_idx += 1
    # wkbk.save(out_file + '-total.xlsx')
