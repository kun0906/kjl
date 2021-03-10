"""Main classes and functions for obtaining the speedup results
"""
# Authors: kun.bj@outlook.com
# License: xxx

import copy
import itertools
import os.path as pth
import time
import traceback
from collections import Counter

import numpy as np
import sklearn
from func_timeout import func_set_timeout, FunctionTimedOut
from joblib import delayed, Parallel
from sklearn import metrics
from sklearn.metrics import pairwise_distances, roc_curve

from kjl.log import get_log
from kjl.model.gmm import GMM, compute_gmm_init
from kjl.model.kjl import KJL
from kjl.model.nystrom import NYSTROM
from kjl.model.ocsvm import OCSVM
from kjl.model.seek_mode import SeekModes
from kjl.utils.data import dump_data, split_train_val, split_left_test, seperate_normal_abnormal
from kjl.utils.tool import execute_time

# create a customized log instance that can print the information.
lg = get_log(level='info')

FUNC_TIMEOUT = 3 * 60  # (if function takes more than 3 mins, then it will be killed)


class BASE:

    def __init__(self, random_state=42):
        self.random_state = random_state

    # # use_signals=False: the fuction cannot return a object that cannot be pickled (here "self" is not pickled,
    # # so it will be PicklingError)
    # # use_signals=True: it only works on main thread (here train_test_intf is not the main thread)
    # @timeout_decorator.timeout(seconds=30 * 60, use_signals=False, timeout_exception=StopIteration)
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
        start = time.process_time()
        try:
            model.fit(X_train)
        except (TimeoutError, Exception) as e:
            msg = f'fit error: {e}'
            raise ValueError(f'{msg}')
        end = time.process_time()
        train_time = end - start
        # lg.debug("Fitting model takes {} seconds".format(train_time))

        return model, train_time

    def _test(self, model, X_test, y_test):
        """Evaluate the model on the X_test, y_test

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

        self.test_time = 0

        #####################################################################################################
        # 1. standardization
        # start = time.process_time()
        # # if self.params['is_std']:
        # #     X_test = self.scaler.transform(X_test)
        # # else:
        # #     pass
        # end = time.process_time()
        # self.std_test_time = end - start
        self.std_test_time = 0
        self.test_time += self.std_test_time

        #####################################################################################################
        # 2. projection
        start = time.process_time()
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            X_test = self.project.transform(X_test)
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            X_test = self.project.transform(X_test)
        else:
            pass
        end = time.process_time()
        self.proj_test_time = end - start
        self.test_time += self.proj_test_time

        # no need to do seek in the testing phase
        self.seek_test_time = 0

        #####################################################################################################
        # 3. prediction
        start = time.process_time()
        # For inlier, a small value is used; a larger value is for outlier (positive)
        # it must be abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
        y_score = model.decision_function(X_test)
        end = time.process_time()
        self.model_test_time = end - start
        self.test_time += self.model_test_time

        # For binary  y_true, y_score is supposed to be the score of the class with greater label.
        # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
        # pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        self.auc = metrics.auc(fpr, tpr)
        lg.debug(f"AUC: {self.auc}")

        lg.info(f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, '
                f'seek_test_time: {self.seek_test_time}, proj_test_time: {self.proj_test_time}, '
                f'model_test_time: {self.model_test_time}')

        return self.auc, self.test_time

    def save(self, data, out_file='.dat'):
        dump_data(data, name=out_file)


class GMM_MAIN(BASE):

    def __init__(self, params):
        super(GMM_MAIN, self).__init__()

        self.params = params
        self.random_state = params['random_state']

    def train(self, X_train, y_train=None):
        """

        Parameters
        ----------
        X_train
        y_train

        Returns
        -------

        """

        self.train_time = 0
        N, D = X_train.shape

        #####################################################################################################
        # 1.1 normalization
        # start = time.process_time()
        # # if self.params['is_std']:
        # #     self.scaler = StandardScaler(with_mean=self.params['is_std_mean'])
        # #     self.scaler.fit(X_train)
        # #     X_train = self.scaler.transform(X_train)
        # #     # if self.verbose > 10: data_info(X_train, name='X_train')
        # # else:
        # #     pass
        # end = time.process_time()
        # self.std_train_time = end - start
        self.std_train_time = 0
        self.train_time += self.std_train_time

        #####################################################################################################
        # 1.2. projection
        start = time.process_time()
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            self.project = KJL(self.params)
            self.project.fit(X_train, y_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['kjl_d']
            n = self.params['kjl_n']
            q = self.params['kjl_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            self.project = NYSTROM(self.params)
            self.project.fit(X_train, y_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['nystrom_d']
            n = self.params['nystrom_n']
            q = self.params['nystrom_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
        else:
            d = D
            n = N
            q = 0.25
        # self.params['is_kjl'] = False # for debugging
        # d, n, q = D, N, self.params['kjl_q']
        end = time.process_time()
        self.proj_train_time = end - start
        self.train_time += self.proj_train_time

        #####################################################################################################
        # 1.3 seek modes after projection
        start = time.process_time()
        if self.params['after_proj']:
            self.thres_n = 0.95  # used to filter clusters
            if 'is_quickshift' in self.params.keys() and self.params['is_quickshift']:
                self.seek_mode = SeekModes(seek_name='quickshift', random_state=self.random_state)
                self.seek_mode.fit(X_train, qs_k=self.params['quickshift_k'], qs_beta=self.params['quickshift_beta'],
                                   thres=self.thres_n, GMM_covariance_type=self.params['GMM_covariance_type'],
                                   n_comp_thres=20)
                self.n_components = self.seek_mode.n_clusters
                self.tot_clusters = self.seek_mode.tot_clusters
                self.params['qs_res'] = {'tot_clusters': self.tot_clusters,
                                         'n_clusters': self.n_components,
                                         'q_proj': q,
                                         'k_qs': self.params['quickshift_k'],
                                         'beta_qs': self.params['quickshift_beta']}
                self.params['GMM_n_components'] = self.n_components

            if 'is_meanshift' in self.params.keys() and self.params['is_meanshift']:
                # if hasattr(self, 'sigma'):
                #     pass
                # else:
                #     dists = pairwise_distances(X_train)
                #     self.sigma = np.quantile(dists, self.params['kjl_q'])  # also used for kjl
                # self.seek_mode = SeekModes(seek_name='meansshift', random_state=self.random_state)
                # self.means_init, self.n_components, self.clusters, self.seek_train_time, self.all_n_clusters = meanshift_seek_modes(
                #     X_train, bandwidth=self.sigma, thres_n=self.thres_n)
                # self.n_components = self.seek_mode.n_clusters
                # self.tot_clusters = self.seek_mode.tot_clusters
                # self.params['ms_res'] = {'tot_clusters': self.tot_clusters, 'n_clusters': self.n_components,
                #                          'q_proj': q}
                # self.params['GMM_n_components'] = 20 if self.n_components > 20 else self.n_components
                msg = 'meanshift'
                raise NotImplementedError(msg)

        else:
            pass
        end = time.process_time()
        self.seek_train_time = end - start
        self.train_time += self.seek_train_time

        #####################################################################################################
        # 2.1 Initialize the model
        start = time.process_time()
        model = GMM()
        if self.params['GMM_is_init_all'] and (self.params['is_quickshift'] or self.params['is_meanshift']):
            # get the init params of GMM
            self.weights_init, self.means_init, self.precisions_init, _ = compute_gmm_init(
                n_components=self.params['GMM_n_components'],
                X=X_train, n_thres=self.seek_mode.n_thres,
                tot_clusters=self.tot_clusters,
                tot_labels=self.seek_mode.tot_labels,
                covariance_type=self.params['GMM_covariance_type'])

            model_params = {'n_components': self.params['GMM_n_components'],
                            'covariance_type': self.params['GMM_covariance_type'],
                            'weights_init': self.weights_init,
                            'means_init': self.means_init,
                            'precisions_init': self.precisions_init,
                            'random_state': self.random_state}
        else:
            model_params = {'n_components': self.params['GMM_n_components'],
                            'covariance_type': self.params['GMM_covariance_type'],
                            'means_init': None, 'random_state': self.random_state}
        # set model default parameters
        model.set_params(**model_params)
        end = time.process_time()
        self.init_model_time = end - start
        self.train_time += self.init_model_time
        lg.debug(f'model.get_params(): {model.get_params()}')

        #####################################################################################################
        # 2.2 Train the model
        try:
            self.model, self.model_train_time = self._train(model, X_train)
        except (FunctionTimedOut, Exception) as e:
            lg.warning(f'{e}, retrain with a larger reg_covar')
            model.reg_covar = 1e-5
            self.model, self.model_train_time = self._train(model, X_train)
        self.train_time += self.model_train_time

        #####################################################################################################
        # 3. get space size
        start = time.process_time()
        if self.model.covariance_type == 'full':
            # space_size = (d ** 2 + d) * n_comps + n * (d + D)
            self.space_size = (d ** 2 + d) * self.model.n_components + n * (d + D)
        elif self.model.covariance_type == 'diag':
            # space_size = (2* d) * n_comps + n * (d + D)
            self.space_size = (2 * d) * self.model.n_components + n * (d + D)
        else:
            msg = self.model.covariance_type
            raise NotImplementedError(msg)
        end = time.process_time()
        self.space_train_time = end - start
        # self.train_time += self.space_train_time

        self.N = N
        self.D = D

        lg.info(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
                f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
                f'init_model_time: {self.init_model_time}, model_train_time: {self.model_train_time}, '
                f'D:{D}, space_size: {self.space_size}, N:{N}, n_comp: {self.model.n_components}, d: {d}, n: {n}, '
                f'q: {q}')
        
        return self

    def test(self, X_test, y_test):
        return self._test(self.model, X_test, y_test)


class OCSVM_MAIN(BASE):

    def __init__(self, params):
        super(OCSVM_MAIN, self).__init__()

        self.params = params
        self.random_state = params['random_state']

    def train(self, X_train, y_train=None):
        """

        Parameters
        ----------
        X_train
        y_train
        Returns
        -------

        """

        self.train_time = 0
        N, D = X_train.shape

        #####################################################################################################
        # 1.1 normalization
        # start = time.process_time()
        # # if self.params['is_std']:
        # #     self.scaler = StandardScaler(with_mean=self.params['is_std_mean'])
        # #     self.scaler.fit(X_train)
        # #     X_train = self.scaler.transform(X_train)
        # #     # if self.verbose > 10: data_info(X_train, name='X_train')
        # # else:
        # #     pass
        # end = time.process_time()
        # self.std_train_time = end - start
        self.std_train_time = 0
        self.train_time += self.std_train_time

        ######################################################################################################
        # 1.2 OCSVM does not need to seek modes
        self.seek_train_time = 0

        ######################################################################################################
        # 1.3 Projection
        start = time.process_time()
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            # self.sigma = np.sqrt(X_train.shape[0]* X_train.var())
            self.project = KJL(self.params)
            self.project.fit(X_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['kjl_d']
            n = self.params['kjl_n']
            q = self.params['kjl_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
            # use 'linear' kernel for OCSVM when kjl is True
            self.params['OCSVM_kernel'] = 'linear'
            model_params = {'kernel': self.params['OCSVM_kernel'], 'nu': self.params['OCSVM_nu']}

        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            self.project = NYSTROM(self.params)
            self.project.fit(X_train)
            X_train = self.project.transform(X_train)
            self.sigma = self.project.sigma
            d = self.params['nystrom_d']
            n = self.params['nystrom_n']
            q = self.params['nystrom_q']
            lg.debug(f'self.sigma: {self.sigma}, q={q}')
            self.params['OCSVM_kernel'] = 'linear'
            model_params = {'kernel': self.params['OCSVM_kernel'], 'nu': self.params['OCSVM_nu']}

        else:
            # when KJL=False or Nystrom=False, using rbf for OCSVM.
            sigma = np.quantile(pairwise_distances(X_train), self.params['OCSVM_q'])
            self.sigma = 1e-7 if sigma == 0 else sigma
            self.model_gamma = 1 / self.sigma ** 2
            q = self.params['OCSVM_q']
            lg.debug(f'model_sigma: {self.sigma}, model_gamma: {self.model_gamma}, q={q}')
            self.params['OCSVM_kernel'] = 'rbf'
            model_params = {'kernel': self.params['OCSVM_kernel'], 'gamma': self.model_gamma,
                            'nu': self.params['OCSVM_nu']}

        end = time.process_time()
        self.proj_train_time = end - start
        self.train_time += self.proj_train_time

        ######################################################################################################
        # 2.1 Initialize the model with preset parameters
        start = time.process_time()
        model = OCSVM()
        # set model default parameters
        model.set_params(**model_params)
        end = time.process_time()
        self.init_model_time = end - start
        self.train_time += self.init_model_time
        lg.info(f'model.get_params(): {model.get_params()}')

        ######################################################################################################
        # 2.2 Build the model with train set
        try:
            self.model, self.model_train_time = self._train(model, X_train)
        except (FunctionTimedOut, Exception) as e:
            lg.warning(f'{e}, try a fixed number of iterations (here is 1000)')
            model.max_iter = 1000  #
            self.model, self.model_train_time = self._train(model, X_train)
        self.train_time += self.model_train_time

        ######################################################################################################
        # 3. Get space size based on support vectors
        start = time.process_time()
        n_sv = self.model.support_vectors_.shape[0]  # number of support vectors
        if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
            self.space_size = n_sv + n_sv * d + n * (d + D)
        elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
            self.space_size = n_sv + n_sv * d + n * (d + D)
        else:
            self.space_size = n_sv + n_sv * D
        end = time.process_time()
        self.space_train_time = end - start
        # self.train_time += self.space_train_time

        self.n_sv = n_sv
        self.D = D
        self.N = N

        lg.info(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, seek_train_time: '
                f'{self.seek_train_time}, proj_train_time: {self.proj_train_time}, '
                f'init_model_time: {self.init_model_time}, model_train_time: {self.model_train_time}, '
                f'n_sv: {n_sv}, D:{D}, space_size: {self.space_size}, N:{N}, q: {q}')

        return self

    def test(self, X_test, y_test):
        return self._test(self.model, X_test, y_test)


def _model_train_test(X_train, y_train, X_test, y_test, params, **kwargs):
    """

    Parameters
    ----------
    params
    kargs

    Returns
    -------

    """
    # ##################### memory allocation snapshot
    #
    # tracemalloc.start()
    # start_time = time.process_time()
    # snapshot1 = tracemalloc.take_snapshot()
    # lg.debug(kwargs, params)
    try:
        ################################################################################################################
        # 1. Update parameters
        for k, v in kwargs.items():
            params[k] = v

        ################################################################################################################
        # 2. Build model and evaluate it
        if "GMM" in params['detector_name']:
            model = GMM_MAIN(params)
        elif "OCSVM" in params['detector_name']:
            model = OCSVM_MAIN(params)
        model.train(X_train, y_train)

        is_average = True
        if is_average: # time more stable
            auc = []
            test_time = []
            for _ in range(100):
                auc_, test_time_ = model.test(copy.deepcopy(X_test), copy.deepcopy(y_test))
                auc.append(auc_)
                test_time.append(test_time_)
            auc = np.mean(auc)
            test_time = np.mean(test_time)
        else:
            auc, test_time = model.test(copy.deepcopy(X_test), copy.deepcopy(y_test))
        lg.info(f'auc: {auc} =? {model.auc}, test_time: {test_time} =? {model.test_time}, X_test: {X_test.shape}')

        info = {'train_time': model.train_time, 'test_time': test_time, 'auc': auc,
                'params': model.params, 'space_size': model.space_size, 'model': copy.deepcopy(model),
                'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}

    except (FunctionTimedOut, Exception) as e:
        # lg.debug(f"FunctionTimedOut or Error: {e}")
        traceback.print_exc()
        info = {'train_time': 0.0, 'test_time': 0.0, 'auc': 0.0, 'apc': '',
                'params': params, 'space_size': 0.0, 'model': '',
                'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape}

    # ################### Second memory allocation snapshot
    # snapshot2 = tracemalloc.take_snapshot()
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    # lg.debug("[ Top 5 ]")
    # for stat in top_stats[:5]:
    #     lg.debug(stat)

    del model

    import gc
    gc.collect()

    return info


@execute_time
def get_best_results(X_train, y_train, X_val, y_val, X_test, y_test, params, is_parallel=False, random_state=42):
    """

    Parameters
    ----------
    X_train
    y_train
    X_val
    y_val
    X_test
    y_test
    params: dict
        model_params
    is_parallel
    random_state

    Returns
    -------

    """
    ################################################################################################################
    # print data size
    is_gs = params['is_gs']
    if is_gs:
        # Pick the best parameters according to the val set
        lg.debug(f'--is_gs: {is_gs}, X_val != X_test')
        if len(y_val) < 10:
            lg.warning('just for avoiding error when gs=True')
            X_val, y_val = sklearn.utils.resample(X_test, y_test, n_samples=10, stratify=y_test)
    else:
        # Use the default parameter
        X_val = X_test
        y_val = y_test
        lg.debug(f'--is_gs: {is_gs}, X_val == X_test')
    lg.debug(f'X_train.shape: {X_train.shape}, y_train: {Counter(y_train)}')
    lg.debug(f'X_val.shape: {X_val.shape}, y_val: {Counter(y_val)}')
    lg.debug(f'X_test.shape: {X_test.shape}, y_test: {Counter(y_test)}')

    ################################################################################################################
    # 1. Get the best result
    if is_parallel:
        parallel = Parallel(n_jobs=params['n_jobs'], verbose=30, backend='loky', pre_dispatch=1, batch_size=1)

    # best params and defaults params use the same API
    if 'GMM' in params['model_name']:
        ################################################################################################################
        # 2.1 GMM
        n_components_arr = params['GMM_n_components']
        del params['GMM_n_components']
        if params['model_name'] == 'GMM(full)' or params['model_name'] == 'GMM(diag)':
            ################################################################################################################
            # only GMM
            if is_parallel:
                with parallel:
                    outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                               copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                               copy.deepcopy(params),
                                                               GMM_n_components=n_components) for n_components, _ in
                                    list(itertools.product(n_components_arr, [0])))
            else:
                outs = []
                for n_components, _ in list(itertools.product(n_components_arr, [0])):
                    out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                            copy.deepcopy(X_val), copy.deepcopy(y_val),
                                            copy.deepcopy(params),
                                            GMM_n_components=n_components)
                    outs.append(out)


        elif params['is_kjl']:  # KJL-GMM
            ################################################################################################################
            # KJL
            kjl_ds = params['kjl_ds']
            kjl_ns = params['kjl_ns']
            kjl_qs = params['kjl_qs']
            del params['kjl_ds']
            del params['kjl_ns']
            del params['kjl_qs']
            if params['model_name'] == 'KJL-GMM(full)' or params['model_name'] == 'KJL-GMM(diag)':
                if is_parallel:
                    with parallel:
                        outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                                   copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                                   copy.deepcopy(params),
                                                                   kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                                                   GMM_n_components=n_components) for
                                        kjl_d, kjl_n, kjl_q, n_components in
                                        list(itertools.product(kjl_ds, kjl_ns, kjl_qs, n_components_arr)))
                else:
                    outs = []
                    for kjl_d, kjl_n, kjl_q, n_components in list(itertools.product(kjl_ds, kjl_ns,
                                                                                    kjl_qs, n_components_arr)):
                        out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                copy.deepcopy(params),
                                                kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                                GMM_n_components=n_components)
                        outs.append(out)

            elif params['model_name'] == 'KJL-QS-GMM(full)' or params['model_name'] == 'KJL-QS-GMM(diag)' or \
                    params['model_name'] == 'KJL-QS-init_GMM(full)' or params['model_name'] == 'KJL-QS-init_GMM(diag)':
                quickshift_ks = params['quickshift_ks']
                quickshift_betas = params['quickshift_betas']
                del params['quickshift_ks']
                del params['quickshift_betas']
                if is_parallel:
                    with parallel:
                        outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                                   copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                                   copy.deepcopy(params),
                                                                   kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                                                   quickshift_k=quickshift_k,
                                                                   quickshift_beta=quickshift_beta)
                                        for kjl_d, kjl_n, kjl_q, quickshift_k, quickshift_beta in
                                        list(
                                            itertools.product(kjl_ds, kjl_ns, kjl_qs, quickshift_ks, quickshift_betas)))
                else:
                    outs = []
                    for kjl_d, kjl_n, kjl_q, quickshift_k, quickshift_beta in list(itertools.product(
                            kjl_ds, kjl_ns, kjl_qs, quickshift_ks, quickshift_betas)):
                        out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                copy.deepcopy(params),
                                                kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                                quickshift_k=quickshift_k,
                                                quickshift_beta=quickshift_beta)
                        outs.append(out)

            elif params['model_name'] == 'KJL-MS-GMM(full)' or params['model_name'] == 'KJL-MS-GMM(diag)':
                # meanshift uses the same kjl_qs, and only needs to tune one of them
                meanshift_qs = params['meanshift_qs']
                del params['meanshift_qs']
                if is_parallel:
                    with parallel:
                        outs = parallel(
                            delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                       copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                       copy.deepcopy(params),
                                                       kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q)
                            for kjl_d, kjl_n, kjl_q in
                            list(itertools.product(kjl_ds, kjl_ns, kjl_qs)))
                else:
                    outs = []
                    for kjl_d, kjl_n, kjl_q in list(itertools.product(kjl_ds, kjl_ns, kjl_qs)):
                        out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                copy.deepcopy(params),
                                                kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q)
                        outs.append(out)

            else:
                msg = params['model_name']
                raise NotImplementedError(f'Error: {msg}')

        elif params['is_nystrom']:
            ################################################################################################################
            # Nystrom-GMM
            nystrom_ns = params['nystrom_ns']
            nystrom_ds = params['nystrom_ds']
            nystrom_qs = params['nystrom_qs']
            del params['nystrom_ns']
            del params['nystrom_ds']
            del params['nystrom_qs']
            if params['model_name'] == 'Nystrom-GMM(full)' or params['model_name'] == 'Nystrom-GMM(diag)':
                if is_parallel:
                    with parallel:
                        outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                                   copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                                   copy.deepcopy(params),
                                                                   nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                                                   nystrom_q=nystrom_q,
                                                                   GMM_n_components=n_components) for
                                        nystrom_n, nystrom_d, nystrom_q, n_components in
                                        list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, n_components_arr)))
                else:
                    outs = []
                    for nystrom_n, nystrom_d, nystrom_q, n_components in list(itertools.product(
                            nystrom_ns, nystrom_ds, nystrom_qs, n_components_arr)):
                        out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                copy.deepcopy(params),
                                                nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                                nystrom_q=nystrom_q,
                                                GMM_n_components=n_components)
                        outs.append(out)

            elif params['model_name'] == 'Nystrom-QS-GMM(full)' or params['model_name'] == 'Nystrom-QS-GMM(diag)' or \
                    params['model_name'] == 'Nystrom-QS-init_GMM(full)' or params['model_name'] == \
                    'Nystrom-QS-init_GMM(diag)':
                quickshift_ks = params['quickshift_ks']
                quickshift_betas = params['quickshift_betas']
                del params['quickshift_ks']
                del params['quickshift_betas']
                if is_parallel:
                    with parallel:
                        outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                                   copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                                   copy.deepcopy(params),
                                                                   nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                                                   nystrom_q=nystrom_q,
                                                                   quickshift_k=quickshift_k,
                                                                   quickshift_beta=quickshift_beta)
                                        for nystrom_n, nystrom_d, nystrom_q, quickshift_k, quickshift_beta in
                                        list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, quickshift_ks,
                                                               quickshift_betas)))
                else:
                    outs = []
                    for nystrom_n, nystrom_d, nystrom_q, quickshift_k, quickshift_beta in list(itertools.product(
                            nystrom_ns, nystrom_ds, nystrom_qs, quickshift_ks, quickshift_betas)):
                        out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                copy.deepcopy(params),
                                                nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                                nystrom_q=nystrom_q,
                                                quickshift_k=quickshift_k,
                                                quickshift_beta=quickshift_beta)
                        outs.append(out)

            elif params['model_name'] == 'Nystrom-MS-GMM(full)' or params['model_name'] == 'Nystrom-MS-GMM(diag)':
                # meanshift_qs = params[ 'meanshift_qs']  # meanshift uses the same kjl_qs, and only needs to tune one of them
                del params['meanshift_qs']
                if is_parallel:
                    with parallel:
                        outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                                   copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                                   copy.deepcopy(params),
                                                                   nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                                                   nystrom_q=nystrom_q)
                                        for nystrom_n, nystrom_d, nystrom_q in
                                        list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs)))
                else:
                    outs = []
                    for nystrom_n, nystrom_d, nystrom_q in list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs)):
                        out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                copy.deepcopy(params),
                                                nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                                nystrom_q=nystrom_q)
                        outs.append(out)

        else:
            msg = params['model_name']
            raise NotImplementedError(f'Error: {msg}')

    elif 'OCSVM' in params['model_name']:  # OCSVM
        ################################################################################################################
        # 2.2 OCSVM
        if params['model_name'] == 'OCSVM(rbf)':  # OCSVM(rbf)
            model_qs = params['OCSVM_qs']
            model_nus = params['OCSVM_nus']
            del params['OCSVM_qs']
            del params['OCSVM_nus']
            if is_parallel:
                with parallel:
                    outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                               copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                               copy.deepcopy(params),
                                                               OCSVM_q=OCSVM_q, OCSVM_nu=OCSVM_nu) for
                                    OCSVM_q, OCSVM_nu in list(itertools.product(model_qs, model_nus)))
            else:
                outs = []
                for OCSVM_q, OCSVM_nu in list(itertools.product(model_qs, model_nus)):
                    out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                            copy.deepcopy(X_val), copy.deepcopy(y_val),
                                            copy.deepcopy(params),
                                            OCSVM_q=OCSVM_q, OCSVM_nu=OCSVM_nu)
                    outs.append(out)

        elif params['model_name'] == 'KJL-OCSVM(linear)':
            kjl_ds = params['kjl_ds']
            kjl_ns = params['kjl_ns']
            kjl_qs = params['kjl_qs']
            del params['kjl_ds']
            del params['kjl_ns']
            del params['kjl_qs']
            model_nus = params['OCSVM_nus']
            del params['OCSVM_nus']
            if is_parallel:
                with parallel:
                    outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                               copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                               copy.deepcopy(params),
                                                               kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                                               OCSVM_nu=OCSVM_nu) for
                                    kjl_d, kjl_n, kjl_q, OCSVM_nu in
                                    list(itertools.product(kjl_ds, kjl_ns, kjl_qs, model_nus)))
            else:
                outs = []
                for kjl_d, kjl_n, kjl_q, OCSVM_nu in list(itertools.product(kjl_ds, kjl_ns, kjl_qs, model_nus)):
                    out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                            copy.deepcopy(X_val), copy.deepcopy(y_val),
                                            copy.deepcopy(params),
                                            kjl_d=kjl_d, kjl_n=kjl_n, kjl_q=kjl_q,
                                            OCSVM_nu=OCSVM_nu)
                    outs.append(out)

        elif params['model_name'] == 'Nystrom-OCSVM(linear)':
            nystrom_ns = params['nystrom_ns']
            nystrom_ds = params['nystrom_ds']
            nystrom_qs = params['nystrom_qs']
            del params['nystrom_ns']
            del params['nystrom_ds']
            del params['nystrom_qs']
            model_nus = params['OCSVM_nus']
            del params['OCSVM_nus']
            if is_parallel:
                with parallel:
                    outs = parallel(delayed(_model_train_test)(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                                               copy.deepcopy(X_val), copy.deepcopy(y_val),
                                                               copy.deepcopy(params),
                                                               nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                                               nystrom_q=nystrom_q,
                                                               OCSVM_nu=OCSVM_nu) for
                                    nystrom_n, nystrom_d, nystrom_q, OCSVM_nu in
                                    list(itertools.product(nystrom_ns, nystrom_ds, nystrom_qs, model_nus)))
            else:
                outs = []
                for nystrom_n, nystrom_d, nystrom_q, OCSVM_nu in list(itertools.product(
                        nystrom_ns, nystrom_ds, nystrom_qs, model_nus)):
                    out = _model_train_test(copy.deepcopy(X_train), copy.deepcopy(y_train),
                                            copy.deepcopy(X_val), copy.deepcopy(y_val),
                                            copy.deepcopy(params),
                                            nystrom_n=nystrom_n, nystrom_d=nystrom_d,
                                            nystrom_q=nystrom_q,
                                            OCSVM_nu=OCSVM_nu)
                    outs.append(out)
        else:
            msg = params['model_name']
            raise NotImplementedError(f'Error: {msg}')

    else:
        msg = params['model_name']
        raise NotImplementedError(f'Error: {msg}')

    ################################################################################################################
    # 3. get the best auc
    best_avg_auc = -1
    lg.debug(outs)
    for out in outs:
        if out['auc'] > best_avg_auc:
            best_avg_auc = out['auc']
            best_results = copy.deepcopy(out)


    ################################################################################################################
    # it's better to save all middle results too
    middle_results = outs
    time.sleep(10)   # sleep 5 seconds

    ################################################################################################################
    # 4. Build a new model with the best parameters
    lg.info('---Pick the best parameters from val set and then get accurate time of training and testing '
            'with the best params---')
    if 'qs_res' in best_results['params']:
        del best_results['params']['qs_res']

    out = _model_train_test(X_train, y_train, X_test, y_test, params=best_results['params'])
    best_results = out

    return copy.deepcopy(best_results), copy.deepcopy(middle_results)


def _get_single_result(model_cfg, data_cfg):
    """ Get result by the given model on the given dataset

    Parameters
    ----------
    model_cfg
    data_cfg

    Returns
    -------

    """
    ################################################################################################################
    # 1. Get local variables of the dataset and model
    data_name = data_cfg['data_name']
    data_file = data_cfg['data_file']
    (X, y) = data_cfg['data']
    feat_type = data_cfg['feat']

    model_name = model_cfg['model_name']
    train_size = model_cfg['train_size']
    is_gs = model_cfg['is_gs']  # Picking the best parameters from train set if is_gs is True.

    ################################################################################################################
    # 2. Get result (such as AUCs, times and spaces)
    # 2.1 Split data to test set and left set (it will be used to obtain train sets and val sets
    #                                           according to different random seeds)
    # 2.2. Run experiment n_repeats (here is 5) times
    #   a) For each train set and val set, build one model
    #   b) Evaluate each model on the same test set.
    if feat_type.upper().startswith('SAMP_'):
        ################################################################################################################
        # Get result with SAMP features. each SAMP feature has 10 sampling rates
        # only find the maximum one
        best_auc = -1
        lg.debug(f'****X: {X}')
        for q_samp_rate in X.keys():
            ###########################################################################################################
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
            params = []
            middle_results = []
            best_train_times = []
            best_test_times = []
            best_aucs = []
            best_space_sizes = []
            best_params = []
            best_middle_results = []
            for i in range(n_repeats):
                lg.debug(f"\n\n==={i + 1}/{n_repeats}(n_repeats): {model_cfg}===")
                X_train, y_train, X_val, y_val = split_train_val(X_normal, y_normal, X_abnormal, y_abnormal,
                                                                 train_size=train_size,
                                                                 val_size=int(len(y_test) * 0.25),
                                                                 random_state=(i + 1) * 100)
                _best_results_i, _middle_results_i = get_best_results(X_train, y_train, X_val, y_val,
                                                                      copy.deepcopy(X_test), copy.deepcopy(y_test),
                                                                      copy.deepcopy(model_cfg),
                                                                      random_state=random_state)
                middle_results.append(_middle_results_i)

                train_times.append(_best_results_i['train_time'])
                test_times.append(_best_results_i['test_time'])
                aucs.append(_best_results_i['auc'])
                space_sizes.append(_best_results_i['space_size'])
                params.append(_best_results_i['params'])

            # find the best average AUC from all q_samp_rate
            if np.mean(aucs) > best_auc:
                best_train_times = copy.deepcopy(train_times)
                best_test_times = copy.deepcopy(test_times)
                best_aucs = copy.deepcopy(aucs)
                best_space_sizes = copy.deepcopy(space_sizes)
                best_params = copy.deepcopy(params)
                best_middle_results = copy.deepcopy(middle_results)

        if is_gs:
            lg.debug(f'--is_gs: {is_gs}, X_val != X_test')
        else:
            X_val = X_test
        best_results_ = {'train_times': best_train_times, 'test_times': best_test_times, 'aucs': best_aucs,
                         'params': best_params,
                         'space_sizes': best_space_sizes,
                         'X_train_shape': X_train.shape, 'X_val_shape': X_val.shape, 'X_test_shape': X_test.shape}

        result = ((f'{data_name}|{data_file}', model_name), (best_results_, best_middle_results))
        # # dump each result to disk to avoid runtime error in parallel context.
        # dump_data(result, out_file=(f'{os.path.dirname(data_file)}/gs_{is_gs}-{GMM_covariance_type}/{case}.dat'))

    else:
        ################################################################################################################
        # Get result on IAT_SIZE or STATs feature
        n_repeats = model_cfg['n_repeats']
        random_state = model_cfg['random_state']

        ################################################################################################################
        # 2.1 Get test set and left set
        X_normal, y_normal, X_abnormal, y_abnormal = seperate_normal_abnormal(X, y, random_state=random_state)
        # get the unique test set
        X_normal, y_normal, X_abnormal, y_abnormal, X_test, y_test = split_left_test(X_normal, y_normal, X_abnormal,
                                                                                     y_abnormal, test_size=600,
                                                                                     random_state=random_state)

        ################################################################################################################
        # 2.2 Run experiment n_repeats (here is 5) times

        # Store the results
        train_times = []
        test_times = []
        aucs = []
        space_sizes = []
        params = []
        middle_results = []  # store all the middle results
        out_dir = pth.join(model_cfg['out_dir'],
                           data_cfg['direction'],
                           data_cfg['feat'] + "-header_" + str(data_cfg['is_header']),
                           data_cfg['data_name'],
                           "before_proj_" + str(model_cfg['before_proj']) + \
                           "-gs_" + str(model_cfg['is_gs']),
                           model_cfg['model_name'] + "-std_" + str(model_cfg['is_std'])
                           + "_center_" + str(model_cfg['is_std_mean']) + "-d_" + str(model_cfg['kjl_d']) \
                           + "-" + str(model_cfg['GMM_covariance_type']))

        # Run 5 times
        for i in range(n_repeats):
            lg.debug(f"\n\n==={i + 1}/{n_repeats}(n_repeats): {model_cfg}===")
            ###########################################################################################################
            # a) According to the different random seed (here is random_state), it will generate different
            # train set and val set
            X_train, y_train, X_val, y_val = split_train_val(X_normal, y_normal, X_abnormal, y_abnormal,
                                                             train_size=train_size, val_size=int(len(y_test) * 0.25),
                                                             random_state=(i + 1) * 100)
            ###########################################################################################################
            # b) Build the model on the train set, and get the final result on the same test set.
            _best_results_i, _middle_results_i = get_best_results(X_train, y_train, X_val, y_val,
                                                                  copy.deepcopy(X_test), copy.deepcopy(y_test),
                                                                  copy.deepcopy(model_cfg),
                                                                  random_state=random_state)
            ###########################################################################################################
            # c) Append each result to lists
            middle_results.append(_middle_results_i)

            train_times.append(_best_results_i['train_time'])
            test_times.append(_best_results_i['test_time'])
            aucs.append(_best_results_i['auc'])
            space_sizes.append(_best_results_i['space_size'])
            params.append(_best_results_i['params'])

            ###########################################################################################################
            # d) Only save the best model and test set
            dump_data(_best_results_i['model'], out_file=pth.join(out_dir, f'repeat_{i}.model'))
            dump_data((X_test, y_test), out_file=pth.join(out_dir, f'Test_set-repeat_{i}.dat'))

        ################################################################################################################
        # 3. Store all the results to a dict and return it
        if is_gs:
            lg.debug(f'--is_gs: {is_gs}, X_val != X_test')
        else:
            X_val = X_test
        best_results = {'train_times': train_times, 'test_times': test_times, 'aucs': aucs,
                        'params': params,
                        'space_sizes': space_sizes,
                        'X_train_shape': X_train.shape, 'X_val_shape': X_val.shape, 'X_test_shape': X_test.shape}

        result = ((f'{data_name}|{data_file}', model_name), (best_results, middle_results))

        ################################################################################################################
        # 4. Dump each result to disk to avoid data loss (optional)
        dump_data(result, out_file=pth.join(out_dir, f'results.dat'))

    return result
