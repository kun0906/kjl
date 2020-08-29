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
from kjl.utils.data import split_train_test, load_data, extract_data, dump_data, save_result, batch
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
            params['n_components'] = [2]
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

            # Fit a model on the train set (how to get the best parameters of GMM on the init_set?)
            self._init_train(self.init_model, X_train, y_train)  # update self.init_model
            # Evaluate the model on the test set
            self._init_test(self.init_model, X_test, y_test)
            print(f'auc: {self.auc}')
            # for out in outs:
            if best_auc <= self.auc:
                best_auc = self.auc
                best_params = self.model.get_params()
                self.params_copy = copy.deepcopy(self.params)

        # get best model with best_params
        self.params = self.params_copy
        self.init_model = GMM()
        self.init_model.set_params(**best_params)

        # Fit a model on the train set (how to get the best parameters of GMM on the init_set?)
        self._init_train(self.init_model, X_train, y_train)  # update self.init_model
        # Evaluate the model on the test set
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
            self.kjl_inst = KJL(self.params)
            _, kjl_train_time = func_running_time(self.kjl_inst.fit, X_train)
            proj_train_time += kjl_train_time
            X_train, kjl_train_time = func_running_time(self.kjl_inst.transform, X_train)
            proj_train_time += kjl_train_time
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            self.nystrom_inst = NYSTROM()
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
        model.set_params(**model_params)
        if self.verbose > 5: print(model.get_params())
        # train model
        self.model, model_train_time = func_running_time(model.fit, X_train)
        self.train_time += model_train_time

        return self

    def _init_test(self, model, X_test, y_test):
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
        proj_test_time = 0.0
        if 'kjl' in self.params.keys() and self.params['kjl']:
            X_test, proj_test_time = func_running_time(self.kjl_inst.transform, X_test)

        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
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

    def train_test_model(self, X_train, y_train, X_test, y_test):
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
        # Step 1. Split train set into two subsets: initial set (init_set) and new arrival set (arvl_set)
        # with ratio 3:7.
        X_train, X_arrival, y_train, y_arrival = train_test_split(X_train, y_train, test_size=0.7,
                                                                  random_state=self.random_state)
        self.n_samples, self.n_feats = X_train.shape

        ##########################################################################################
        # # Step 2. Get initial model (init_model) on initial set (init_set) and evaluate it on test set
        self.init_model = self.get_best_model(X_train, y_train, X_test, y_test)

        ##########################################################################################
        # Step 3. Online train and evaluate model
        online_train_times = []
        online_test_times = []
        online_aucs = []
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=1000)):
            if i == 0:
                self.model = self.init_model

            # online train model (update GMM model values, such as, means, covariances, kjl_U, and n_components)
            self._online_train(X_batch, Y_batch)  # update self.model
            online_train_times.append(self.train_time)

            # online test model
            self._online_test(X_test, y_test)
            online_test_times.append(self.test_time)
            online_aucs.append(self.auc)

            if self.verbose > 5:
                print(f'***batch: {i + 1}, train_time: {self.train_time}, test_time: {self.test_time}, auc: {self.auc}')

        self.info['train_times'] = online_train_times
        self.info['test_times'] = online_test_times
        self.info['aucs'] = online_aucs
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
        X_batch_copy = copy.deepcopy(X_batch)

        ##########################################################################################
        # Use the model to predict X_batch first, and according to the result,
        # only the normal data will be used to train a new model instead of the current one.

        ### Step 1: Preprocessing
        # Step 1.1: standardization
        # transform x
        X_batch_std, std_transform_time = func_running_time(self.std_inst.transform, X_batch)
        model_train_time += std_transform_time

        # Step 1.2: Seek the modes by quickshift++ or meanshift
        seek_train_time = 0
        model_train_time += seek_train_time

        # Step 3: Project the data onto a lower space with kjl or nystrom
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
        # if self.verbose > 5:
        #     print(f'Total test time: {self.test_time} <= std_test_time: {self.std_test_time}, '
        #           f'seek_test_time: {self.seek_test_time}'
        #     f', proj_test_time: {self.proj_test_time}, '
        #     f'model_test_time: {self.model_test_time}')

        for i, (x, x_std, x_proj, y) in enumerate(zip(X_batch_copy, X_batch_std, X_batch_proj, y_score)):
            ##########################################################################################
            # Step 3. online train a GMM based on the previous GMM (such as means, covariances)
            # and use it to replace the previous one.
            if y < self.abnormal_thres:
                new_model = ONLINE_GMM(n_components=self.model.n_components, weights_init=self.model.weights_,
                                       means_init=self.model.means_, covariances_init=self.model.covariances_)
                new_model.warm_start = True
                new_model.n_samples = self.model.n_samples
                # online update the model

                # According to the _y_score, the x is predicted as a normal datapoint.
                new_model.n_samples += 1
                # two sub-scenarios:
                # 1): just update the previous components
                # 2): create a new component for the new x
                if y < self.novelty_thres:
                    create_new_component = False
                else:
                    create_new_component = True
                self._online_train_update(new_model, x, x_std, x_proj, create_new_component, to_convergent=True)

            else:
                # if _y_score >= self.abnormal_thres, the x is predicted as a abnormal flow, so we should drop it.
                print('this flow is an abnormal flow, so we drop it.')

        end_0 = datetime.now()
        model_online_train_time = (end_0 - start_0).total_seconds()
        print(f'Total batch time: {model_online_train_time}')

        return self

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
            X_tesX_test, proj_test_time = func_running_time(self.nystrom_inst.transform, X_test)
        else:
            proj_test_time = 0
        self.test_time += proj_test_time

        ##########################################################################################
        # Step 2: evaluate GMM on the test set
        print(X_test)
        y_score, model_test_time = func_running_time(self.model.decision_function, X_test)
        self.auc, _ = func_running_time(self.get_score, y_test, y_score)
        self.test_time += model_test_time
        print(
            f'Total test time: {self.test_time} <= std_test_time: {std_test_time},'
            f', proj_test_time: {proj_test_time}, '
            f'model_test_time: {model_test_time}')

        return self

    def _online_train_update(self, new_model, x, x_std, x_proj, create_new_component=False, to_convergent=False):
        """Online train and update model

        Parameters
        ----------
        new_model: model instance
            Use the previous model's values (such as means, covariances, and weights) to initialize new_model

        x:
            the original data

        x_std:
            after standardization

        x_proj:
            after projection

        create_new_component: bool
            If create a new component or not

        to_convergent: bool
            If training the new model until convergence or not

        Returns
        -------
            self

        """

        start_0 = datetime.now()

        ##########################################################################################
        # Step 1: online train GMM
        # two sub-scenarios:
        # 1): just update the previous components
        # 2): create a new component for the new x

        lower_bound = -np.infty
        x = x.reshape(1, -1)
        x_std = x_std.reshape(1, -1)
        x_proj = x_proj.reshape(1, -1)

        if not create_new_component:
            ##########################################################################################
            # sub-scenario 1.1:  just update the previous components
            # the x is predicted as a normal point, and  we just need to update the previous components
            if not to_convergent:
                # get log_prob and resp
                log_prob_norm, log_resp = new_model._e_step(x_proj)
                new_model.log_resp = log_resp
                # use m_step to update params: weights (i.e., mixing coefficients), means, and covariances with x and
                # the previous params: log_resp (the log probability of each component), means and covariances
                new_model._m_step(x_proj, new_model.log_resp,
                                  new_model.n_samples - 1)  # update mean, covariance and weight

            else:
                for n_iter in range(1, new_model.max_iter + 1):
                    prev_lower_bound = lower_bound

                    # get log_prob and resp
                    log_prob_norm, log_resp = new_model._e_step(x_proj)
                    new_model.log_resp = log_resp
                    print(f'n_iter: {n_iter}, log_resp: {log_resp}, x_proj: {x_proj}')

                    # use m_step to update params: weights (i.e., mixing coefficients), means, and covariances with x and
                    # the previous params: log_resp (the log probability of each component), means and covariances
                    new_model._m_step(x_proj, new_model.log_resp,
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
            ##########################################################################################
            # sub-scenario 1.2:  create a new component for the new x
            # self.novelty_thres < _y_score < self.abnormal_thres
            # x is predicted as a novelty datapoint (but still is a normal datapoint), so we create a new
            # component and update GMM.

            new_model.n_components += 1
            # # assign the x to a new class and expand the previous "log_resp", which is used to obtain
            # # the "weights" of each component.
            # new_model.log_resp = np.concatenate([model.log_resp, np.zeros((model.n_samples, 1))], axis=1)
            log_resp = np.zeros((1, new_model.n_components))
            log_resp[-1] = 1
            # new_model.log_resp = np.concatenate([new_model.log_resp, log_resp], axis=0)
            new_model.log_resp = log_resp
            new_model.weights_ = np.concatenate([new_model.weights_, np.zeros((1,)) + 1e-6], axis=0)

            # compute the mean and covariance of the new components
            # For the mean, we use the x value as the mean of the new component
            # (because the new component only has one point (i.e., x)), and append it to the previous means.
            new_mean = x_proj
            new_covar = self.generate_new_covariance(x_proj, new_model.means_, new_model.covariances_)
            new_model.means_ = np.concatenate([new_model.means_, new_mean], axis=0)
            _, dim = new_mean.shape
            new_model.covariances_ = np.concatenate([new_model.covariances_,
                                                     new_covar.reshape(1, dim, dim)], axis=0)

            print(f'new_model.params: {new_model.get_params()}')

            if not to_convergent:
                pass
            else:
                # train the new model on x, update params, and use the new model to update the previous model
                for n_iter in range(1, new_model.max_iter + 1):
                    # for n_iter in range(1, self.max_iter ):
                    # convergence conditions: check if self.max_iter or self.tol exceeds the preset value.
                    prev_lower_bound = lower_bound

                    # use m_step to update params: weights, means, and covariances with x and the initialized
                    # params: log_resp, means and covariances
                    new_model._m_step(x_proj, new_model.log_resp,
                                      new_model.n_samples - 1)  # update weight, means, covariances

                    # get log_prob and resp
                    log_prob_norm, log_resp = new_model._e_step(x_proj)
                    new_model.log_resp = log_resp

                    # get the difference
                    lower_bound = new_model._compute_lower_bound(log_resp, log_prob_norm)
                    change = lower_bound - prev_lower_bound
                    if abs(change) < new_model.tol:
                        self.converged_ = True
                        print(f'n_iter: {n_iter}')
                        break

        ##########################################################################################
        # Step 2: Update model, std, and projection
        # Step 2.1: update the current model with the new_model
        self.model = new_model

        # Step 2.2: update std
        # update the mean and scaler of self.std_inst with 'x'
        std_update_time = func_running_time(self.std_inst.update, x)  # use the original X_batch to update std_inst

        # Step 2.3: update projection
        if 'kjl' in self.params.keys() and self.params['kjl']:
            # update kjl or nystrom
            #  Update kjl: self.U_kjl, self.Xrow_kjl.
            # use the X_batch without projection to update kjl_inst
            proj_update_time = func_running_time(self.kjl_inst.update, x_std)
        elif 'nystrom' in self.params.keys() and self.params['nystrom']:
            # Update nystrom_inst
            proj_update_time = func_running_time(self.nystrom_inst.update, x_proj)

        else:
            proj_train_time = 0

        end_0 = datetime.now()
        self.train_time = (end_0 - start_0).total_seconds()

        return self

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
        # Step 1. Split train set into two subsets: initial set (init_set) and new arrival set (arvl_set)
        # with ratio 3:7.
        X_train, X_arrival, y_train, y_arrival = train_test_split(X_train, y_train, test_size=0.7,
                                                                  random_state=self.random_state)
        self.n_samples, self.n_feats = X_train.shape

        ##########################################################################################
        # # Step 2. Get initial model (init_model) on initial set (init_set) and evaluate it on test set
        self.init_model = self.get_best_model(X_train, y_train, X_test, y_test)

        ##########################################################################################
        # Step 3. train the model on the batch data (previous+batch) and evaluate it.
        batch_train_times = []
        batch_test_times = []
        batch_aucs = []
        for i, (X_batch, Y_batch) in enumerate(batch(X_arrival, y_arrival, step=1000)):
            if i == 0:
                self.model = self.init_model

            self._batch_train(X_batch, Y_batch)
            batch_train_times.append(self.train_time)

            # batch test model
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

        # Step 3.2: train a new model and use it to instead of the current one.
        # we use model.n_compoents to initialize n_components or the value found by quickhsift++ or meanshift
        new_model = GMM(n_components=self.model.n_components)
        self._init_train(new_model, self.init_X_train, self.init_y_train)  # self.train_time

        # the datapoint is predicted as a abnormal flow, so we should drop it.
        print(f'{abnormal_cnt} flows are predicted as abnormal, so we drop them.')

        end_0 = datetime.now()
        model_batch_train_time = (end_0 - start_0).total_seconds()
        print(f'Total batch time: {model_batch_train_time}')

        # replace the current model with new_model
        self.model = new_model

        return self

    def _batch_test(self, X_batch, y_batch):
        self._init_test(self.model, X_batch, y_batch)


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

                    line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs:{aucs_str}, train_times:' \
                           f'{train_times_str}, test_times:{test_times_str}, with params: {params}: {_suffex}'

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
                _best_results = model.info  # model.info['train_times']
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
