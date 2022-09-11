"""Kernel density estimation

"""
# Authors: kun.bj@outlook.com
#
# License: XXX
import cProfile
import time

import numpy as np
from func_timeout import FunctionTimedOut
from loguru import logger as lg
from pyod.models.base import BaseDetector
from scipy.spatial import distance
from sklearn.compose._column_transformer import _check_X
from sklearn.neighbors import KernelDensity
from sklearn.neighbors._kde import VALID_KERNELS

from kjl.models._base import BASE
from kjl.utils import pstats


class _KDE(KernelDensity, BaseDetector):

	def __init__(self, bandwidth=1.0, algorithm='auto',
	             kernel='gaussian', metric="euclidean", atol=0, rtol=0, contamination=0.1,
	             breadth_first=True, leaf_size=40, metric_params=None, random_state=42):
		"""Kernel density estimation (KDE)
		Parameters
		----------
		bandwidth : float
			The bandwidth of the kernel.

		algorithm : str
			The tree algorithm to use.  Valid options are
			['kd_tree'|'ball_tree'|'auto'].  Default is 'auto'.

		kernel : str
			The kernel to use.  Valid kernels are
			['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
			Default is 'gaussian'.

		metric : str
			The distance metric to use.

		atol : float
			The desired absolute tolerance of the result.  A larger tolerance will
			generally lead to faster execution. Default is 0.

		rtol : float
			The desired relative tolerance of the result.

		breadth_first : bool
			If true (default), use a breadth-first approach to the problem.
			Otherwise use a depth-first approach.

		leaf_size : int
			Specify the leaf size of the underlying tree.

		metric_params : dict
			Additional parameters to be passed to the tree for use with the
			metric.
		"""
		self.algorithm = algorithm
		self.bandwidth = bandwidth
		self.kernel = kernel
		self.metric = metric
		self.atol = atol
		self.rtol = rtol
		self.breadth_first = breadth_first
		self.leaf_size = leaf_size
		self.metric_params = metric_params
		self.contamination = contamination
		self.random_state = random_state

		# run the choose algorithm code so that exceptions will happen here
		# we're using clone() in the GenerativeBayes classifier,
		# so we can't do this kind of logic in __init__
		self._choose_algorithm(self.algorithm, self.metric)

		if bandwidth <= 0:
			raise ValueError("bandwidth must be positive")
		if kernel not in VALID_KERNELS:
			raise ValueError("invalid kernel: '{0}'".format(kernel))

	def fit(self, X_train, y_train=None):
		"""Fit KDE.

		Parameters
		----------
		X_train : numpy array of shape (n_samples, n_features)
			The input samples.

		y_train : numpy array of shape (n_samples,), optional (default=None)
			The ground truth of the input samples (labels).

		Returns
		-------
		self : object
			the fitted estimator.
		"""
		X_train = _check_X(X_train)
		self.model = KernelDensity(bandwidth=self.bandwidth,
		                           algorithm=self.algorithm,
		                           kernel=self.kernel,
		                           metric=self.metric,
		                           atol=self.atol,
		                           rtol=self.rtol,
		                           breadth_first=self.breadth_first,
		                           leaf_size=self.leaf_size,
		                           metric_params=self.metric_params)

		self.model.fit(X_train)
		return self

	def decision_function(self, X):
		"""Predict raw anomaly scores of X using the fitted detector.
		After invert_order(): the higher score, the more probability of x that is predicted as abnormal

		Parameters
		----------
		X : numpy array of shape (n_samples, n_features)
			The input samples. Sparse matrices are accepted only
			if they are supported by the base estimator.

		Returns
		-------
		anomaly_scores : numpy array of shape (n_samples,)
			The anomaly score of the input samples.
		"""
		# check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
		return -1 * self.model.score_samples(X)

	def predict_proba(self, X):
		return -1 * self.score_samples(X)  #


class KDE(BASE, _KDE):

	def __init__(self, params):
		self.params = params
		self.random_state = params['random_state']

	def fit(self, X_train, y_train=None):

		self.train_time = 0
		N, D = X_train.shape

		#####################################################################################################
		# 1.1 normalization
		# pr = cProfile.Profile(time.perf_counter)
		# pr.enable()
		# # if self.params['is_std']:
		# #     self.scaler = StandardScaler(with_mean=self.params['is_std_mean'])
		# #     self.scaler.fit(X_train)
		# #     X_train = self.scaler.transform(X_train)
		# #     # if self.verbose > 10: data_info(X_train, name='X_train')
		# # else:
		# #     pass
		# pr.disable()
		# ps = pstats.Stats(pr).sort_stats('line')  # cumulative
		# ps.print_stats()
		# self.std_train_time = ps.total_tt
		self.std_train_time = 0
		self.train_time += self.std_train_time
		self.seek_train_time = 0

		q = self.params['kde_q']
		# find the best parameters of the detector
		distances = distance.pdist(X_train, metric='euclidean')
		sigma = np.quantile(distances, q=q)
		if sigma == 0:  # find a new non-zero sigma
			lg.debug(f'sigma: {sigma}, q: {q}')
			q_lst = list(np.linspace(q + 0.01, 1, 10, endpoint=False))
			sigma_lst = np.quantile(distances, q=q_lst)
			sigma, q = [(s_v, q_v) for (s_v, q_v) in zip(sigma_lst, q_lst) if s_v > 0][0]
		lg.debug(f'q: {q}, sigma: {sigma}, {np.quantile(distances, q=[0, 0.3, 0.9, 1])}')

		self.bandwidth = sigma

		model = _KDE(bandwidth=self.bandwidth,
		             random_state=self.random_state)  # the rest of params uses the default values

		# 2.2 Train the models
		try:
			self.model, self.model_train_time = self._train(model, X_train)
		except (FunctionTimedOut, Exception) as e:
			lg.warning(f'{e}, retrain with a larger reg_covar')
			model.reg_covar = 1e-5
			self.model, self.model_train_time = self._train(model, X_train)
		self.train_time += self.model_train_time

		# 3. get space size
		pr = cProfile.Profile(time.perf_counter)
		pr.enable()
		self.space_size = N * D + 1  # 1 is bandwidth
		pr.disable()
		ps = pstats.Stats(pr).sort_stats('line')  # cumulative
		# ps.print_stats()
		self.space_train_time = ps.total_tt
		# self.train_time += self.space_train_time

		lg.info(f'Total train time: {self.train_time} <= std_train_time: {self.std_train_time}, '
		        f'model_train_time: {self.model_train_time}, '
		        f'space_size: {self.space_size}')

	def test(self, X_test, y_test, idx=None):
		return self._test(self.model, X_test, y_test, idx=idx)
