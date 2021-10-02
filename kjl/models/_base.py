"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import cProfile
import time

import numpy as np
# from func_timeout import func_set_timeout, FunctionTimedOut
from func_timeout import func_set_timeout
from loguru import logger as lg

from kjl.utils import pstats

FUNC_TIMEOUT = 5 * 60  # (if function takes more than 10 mins, then it will be killed)
np.set_printoptions(precision=2, suppress=True)

from sklearn import metrics
from sklearn.metrics import roc_curve

from kjl.utils.tool import dump


class BASE:

	def __init__(self, random_state=42):
		self.random_state = random_state

	# # use_signals=False: the fuction cannot return a object that cannot be pickled (here "self" is not pickled,
	# # so it will be PicklingError)
	# # use_signals=True: it only works on main thread (here train_test_intf is not the main thread)
	# @timeout_decorator.timeout(seconds=30 * 60, use_signals=False, timeout_exception=StopIteration)
	# @profile
	# func_timeout will run the specified function in a thread with the specified arguments until it returns.
	@func_set_timeout(FUNC_TIMEOUT)  # seconds
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
		pr = cProfile.Profile(time.perf_counter)
		pr.enable()
		try:
			model.fit(X_train)
		except (TimeoutError, Exception) as e:
			msg = f'fit error: {e}'
			raise ValueError(f'{msg}')
		pr.disable()
		ps = pstats.Stats(pr).sort_stats('line')  # cumulative
		# ps.print_stats()
		train_time = ps.total_tt
		# lg.debug("Fitting models takes {} seconds".format(train_time))

		return model, train_time

	def _test(self, model, X_test, y_test, idx=None):
		"""Evaluate the models on the X_test, y_test

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
		# lg.info(X_test[0, :])
		self.test_time = 0

		#####################################################################################################
		# 1. standardization
		# pr = cProfile.Profile(time.perf_counter)
		# pr.enable()
		# # if self.params['is_std']:
		# #     X_test = self.scaler.transform(X_test)
		# # else:
		# #     pass
		# pr.disable()
		# ps = pstats.Stats(pr).sort_stats('line')  # cumulative
		# ps.print_stats()
		# self.std_test_time = ps.total_tt
		self.std_test_time = 0
		self.test_time += self.std_test_time

		#####################################################################################################
		# 2. projection
		pr = cProfile.Profile(time.perf_counter)
		pr.enable()
		if 'is_kjl' in self.params.keys() and self.params['is_kjl']:
			X_test = self.project.transform(X_test)
		elif 'is_nystrom' in self.params.keys() and self.params['is_nystrom']:
			X_test = self.project.transform(X_test)
		else:
			pass
		pr.disable()
		ps = pstats.Stats(pr).sort_stats('line')  # cumulative
		# ps.print_stats()
		self.proj_test_time = ps.total_tt
		self.test_time += self.proj_test_time
		# lg.info(X_test[0, :])
		# no need to do seek in the testing phase
		self.seek_test_time = 0

		#####################################################################################################
		# 3. prediction
		pr = cProfile.Profile(time.perf_counter)
		pr.enable()
		# For inlier, a small value is used; a larger value is for outlier (positive)
		# it must be abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
		y_score = model.decision_function(X_test)
		pr.disable()
		ps = pstats.Stats(pr).sort_stats('line')  # cumulative
		# ps.print_stats()
		self.model_test_time = ps.total_tt
		self.test_time += self.model_test_time
		# lg.debug(f'y_score: {y_score}, y_test: {y_test}')

		# For binary  y_true, y_score is supposed to be the score of the class with greater label.
		# auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
		# pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
		fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
		self.auc = metrics.auc(fpr, tpr)
		lg.info(f"AUC: {self.auc}")

		lg.info(f'Test time: {self.test_time} <= std_test_time: {self.std_test_time}, '
		        f'seek_test_time: {self.seek_test_time}, proj_test_time: {self.proj_test_time}, '
		        f'model_test_time: {self.model_test_time}, idx={idx}')

		# return self.auc, self.test_time
		res = {'score': self.auc, 'auc': self.auc, 'test_time': self.test_time}
		return res

	def save(self, data, out_file='.dat'):
		dump(data, name=out_file)
