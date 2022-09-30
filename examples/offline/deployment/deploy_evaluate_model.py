""" Deploy the built models to different servers and then evaluate their performance.

	Main steps:
		1. Deployment: only upload the needed parameters of each model to the servers
		2. Reconstruct new models according to the parameters
		3. Evaluate each new model on the test set


	Command:
	current directory is project_root_dir (i.e., kjl/.)
	PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/deployment/deploy_evaluate_model.py > deploy.txt 2>&1 &
"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import cProfile
import copy
import itertools
import os.path
import shutil
import time
import traceback
from collections import Counter

from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.metrics import roc_curve

from examples.offline._constants import *
from examples.offline._gather import gather
from examples.offline.offline import save_dict2txt
from kjl.models.gmm import _GMM
from kjl.models.kde import _KDE
from kjl.models.ocsvm import _OCSVM
from kjl.projections.kjl import KJL
from kjl.projections.nystrom import Nystrom
from kjl.utils import pstats
from kjl.utils.tool import load, dump, check_path, timer

RESULT_DIR = f'results/{START_TIME}'
# DATASETS = ['MAWI1_2020']  # Two different normal data
# MODELS = ["KJL-OCSVM(linear)"]
FEATURES = ['IAT+SIZE']
HEADERS = [False]
TUNINGS = [False, True]
n_test_repeats = 100
print(f'RESULT_DIR: {RESULT_DIR}')
out_dir = 'examples/offline/deployment/out/src_dst'
if os.path.exists(out_dir):
	shutil.rmtree(out_dir, ignore_errors=True)

MODELS = [
    ################################################################################################################
    # # 1. OCSVM
    "OCSVM(rbf)",
    # "KJL-OCSVM(linear)",
    # "Nystrom-OCSVM(linear)",

    "KDE",

    # ################################################################################################################
    "GMM(full)",
	# "GMM(diag)",

    # # # ################################################################################################################s
    # # # # 2. KJL/Nystrom
    # "KJL-GMM(full)", # "KJL-GMM(diag)",
    # "Nystrom-GMM(full)", # "Nystrom-GMM(diag)",
	#
    # ################################################################################################################
    # # quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    # # "QS-KJL-GMM(full)", "QS-KJL-GMM(diag)",
    # # "MS-KJL-GMM(full)", "MS-KJL-GMM(diag)",
	#
    # # "QS-Nystrom-GMM(full)", "QS-Nystrom-GMM(diag)",
    # # "MS-Nystrom-GMM(full)", "MS-Nystrom-GMM(diag)",
	#
    # ################################################################################################################
    # # 3. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    # "KJL-QS-GMM(full)", #  "KJL-QS-GMM(diag)",
    # # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)",
	#
    # "Nystrom-QS-GMM(full)",  #  "Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"
    # #
    # # ################################################################################################################
    # # # 4. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection and initialize GMM (set 'GMM_is_init_all'=True)
    # # "KJL-QS-init_GMM(full)",   "KJL-QS-init_GMM(diag)",
    # # "Nystrom-QS-init_GMM(full)",   "Nystrom-QS-init_GMM(diag)",
]


def _get_parameters(model_all_data):
	model = model_all_data['model']['model']
	data = model_all_data['model']['data']  # X_train, y_train, x_val, y_val
	args = model_all_data['model']['args']
	extra_params = {'train': {'train_time': model.train_time}}  # for paper to compute the training speedup

	model_name = args.model

	model_params = {}
	project_params = {}
	if 'OCSVM' in model_name:
		model_params['kernel'] = model.kernel
		if model_params['kernel'] == 'rbf':
			model_params['_gamma'] = model.model._gamma
		else:
			# model_params['_gamma'] = 0
			msg = f'Error: {model.kernel}'
			raise NotImplementedError(msg)

		model_params['support_vectors_'] = model.model.support_vectors_
		model_params['_dual_coef_'] = model.model._dual_coef_
		model_params['_intercept_'] = model.model._intercept_

		# other parameters
		model_params['_sparse'] = model.model._sparse
		model_params['shape_fit_'] = model.model.shape_fit_
		model_params['_n_support'] = model.model._n_support
		# model_params['support_'] = models.models.support_
		# model_params['probA_'] = models.models.probA_
		# model_params['probB_'] = models.models.probB_
		# model_params['offset_'] = models.models.offset_

		if 'KJL' in model_name:  # KJL-OCSVM(linear)
			project_params['sigma'] = model.project.sigma
			project_params['Xrow'] = model.project.Xrow
			project_params['U'] = model.project.U
		elif 'Nystrom' in model_name:  # Nystrom-OCSVM(linear)
			project_params['sigma'] = model.project.sigma
			project_params['Xrow'] = model.project.Xrow
			project_params['eigvec_lambda'] = model.project.eigvec_lambda

	elif 'GMM' in model_name:
		# GMM params
		model_params['covariance_type'] = model.model.covariance_type
		model_params['weights_'] = model.model.weights_
		model_params['means_'] = model.model.means_
		# model_params['precisions_'] = models.models.precisions_
		model_params['precisions_cholesky_'] = model.model.precisions_cholesky_

		if 'KJL' in model_name:  # KJL-GMM
			project_params['sigma'] = model.project.sigma
			project_params['Xrow'] = model.project.Xrow
			project_params['U'] = model.project.U
		elif 'Nystrom' in model_name:  # Nystrom-GMM
			project_params['sigma'] = model.project.sigma
			project_params['Xrow'] = model.project.Xrow
			project_params['eigvec_lambda'] = model.project.eigvec_lambda

	elif 'KDE' in model_name:
		# KDE params
		# model_params['kernel'] = model.kernel
		# if model_params['kernel'] == 'rbf':
		# 	model_params['_gamma'] = model.model._gamma
		# else:
		# 	# model_params['_gamma'] = 0
		# 	msg = f'Error: {model.kernel}'
		# 	raise NotImplementedError(msg)
		model_params['kernel'] = model.model.kernel
		model_params['bandwidth'] = model.bandwidth
		# model_params['_gamma'] = model.model._gamma
		# have no answer until now.
		# https://github.com/pulimeng/eToxPred/issues/1
		# https://github.com/scikit-learn/scikit-learn/issues/19602
		model_params['tree_'] = model.model.tree_   # this one will have issues on RSPI(32bit): require SIZE_t, but long long for kde.
		# if 'KJL' in model_name:  # KJL-GMM
		# 	project_params['sigma'] = model.project.sigma
		# 	project_params['Xrow'] = model.project.Xrow
		# 	project_params['U'] = model.project.U
		# elif 'Nystrom' in model_name:  # Nystrom-GMM
		# 	project_params['sigma'] = model.project.sigma
		# 	project_params['Xrow'] = model.project.Xrow
		# 	project_params['eigvec_lambda'] = model.project.eigvec_lambda

	else:
		raise NotImplementedError()

	return {'model_name': model_name, 'model': model_params, 'project': project_params, 'extra_params': extra_params}


def get_space(model_params, unit='KB'):
	""" Return the size, in bytes, of path.

	Parameters
	----------
	model_params_file
	project_params_file

	Returns
	-------

	"""
	tmp_file = '~tmp.dat'
	dump(model_params, tmp_file)
	space = os.path.getsize(tmp_file)
	# space2 = sys.getsizeof(model_params)  # doesn't work
	# print(f'space: {space}, space2: {space2}')
	if unit == 'KB':
		space /= 1e+3
	elif unit == 'MB':
		space /= 1e+6
	else:
		pass

	return space


def extract_needed_parameters(in_dir='out/src_dst', out_dir='deployment/models', n_repeats=5):
	for dataset, feature, header, model, tuning in list(
			itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS)):
		try:
			lg.info(f'*** {dataset}-{feature}-header_{header}, {model}-tuning_{tuning}')
			for i in range(n_repeats):
				f = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
				                 f'model_{i}th.dat')
				params = load(f)
				needed_params = _get_parameters(params)
				needed_params['space'] = get_space(needed_params)
				model_params_file = os.path.join(out_dir, dataset, feature, f'header_{header}', model,
				                                 f'tuning_{tuning}',
				                                 f'model_params_{i}th.dat')
				check_path(model_params_file)
				dump(needed_params, model_params_file)

				if i == 0:  # only need to extract the test set from one repeat (because all the test set all the same)
					# save test set to disk
					f = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
					                 f'repeat_{i}th.dat')
					params = load(f)
					out_file = os.path.join(out_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
					                        f'test_set.dat')
					test = params['test']
					dump((test['X_test'], test['y_test']), out_file)
					lg.debug(f'test_set: {out_file}')
		except Exception as e:
			lg.error(f'Error: {e}. [{dataset}, {feature}, {header}, {model}, {tuning}]')
			traceback.print_exc()


def reconstruct_model(params):
	model_name = params['model_name']
	model_params = params['model']
	project_params = params['project']
	#######################################################################################################
	# 1. recreate project object from saved parameters
	extra_params = {'is_kjl': False, 'is_nystrom': False}  # used for testing
	if 'KJL' in model_name:  # KJL-OCSVM
		project = KJL({'kjl_d': 0, 'kjl_n': 0, 'kjl_q': 0, 'random_state': 0})
		project.sigma = project_params['sigma']
		project.Xrow = project_params['Xrow']
		project.U = project_params['U']
		extra_params['is_kjl'] = True
	elif 'Nystrom' in model_name:  # Nystrom-OCSVM
		project = Nystrom({'nystrom_d': 0, 'nystrom_n': 0, 'nystrom_q': 0, 'random_state': 0})
		project.sigma = project_params['sigma']
		project.Xrow = project_params['Xrow']
		project.eigvec_lambda = project_params['eigvec_lambda']
		extra_params['is_nystrom'] = True
	else:
		project = None

	#######################################################################################################
	# 2. recreate a new models from saved parameters
	if 'OCSVM' in model_name:
		model = _OCSVM()
		model.kernel = model_params['kernel']
		model.gamma = model_params['_gamma']  # only used for 'rbf', 'linear' doeesn't need '_gamma'
		model.support_vectors_ = model_params['support_vectors_']
		model.dual_coef_ = model_params['_dual_coef_']  # Coefficients of the support vectors in the decision function.
		model.intercept_ = model_params['_intercept_']
	elif 'GMM' in model_name:
		model = _GMM()
		model.covariance_type = model_params['covariance_type']
		model.weights_ = model_params['weights_']
		model.means_ = model_params['means_']
		# models.precisions_ = model_params['precisions_']
		model.precisions_cholesky_ = model_params['precisions_cholesky_']
	elif 'KDE' in model_name:
		model = _KDE()
		model.bandwidth = model_params['bandwidth']
		model.kernel = model_params['kernel']
		model.tree_ = model_params['tree_']
	else:
		raise NotImplementedError()

	return {'model': model, 'project': project, 'extra_params': extra_params, 'space': params['space']}


# @profile(precision=8)
def _test(model, X_test, y_test):
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
	model, project, params = model['model'], model['project'], model['extra_params']
	test_time = 0

	#####################################################################################################
	# 1. standardization
	# start = time.time()
	# # if self.params['is_std']:
	# #     X_test = self.scaler.transform(X_test)
	# # else:
	# #     pass
	# end = time.time()
	# self.std_test_time = end - start
	std_test_time = 0
	test_time += std_test_time

	#####################################################################################################
	# 2. projection
	pr = cProfile.Profile(time.perf_counter)
	pr.enable()
	if 'is_kjl' in params.keys() and params['is_kjl']:
		X_test = project.transform(X_test)
	elif 'is_nystrom' in params.keys() and params['is_nystrom']:
		X_test = project.transform(X_test)
	else:
		pass
	pr.disable()
	ps = pstats.Stats(pr).sort_stats('line')  # cumulative
	# ps.print_stats()
	proj_test_time = ps.total_tt
	test_time += proj_test_time

	# no need to do seek in the testing phase
	seek_test_time = 0

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
	model_test_time = ps.total_tt
	test_time += model_test_time

	# For binary  y_true, y_score is supposed to be the score of the class with greater label.
	# auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
	# pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
	fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
	auc = metrics.auc(fpr, tpr)
	lg.debug(f"AUC: {auc}")

	lg.info(f'Total test time: {test_time} <= std_test_time: {std_test_time}, '
	        f'seek_test_time: {seek_test_time}, proj_test_time: {proj_test_time}, '
	        f'model_test_time: {model_test_time}')

	return {'test_time': test_time, 'auc': auc}


def _evaluate(model, test_set, n_test_repeats=100, is_parallel=False):
	lg.debug(f'n_test_repeats: {n_test_repeats}, is_parallel: {is_parallel}')
	history = {}  # save all results
	X_test, y_test = test_set
	if is_parallel:
		with Parallel(n_jobs=10, verbose=0, backend='loky', pre_dispatch=1, batch_size=1) as parallel:
			outs = parallel(delayed(_test)(copy.deepcopy(model), copy.deepcopy(X_test), copy.deepcopy(y_test)) for _ in
			                range(n_test_repeats))
		test_times = []
		test_aucs = []
		for i, tmp_ in enumerate(outs):
			history[f'{i}_repeat'] = tmp_
			test_times.append(tmp_['test_time'])
			test_aucs.append(tmp_['auc'])
	else:
		test_times = []
		test_aucs = []
		for i in range(n_test_repeats):
			tmp_ = _test(copy.deepcopy(model), copy.deepcopy(X_test), copy.deepcopy(y_test))
			history[f'{i}_repeat'] = tmp_
			test_times.append(tmp_['test_time'])
			test_aucs.append(tmp_['auc'])

	test_res = {'test_time': np.mean(test_times), 'test_auc': np.mean(test_aucs),
	            'times': test_times, 'aucs': test_aucs,
	            'test_shape': X_test.shape, 'y': Counter(y_test),
	            'X_test': X_test, 'y_test': y_test}
	lg.debug(f'test_res: {test_res}')

	return test_res


def save_dict2txt(data, out_file, delimiter=','):
	""" Save result to txt

	Parameters
	----------
	data: dict

	out_file: path
	delimiter: ','

	Returns
	-------

	"""
	with open(out_file, 'w') as f:
		train_times = []
		train_aucs = []
		val_times = []
		val_aucs = []
		test_times = []
		test_aucs = []
		spaces = []
		line = f'{delimiter}'.join([''] * 7) + delimiter
		for i, (i_repeat, vs) in enumerate(data.items()):
			train_, val_, test_ = vs['train'], vs['val'], vs['test']
			train_times.append(train_['train_time'])
			train_aucs.append(0)
			val_times.append(0)
			val_aucs.append(0)
			test_times.append(test_['test_time'])
			test_aucs.append(test_['test_auc'])
			spaces.append(vs['space'])
			if i == 0:
				args = vs['args']
				data_shape = "|".join(str(v) for v in [0, 0, test_['test_shape'][0]])
				dim = str(test_['test_shape'][1])
				line = f'{delimiter}'.join([args.dataset, args.feature, f'header_{args.header}',
				                            args.model, f'tuning_{args.tuning}', data_shape, dim]) + delimiter

		line += f'-'.join([str(v) for v in train_times]) + delimiter + \
		        f'-'.join([str(v) for v in train_aucs]) + delimiter + \
		        f'-'.join([str(v) for v in val_times]) + delimiter + \
		        f'-'.join([str(v) for v in val_aucs]) + delimiter + \
		        f'-'.join([str(v) for v in test_times]) + delimiter + \
		        f'-'.join([str(v) for v in test_aucs]) + delimiter + \
		        f'-'.join([str(v) for v in spaces])

		f.write(line + '\n')
		lg.debug(line)


class Args():

	def __init__(self, params):
		for k, v in params.items():
			# self.args.k = v
			setattr(self, k, v)


def evaluate(in_dir, out_dir, n_repeats=5, n_test_repeats=10, FEATURES=[], HEADERS=[]):
	# 1. reconstruct new models
	for dataset, feature, header, model, tuning in list(
			itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS)):
		try:
			test_file = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
			                         f'test_set.dat')
			test_set = load(test_file)
			history = {}
			for i in range(n_repeats):
				model_params_file = os.path.join(in_dir, dataset, feature, f'header_{header}', model,
				                                 f'tuning_{tuning}',
				                                 f'model_params_{i}th.dat')
				lg.debug(model_params_file)
				params = load(model_params_file)
				reconstructed_model = reconstruct_model(params)
				# 3. Evaluate new models
				out_ = _evaluate(reconstructed_model, test_set, n_test_repeats)
				# out = minimal_model_cost(model_params_file, test_set, n_test_repeats)
				args = Args(
					{'dataset': dataset, 'feature': feature, 'header': header, 'model': model, 'tuning': tuning})
				history[f'{i}th_repeat'] = {'args': args, 'train': params['extra_params']['train'],
				                            'val': '', 'test': out_,
				                            'space': params['space']}

			out_file = os.path.join(out_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
			                        f'res.dat')
			check_path(out_file)
			dump(history, out_file)
			out_file = os.path.splitext(out_file)[0] + '.csv'
			save_dict2txt(history, out_file)
		except Exception as e:
			lg.error(e)
			traceback.print_exc()


@timer
def main(out_dir, n_normal_max_train=1000):
	n_repeats = 5
	model_in_dir = f'examples/offline/deployment/data/train_size_{n_normal_max_train}/src_dst/models'
	flg = False # this is only used in the trained server (e.g., NEON or MacOS), not RSPI(32bit) and NANO(64bit)
	if flg:
		# 1. only extract needed parameters from each built model. It should be done before deploying.
		lg.debug(f'\n***Extract needed parameters')
		in_dir = f'examples/offline/out/train_size_{n_normal_max_train}/src_dst'
		extract_needed_parameters(in_dir, model_in_dir, n_repeats)
	# 2. deployment: upload the models and test set to different servers using (scp)
	# lg.debug(f'\n***Upload models')

	# 3. evaluate models
	for (feature, header) in [('IAT+SIZE', False), ('STATS', True)]:  #
		if feature not in FEATURES: continue
		if header not in HEADERS: continue
		lg.debug(f'\n***Evaluate models, feature: {feature}, header: {header}')
		in_dir = model_in_dir
		out_dir = f'examples/offline/deployment/out/train_size_{n_normal_max_train}/src_dst'
		# n_test_repeats = 100
		# evaluate(in_dir, out_dir, n_repeats, n_test_repeats, FEATURES=[feature], HEADERS=[header])

		# 4. Gather all the individual result
		lg.debug(f'\n***Gather models')
		in_dir = out_dir
		out_file = gather(in_dir, out_dir=os.path.join(in_dir, RESULT_DIR, f'{feature}-header_{header}'),
		                  FEATURES=[feature], HEADERS=[header])
		lg.info(out_file)


if __name__ == '__main__':
	# main()
	for n_normal_max_train in  [1000, 2000, 3000, 4000, 5000]: # 1000, 3000, 5000, 10000, [1000, 2000, 3000, 4000, 5000]
		out_dir = os.path.join(OUT_DIR, f'train_size_{n_normal_max_train}')
		main(out_dir, n_normal_max_train)
