""" Main function for the offline application
The purpose here is to train models and get the fitted models' parameters.
We will deploy the models to different devices and then obtain the final evaluated AUCs.

Main steps:
	1. Parse data and extract features # (origianl data in "../Datasets")
	2. Create and builds models
	3. Evaluate models on variate datasets

Command:
	current directory is project_root_dir (i.e., kjl/.)
	PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/offline.py > log.txt 2>&1 &


# 1. repeat model 5 times and get the best one
# 2. testing the best model 20 tims on the test set to get the average/std auc.

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import copy
import itertools
import os.path
import traceback
from collections import Counter

import configargparse
import pandas as pd
from joblib import Parallel, delayed

from examples.offline import _offline
from examples.offline._constants import *
from examples.offline._offline import Data
from kjl.utils.tool import dump, timer, check_path, remove_file, get_train_val, get_test_rest, load

RESULT_DIR = f'results/{START_TIME}'
DATASETS = ['AECHO1_2020']  # Two different normal data, MAWI1_2020
FEATURES = ['IAT+SIZE']
HEADERS = [False]
# MODELS = [  "Nystrom-QS-GMM(full)",   "Nystrom-QS-GMM(diag)"] # "OCSVM(rbf)", "GMM(full)", "GMM(diag)", "KJL-GMM(full)", "KJL-GMM(diag)",
# MODELS = ['GMM(full)']
TUNINGS = [True, False]

lg.debug(f'DATASETS: {DATASETS}, FEATURES: {FEATURES}, HEADERS: {HEADERS}, MODELS: {MODELS}, TUNINGS: {TUNINGS}')


@timer
def offline_default_best_main(args, train_set, val_set, test_set, i_repeat=0):
	""" The comman function for the algorithms with default and best parameters

	Parameters
	----------
	args
	train_set
	val_set
	test_set
	i_repeat

	Returns
	-------

	"""
	# model =   # full model name
	params = {
		'is_kjl': False, 'kjl_d': 5, 'kjl_n': 100, 'kjl_q': 0.3,
		'is_nystrom': False, 'nystrom_d': 5, 'nystrom_n': 100, 'nystrom_q': 0.3,
		'is_quickshift': False, 'quickshift_k': 100, 'quickshift_beta': 0.9,
		'random_state': RANDOM_STATE
	}
	# for k, v in params.items():
	# 	# self.args.k = v
	# 	setattr(args, k, v)
	args.params = params
	qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
	GMM_n_components = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	X_train, y_train = train_set
	quickshift_k = int(X_train.shape[0] ** (2 / 3))
	quickshift_beta = 0.9
	if args.model == 'OCSVM(rbf)':
		args.params['kernel'] = 'rbf'
		if not args.tuning:
			args.params['OCSVM_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.params['OCSVM_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)
	elif args.model == 'KDE':
		if not args.tuning:
			args.params['kde_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.params['kde_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	elif args.model == "KJL-OCSVM(linear)":
		args.params['kernel'] = 'linear'
		args.params['is_kjl'] = True
		args.params['kjl_d'] = 5
		args.params['kjl_n'] = 100
		if not args.tuning:
			args.params['OCSVM_q'] = 0.3
			args.params['kjl_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.params['OCSVM_q'] = q_
				args.params['kjl_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	elif args.model == "Nystrom-OCSVM(linear)":
		args.params['kernel'] = 'linear'
		args.params['is_nystrom'] = True
		args.params['nystrom_d'] = 5
		args.params['nystrom_n'] = 100
		if not args.tuning:
			args.params['OCSVM_q'] = 0.3
			args.params['nystrom_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.params['OCSVM_q'] = q_
				args.params['nystrom_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	elif args.model in ["GMM(full)", "GMM(diag)"]:
		if 'full' in args.model:
			args.params['GMM_covariance_type'] = 'full'
		elif 'diag' in args.model:
			args.params['GMM_covariance_type'] = 'diag'
		else:
			msg = args.model
			raise NotImplementedError(msg)

		if not args.tuning:
			args.params['GMM_n_components'] = 1
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best components
			res = {'score': 0}
			for n_comps_ in GMM_n_components:
				args.params['GMM_n_components'] = n_comps_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	elif args.model in ["KJL-GMM(full)", "KJL-GMM(diag)"]:
		args.params['is_kjl'] = True
		args.params['kjl_d'] = 5
		args.params['kjl_n'] = 100
		if 'full' in args.model:
			args.params['GMM_covariance_type'] = 'full'
		elif 'diag' in args.model:
			args.params['GMM_covariance_type'] = 'diag'
		else:
			msg = args.model
			raise NotImplementedError(msg)

		if not args.tuning:
			args.params['kjl_q'] = 0.3
			args.params['GMM_n_components'] = 1
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				for n_comps_ in GMM_n_components:
					args.params['kjl_q'] = q_
					args.params['GMM_n_components'] = n_comps_
					res_ = _offline.main(args, train_set, val_set)
					if res_['score'] > res['score']:
						res = copy.deepcopy(res_)

	elif args.model in ["Nystrom-GMM(full)", "Nystrom-GMM(diag)"]:
		args.params['is_nystrom'] = True
		args.params['nystrom_d'] = 5
		args.params['nystrom_n'] = 100
		if 'full' in args.model:
			args.params['GMM_covariance_type'] = 'full'
		elif 'diag' in args.model:
			args.params['GMM_covariance_type'] = 'diag'
		else:
			msg = args.model
			raise NotImplementedError(msg)

		if not args.tuning:
			args.params['nystrom_q'] = 0.3
			args.params['GMM_n_components'] = 1
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				for n_comps_ in GMM_n_components:
					args.params['nystrom_q'] = q_
					args.params['GMM_n_components'] = n_comps_
					res_ = _offline.main(args, train_set, val_set)
					if res_['score'] > res['score']:
						res = copy.deepcopy(res_)

	elif args.model in ["KJL-QS-GMM(full)", "KJL-QS-GMM(diag)"]:

		args.params['is_kjl'] = True
		args.params['kjl_d'] = 5
		args.params['kjl_n'] = 100

		args.params['is_quickshift'] = True
		args.params['quickshift_k'] = quickshift_k
		args.params['quickshift_beta'] = 0.9

		if 'full' in args.model:
			args.params['GMM_covariance_type'] = 'full'
		elif 'diag' in args.model:
			args.params['GMM_covariance_type'] = 'diag'
		else:
			msg = args.model
			raise NotImplementedError(msg)
		if not args.tuning:
			args.params['kjl_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.params['kjl_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	elif args.model in ["Nystrom-QS-GMM(full)", "Nystrom-QS-GMM(diag)"]:

		args.params['is_nystrom'] = True
		args.params['nystrom_d'] = 5
		args.params['nystrom_n'] = 100

		args.params['is_quickshift'] = True
		args.params['quickshift_k'] = quickshift_k
		args.params['quickshift_beta'] = 0.9

		if 'full' in args.model:
			args.params['GMM_covariance_type'] = 'full'
		elif 'diag' in args.model:
			args.params['GMM_covariance_type'] = 'diag'
		else:
			msg = args.model
			raise NotImplementedError(msg)
		if not args.tuning:
			args.params['nystrom_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.q = q_
				args.params['nystrom_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	elif args.model in ["KJL-QS-init_GMM(full)", "KJL-QS-init_GMM(diag)"]:
		args.params['is_kjl'] = True
		args.params['kjl_d'] = 5
		args.params['kjl_n'] = 100

		if 'full' in args.model:
			args.params['GMM_covariance_type'] = 'full'
		elif 'diag' in args.model:
			args.params['GMM_covariance_type'] = 'diag'
		else:
			msg = args.model
			raise NotImplementedError(msg)
		args.params['GMM_is_init_all'] = True

		args.params['is_quickshift'] = True
		args.params['quickshift_k'] = quickshift_k
		args.params['quickshift_beta'] = 0.9

		if not args.tuning:
			args.params['kjl_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.params['kjl_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	elif args.model in ["Nystrom-QS-init_GMM(full)", "Nystrom-QS-init_GMM(diag)"]:

		args.params['is_nystrom'] = True
		args.params['nystrom_d'] = 5
		args.params['nystrom_n'] = 100

		if 'full' in args.model:
			args.params['GMM_covariance_type'] = 'full'
		elif 'diag' in args.model:
			args.params['GMM_covariance_type'] = 'diag'
		else:
			msg = args.model
			raise NotImplementedError(msg)
		args.params['GMM_is_init_all'] = True

		args.params['is_quickshift'] = True
		args.params['quickshift_k'] = quickshift_k
		args.params['quickshift_beta'] = 0.9

		if not args.tuning:
			args.params['nystrom_q'] = 0.3
			res = _offline.main(args, train_set, val_set)
		else:
			# find the best OCSVM_q
			res = {'score': 0}
			for q_ in qs:
				args.params['nystrom_q'] = q_
				res_ = _offline.main(args, train_set, val_set)
				if res_['score'] > res['score']:
					res = copy.deepcopy(res_)

	else:
		msg = f"{args.model}"
		raise NotImplementedError(msg)

	best = res

	# save the best model
	best_model_file = os.path.join(args.out_dir, args.direction, args.dataset, args.feature,
	                               f'header_{args.header}', args.model, f'tuning_{args.tuning}',
	                               f'model_{i_repeat}th.dat')
	check_path(best_model_file)
	dump(best, out_file=best_model_file)
	lg.debug(f'best_model_file: {best_model_file}')

	##########################################################################
	# Evaluation
	# load the best model
	best = load(best_model_file)
	model = best['model']['model']
	# get train results first
	X_train, y_train, X_val, y_val = best['model']['data']
	train_res = {'train_time': model.train_time, 'train_auc': 0,
	             'train_shape': X_train.shape, 'y': Counter(y_train),
	             'X_train': X_train, 'y_train': y_train}
	val_res = {'val_time': model.test_time, 'val_auc': model.auc,
	           'val_shape': X_val.shape, 'y': Counter(y_val),
	           'X_val': X_val, 'y_val': y_val}

	# testing
	lg.debug('***Testing ')
	X_test, y_test = test_set
	out = {}
	test_times = []
	test_aucs = []
	# get average testing results
	n_test_repeats = args.n_test_repeats
	for i in range(n_test_repeats):
		tmp_ = model.test(X_test, y_test)
		out[f'{i}_repeat'] = tmp_
		test_times.append(tmp_['test_time'])
		test_aucs.append(tmp_['auc'])
	test_res = {'test_time': np.mean(test_times), 'test_auc': np.mean(test_aucs),
	            'times': test_times, 'aucs': test_aucs, 'test_shape': X_test.shape, 'y': Counter(y_test),
	            'X_test': X_test, 'y_test': y_test}
	lg.debug(f'test_res: {test_res}')
	res = {'best': best, 'best_model_file': best_model_file, 'train': train_res, 'val': val_res, 'test': test_res}

	# save all data in this repeat (includes model and train, val, test set)
	out_file = os.path.join(args.out_dir, args.direction, args.dataset, args.feature,
	                        f'header_{args.header}', args.model, f'tuning_{args.tuning}',
	                        f'repeat_{i_repeat}th.dat')
	check_path(out_file)
	dump(res, out_file=out_file)
	lg.debug(f'save all data in this repeat to : {out_file}')

	return res


def parser():
	""" parser commands

	Returns
	-------
		args: class (Namespace)
	"""
	p = configargparse.ArgParser()
	p.add_argument('-m', '--model', default='OCSVM', type=str, required=False, help='model name')
	p.add_argument('-p', '--model_params', default={'q': 0.3}, type=str, required=False, help='model params')
	p.add_argument('-t', '--tuning', default=False, type=str, required=False, help='tuning params')
	p.add_argument('-d', '--dataset', default='UCHI(SFRIG_2021)', type=str, help='dataset')
	p.add_argument('-D', '--direction', default='src', type=str, help='flow direction (src or src+dst)')
	p.add_argument('-H', '--header', default=True, type=bool, help='header')
	p.add_argument('-f', '--feature', default='FFT_IAT', type=str, help='IAT+SIZE')
	p.add_argument('-v', '--verbose', default=10, type=int, help='verbose')
	p.add_argument('-W', '--overwrite', default=False, type=bool, help='overwrite')
	p.add_argument('-o', '--out_dir', default='examples/offline/out', type=str, help='output directory')

	args = p.parse_args()
	return args


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
		line = f'{delimiter}'.join([''] * 7) + delimiter
		for i, (i_repeat, vs) in enumerate(data.items()):
			train_, val_, test_ = vs['train'], vs['val'], vs['test']
			train_times.append(train_['train_time'])
			train_aucs.append(train_['train_auc'])
			val_times.append(val_['val_time'])
			val_aucs.append(val_['val_auc'])
			test_times.append(test_['test_time'])
			test_aucs.append(test_['test_auc'])
			if i == 0:
				args = vs['best']['model']['args']
				data_shape = "|".join(
					str(v) for v in [train_['train_shape'][0], val_['val_shape'][0], test_['test_shape'][0]])
				dim = str(train_['train_shape'][1])
				line = f'{delimiter}'.join([args.dataset, args.feature, f'header_{args.header}',
				                            args.model, f'tuning_{args.tuning}', data_shape, dim]) + delimiter

		line += f'{np.mean(train_times):.5f}|{np.std(train_times):.5f}' + delimiter + \
		        f'{np.mean(train_aucs):.5f}|{np.std(train_aucs):.5f}' + delimiter + \
		        f'{np.mean(val_times):.5f}|{np.std(val_times):.5f}' + delimiter + \
		        f'{np.mean(val_aucs):.5f}|{np.std(val_aucs):.5f}' + delimiter + \
		        f'{np.mean(test_times):.5f}|{np.std(test_times):.5f}' + delimiter + \
		        f'{np.mean(test_aucs):.5f}|{np.std(test_aucs):.5f}'

		f.write(line + '\n')
		lg.debug(line)


def _main_entry(args):
	try:
		data = Data(name=args.dataset, flow_direction=args.direction, feature_name=args.feature,
		            header=args.header, out_dir = args.out_dir,
		            overwrite=args.overwrite, random_state=RANDOM_STATE)
		data.generate()
		if 'SAMP' in args.feature:
			previous = -1
			for key in data.y.keys():  # data[0.1] = [0, 0, ... 1, 1,1]
				X = data.X[key]
				y = data.y[key]
				lg.info(f'\n--- 1.2 Split train and test set')
				# split normal and abnormal
				X_normal = X[y == 0]
				y_normal = y[y == 0]
				X_abnormal = X[y == 1]
				y_abnormal = y[y == 1]
				X_test, y_test, X_abnormal_rest, y_abnormal_rest, X_normal, y_normal = get_test_rest(X_normal, y_normal,
				                                                                                     X_abnormal,
				                                                                                     y_abnormal,
				                                                                                     shuffle=True,
				                                                                                     random_state=RANDOM_STATE)
				history = {}
				average = 0
				for i in range(args.n_repeats):
					lg.debug(f'* {i}_th/ {args.n_repeats}(n_repeats)')
					args_i = copy.deepcopy(args)
					X_train, y_train, X_val, y_val = get_train_val(X_normal, y_normal, X_abnormal_rest, y_abnormal_rest,
					                                               val_size=int(len(y_test) / 4),  # test: val = 4:1
					                                               shuffle=True, random_state=RANDOM_STATE * (i + 100))
					# # normalization
					# ss, X_train, y_train, X_val, y_val, X_test, y_test = normalize(X_train, y_train, X_val, y_val, X_test,
					#                                                                y_test)
					lg.debug(f'X_train:{X_train.shape}, y_train: {Counter(y_train)}')
					lg.debug(f'X_val:{X_val.shape}, y_val: {Counter(y_val)}')
					lg.debug(f'X_test:{X_test.shape}, y_test: {Counter(y_test)}')

					# find the best model on the validation set and evaluate the built model on the test set :
					history_ = offline_default_best_main(args_i, (X_train, y_train), (X_val, y_val), (X_test, y_test),
					                                     i_repeat=i)
					history[f'{i}th_repeat'] = history_
					average += history_['test']['test_auc']
				average = average / args.n_repeats
				if average > previous:
					previous = average
					best = copy.deepcopy(history)
			history = copy.deepcopy(best)
		else:
			X = data.X
			y = data.y
			lg.info(f'\n--- 1.2 Split train and test set')
			# split normal and abnormal
			X_normal = X[y == 0]
			y_normal = y[y == 0]
			X_abnormal = X[y == 1]
			y_abnormal = y[y == 1]
			X_test, y_test, X_abnormal_rest, y_abnormal_rest, X_normal, y_normal = get_test_rest(X_normal, y_normal,
			                                                                                     X_abnormal,
			                                                                                     y_abnormal,
			                                                                                     shuffle=True,
			                                                                                     random_state=RANDOM_STATE)
			history = {}
			for i in range(args.n_repeats):
				lg.debug(f'* {i}_th/ {args.n_repeats}(n_repeats)')
				args_i = copy.deepcopy(args)
				X_train, y_train, X_val, y_val = get_train_val(X_normal, y_normal, X_abnormal_rest, y_abnormal_rest,
				                                               val_size=int(len(y_test) / 4),  # test: val = 4:1
				                                               shuffle=True, random_state=RANDOM_STATE * (i + 100),
				                                               n_normal_max_train = args.n_normal_max_train)
				# # normalization
				# ss, X_train, y_train, X_val, y_val, X_test, y_test = normalize(X_train, y_train, X_val, y_val, X_test,
				#                                                                y_test)
				lg.debug(f'X_train:{X_train.shape}, y_train: {Counter(y_train)}')
				lg.debug(f'X_val:{X_val.shape}, y_val: {Counter(y_val)}')
				lg.debug(f'X_test:{X_test.shape}, y_test: {Counter(y_test)}')

				# find the best model on the validation set and evaluate the built model on the test set :
				history_ = offline_default_best_main(args_i, (X_train, y_train), (X_val, y_val), (X_test, y_test),
				                                     i_repeat=i)
				history[f'{i}th_repeat'] = history_

		# save it immediately.
		out_file = os.path.join(args.out_dir, args.direction, args.dataset, args.feature,
		                        f'header_{args.header}', args.model, f'tuning_{args.tuning}', 'res.dat')
		check_path(out_file)
		dump(history, out_file=out_file)

		out_file = os.path.splitext(out_file)[0] + '.csv'
		save_dict2txt(history, out_file, delimiter=',')
		lg.debug(f'{out_file} exists: {os.path.exists(out_file)}')
	except Exception as e:
		lg.error(e)
		traceback.print_exc()
		history = {}
	return history


@timer
def gather(in_dir='src_dst', out_dir=''):
	""" collect all individual results together

	Parameters
	----------
	in_dir:
		search results from the given directory
	out_dir:
		save the gathered results to the given directory
	Returns
	-------
	out_file:
		the short csv for a quick overview

		each row in csv:
		line = f'{delimiter}'.join([args.dataset, args.feature, f'header_{args.header}',
				                            args.model, f'tuning_{args.tuning}', data_shape, dim]) + delimiter
		line += f'{np.mean(train_times):.5f}|{np.std(train_times):.5f}' + delimiter + \
		        f'{np.mean(train_aucs):.5f}|{np.std(train_aucs):.5f}' + delimiter + \
		        f'{np.mean(val_times):.5f}|{np.std(val_times):.5f}' + delimiter + \
		        f'{np.mean(val_aucs):.5f}|{np.std(val_aucs):.5f}' + delimiter + \
		        f'{np.mean(test_times):.5f}|{np.std(test_times):.5f}' + delimiter + \
		        f'{np.mean(test_aucs):.5f}|{np.std(test_aucs):.5f}'
	"""
	res = []
	for dataset, feature, header, model, tuning in list(itertools.product(DATASETS,
	                                                                      FEATURES, HEADERS, MODELS, TUNINGS)):
		f = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}', 'res.csv')
		try:
			line = [str(v) for v in pd.read_csv(f, sep=',', header=None).values.flatten().tolist()][1:]
			lg.debug(f, line)
			if len(str(line[0])) == 0:
				lg.error(f'Error: {line}. [{header}, {tuning}, {feature}, {dataset}, {model}]')
		except Exception as e:
			lg.error(f'Error: {e}. [{header}, {tuning}, {feature}, {dataset}, {model}]')
			line = ['', '0_0|0_0|0_0', '']  # [score, shape, params]
		res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}'] + line)

	# Save all results to gather.csv
	out_file = os.path.join(out_dir, 'gather.csv')
	check_path(out_file)
	with open(out_file, 'w') as f:
		for vs in res:
			f.write(','.join(vs) + '\n')

	# Only save needed data for quick overview
	short_file = os.path.join(os.path.split(out_file)[0], 'short.csv')
	with open(short_file, 'w') as f:
		for vs in res:
			f.write(','.join(vs) + '\n')

	return out_file


@timer
def _main(out_dir, n_normal_max_train=10000):
	""" Main function

	Returns
	-------

	"""
	res = []
	out_file = f'{out_dir}/{FLOW_DIRECTION}/{RESULT_DIR}/res.dat'
	is_parallel = False
	if is_parallel:  # with parallel
		# if backend='loky', the time taken is less than that of serial. but if backend='multiprocessing', we can
		# get very similar time cost comparing with serial.
		with Parallel(n_jobs=1, backend='loky') as parallel:
			args_lst = []
			for dataset, feature, header, model, tuning in list(
					itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS)):
				try:
					lg.info(f'*** {dataset}-{feature}-header_{header}, {model}-tuning_{tuning}')
					args = parser()
					args.out_dir = out_dir
					args.dataset = dataset
					args.feature = feature
					args.header = header
					args.model = model
					args.tuning = tuning
					args.overwrite = OVERWRITE
					args.random_state = RANDOM_STATE
					args.direction = FLOW_DIRECTION
					args.n_repeats = 5  # For training
					# For testing. the number (20) here doesn't matter (because we will upload the build models
					# to different devices and retest the models to get the final results)
					args.n_test_repeats = 20  # For testing
					# get results
					args_lst.append(copy.deepcopy(args))
				except Exception as e:
					lg.error(f'Error: {e}. [{dataset}, {feature}, {header}, {model}, {tuning}]')
					traceback.print_exc()
			res_ = parallel(delayed(_main_entry)(args) for args in args_lst)
		# reorganize results
		res = []
		for history, (dataset, feature, header, model, tuning) in zip(res_, list(
				itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS))):
			res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}', history])
	else:  # without parallel
		for dataset, feature, header, model, tuning in list(
				itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS)):
			try:
				lg.info(f'*** {dataset}-{feature}-header_{header}, {model}-tuning_{tuning}')
				args = parser()
				args.out_dir = out_dir
				args.dataset = dataset
				args.feature = feature
				args.header = header
				args.model = model
				args.tuning = tuning
				args.overwrite = OVERWRITE
				args.random_state = RANDOM_STATE
				args.direction = FLOW_DIRECTION
				args.n_repeats = 5  # For training
				# For testing. the number (20) here doesn't matter (because we will upload the build models
				# to different devices and retest the models to get the final results)
				args.n_test_repeats = 20
				args.n_normal_max_train = n_normal_max_train
				# get results
				history = _main_entry(args)

				res.append([dataset, feature, f'header_{header}', model, f'tuning_{tuning}', history])
				# avoid losing any result, so save it immediately.
				_out_file = f'{args.out_dir}/{args.direction}/{RESULT_DIR}/~res.csv'
				check_path(_out_file)
			# save2txt(res, _out_file, delimiter=',')
			except Exception as e:
				lg.error(f'Error: {e}. [{dataset}, {feature}, {header}, {model}, {tuning}]')
				traceback.print_exc()

	# save the final results: '.dat' and '.csv'
	check_path(out_file)
	dump(res, out_file)
	out_file = os.path.splitext(out_file)[0] + '.csv'
	remove_file(out_file, OVERWRITE)
	# save2txt(res, out_file, delimiter=',')
	lg.info(f'final result: {out_file}')


@timer
def main(out_dir='', n_normal_max_train=10000):
	""" Main function
		1. get all the results
		2. gather all results
	Returns
	-------

	"""
	# # clean()
	# 1. Run the main function and get the results for the given parameters
	try:
		_main(out_dir, n_normal_max_train)
	except Exception as e:
		lg.error(f'Error: {e}.')
		traceback.print_exc()

	# 2. Gather all the individual result
	try:
		in_dir = f'{out_dir}/src_dst'
		out_file = gather(in_dir, out_dir=os.path.join(in_dir, RESULT_DIR))
		lg.info(out_file)
	except Exception as e:
		lg.error(f'Error: {e}.')


if __name__ == '__main__':
	for n_normal_max_train in [1000, 3000, 5000, 8000, 10000]:
		out_dir = os.path.join(OUT_DIR, f'train_size_{n_normal_max_train}')
		main(out_dir, n_normal_max_train)
