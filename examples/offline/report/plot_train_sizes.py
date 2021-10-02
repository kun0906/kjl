"""
"""
import copy
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import itertools
import traceback
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from examples.offline._constants import *
from examples.offline._offline import Data
from examples.offline.offline import offline_default_best_main, parser, save_dict2txt, gather
from kjl.utils.tool import dump, timer, check_path, get_train_val, get_test_rest

OUT_DIR = 'examples/offline/out/different_train_sizes'
RESULT_DIR = f'results/{START_TIME}'
DATASETS = ['MAWI1_2020']  # Two different normal data
FEATURES = ['IAT+SIZE']
HEADERS = [False]
MODELS = ["OCSVM(rbf)"]
TUNINGS = [False]

lg.debug(f'DATASETS: {DATASETS}, FEATURES: {FEATURES}, HEADERS: {HEADERS}, MODELS: {MODELS}, TUNINGS: {TUNINGS}')


def get_train_val(X_normal, y_normal, X_abnormal, y_abnormal, train_size=1000,
                  val_size=100, shuffle=True, random_state=42):
	"""

	Parameters
	----------
	X
	y
	shuffle
	random_state

	Returns
	-------

	"""
	n_abnormal = val_size // 2
	X_abnormal, X_ab_val, y_abnormal, y_ab_val = train_test_split(X_abnormal, y_abnormal,
	                                                              test_size=n_abnormal,
	                                                              stratify=y_abnormal,
	                                                              random_state=random_state)

	# get test set first
	X_normal, X_val, y_normal, y_val = train_test_split(X_normal, y_normal, test_size=n_abnormal,
	                                                    shuffle=shuffle, random_state=random_state)
	X_val = np.concatenate([X_ab_val, X_val], axis=0)
	y_val = np.concatenate([y_ab_val, y_val], axis=0)

	n_normal_max_train = train_size
	X_train = X_normal[:n_normal_max_train]
	y_train = y_normal[:n_normal_max_train]
	return X_train, y_train, X_val, y_val


def _main_entry(args):
	try:
		data = Data(name=args.dataset, flow_direction=args.direction, feature_name=args.feature,
		            header=args.header,
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
					                                               train_size=args.train_size,
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
				                                               train_size=args.train_size,
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
def _main(train_size=10):
	""" Main function

	Returns
	-------

	"""
	res = []
	out_file = f'{OUT_DIR}/{FLOW_DIRECTION}/{RESULT_DIR}/res.dat'
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
					args.dataset = dataset
					args.feature = feature
					args.header = header
					args.model = model
					args.tuning = tuning
					args.overwrite = OVERWRITE
					args.random_state = RANDOM_STATE
					args.direction = FLOW_DIRECTION
					args.train_size = train_size
					args.n_repeats = 5  # For training
					args.n_test_repeats = 20  # For testing
					args.out_dir = os.path.join(OUT_DIR, f'train_size_{train_size}')
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
				args.dataset = dataset
				args.feature = feature
				args.header = header
				args.model = model
				args.tuning = tuning
				args.train_size = train_size
				args.overwrite = OVERWRITE
				args.random_state = RANDOM_STATE
				args.direction = FLOW_DIRECTION
				args.n_repeats = 5  # For training
				args.n_test_repeats = 20  # For testing
				args.out_dir = os.path.join(OUT_DIR, f'train_size_{train_size}')
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
	# out_file = os.path.splitext(out_file)[0] + f'-train_size_{train_size}.csv'
	# remove_file(out_file, OVERWRITE)
	# # save_dict2txt(res, out_file, delimiter=',')
	lg.info(f'final result: {out_file}')
	return res


def show_train_sizes(history, out_file='', title='auc', n_repeats=5):
	font_size = 15

	colors = ['blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green'][:2]
	labels = ['KJL', 'Nystrom']

	sub_dataset = []
	yerrs = []
	data_name = ''
	j = 0
	for i, (train_size, vs) in enumerate(history.items()):
		try:
			data_name = vs[0]
			m_auc, std_auc = str(vs[-1]).split('|')
			vs = [data_name, vs[3], int(train_size), float(m_auc)]
			yerrs.append(float(std_auc))
		except Exception as e:
			print(e)
			# traceback.print_exc()
			vs = ['0', '0', 0, 0]
			yerrs.append(0)
		sub_dataset.append(vs)

	df = pd.DataFrame(sub_dataset, columns=['datasets', 'model_name', 'train_size', 'auc'])
	# g = sns.barplot(y="diff", x='datasets', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
	print(sub_dataset, 'yerrs:', yerrs)
	plt.errorbar(df['train_size'].values, df['auc'].values, yerrs, capsize=3, color=colors[j], ecolor='tab:red',
	             label=labels[j])
	plt.xticks(df['train_size'].values, fontsize=font_size - 1)
	plt.yticks(fontsize=font_size - 1)
	# fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
	plt.xlabel('Train size ($n$)', fontsize=font_size)
	plt.ylabel('AUC', fontsize=font_size)
	# plt.title(f'{data_name}', fontsize = font_size-2)
	plt.legend(labels=['MAWI'], loc='upper right', fontsize=font_size - 2)
	plt.ylim([0.95, 1])
	# n_groups = len(show_datasets)
	# index = np.arange(n_groups)
	# # print(index)
	# # plt.xlim(xlim[0], n_groups)
	# plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
	plt.tight_layout()

	# plt.legend(show_repres, loc='lower right')
	# # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
	plt.savefig(out_file)  # should use before plt.show()
	plt.show()
	plt.close()

	return out_file


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
def main():
	# # clean()
	train_sizes = [1000, 2000, 3000, 4000, 5000]
	# # train_sizes = [500]
	# # 1. Run the main function and get the results for the given parameters
	# try:
	# 	history = {}
	# 	for train_size in train_sizes:
	# 		tmp = _main(train_size)
	# 		history[train_size] = tmp
	# # break
	# except Exception as e:
	# 	lg.error(f'Error: {e}.')
	# 	traceback.print_exc()
	#
	# out_file = os.path.join(OUT_DIR, 'train_size.dat')
	# check_path(out_file)
	# dump(history, out_file)

	history = {}
	for train_size in train_sizes:
		in_dir = f'{OUT_DIR}/train_size_{train_size}/src_dst'
		out_file = gather(in_dir, out_dir=in_dir)
		lg.info(out_file)
		history[train_size] = pd.read_csv(out_file, header=None).values.flatten()

	out_file = os.path.join(OUT_DIR, 'MAWI1_2020-OCSVM(rbf)-tuning_False-train_size.pdf')
	show_train_sizes(history, out_file)
	lg.info(f'{out_file}, {os.path.exists(out_file)}')


if __name__ == '__main__':
	main()
