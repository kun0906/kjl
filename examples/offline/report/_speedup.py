"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import itertools
import traceback

import pandas as pd

from examples.offline._constants import *
from kjl.utils.tool import save2txt


def _get_diff(cell_1, cell_2, same=False):
	""" cell_2 / cell_1

	# combine two random variables=
	# https://stats.stackexchange.com/questions/117741/adding-two-or-more-means-and-calculating-the-new-standard-deviation
	# E(X+Y) = E(X) + E(Y)
	# Var(X+Y) = var(X) + var(Y) + 2cov(X, Y)
	# If X and Y are independent, Var(X+Y) = var(X) + var(Y)

	# divide two independent random normal variables
	# https://en.wikipedia.org/wiki/Ratio_distribution
	#  E(X/Y) = E(X)E(1/Y)
	#  Var(X/Y) =  E([X/Y]**2)-[E([X/Y])]**2 = [E(X**2)] [E(1/Y**2)] - {E(X)]**2 [E(1/Y)]**2
	# {\displaystyle \sigma _{z}^{2}={\frac {\mu _{x}^{2}}{\mu _{y}^{2}}}\left({\frac {\sigma _{x}^{2}}{\mu _{x}^{2}}}+{\frac {\sigma _{y}^{2}}{\mu _{y}^{2}}}\right)}

	Parameters
	----------
	cell_1: "train_time: mean+/std)"
	cell_2

	Returns
	-------

	"""
	try:
		_arr_1 = cell_1.split(':')[1]
		_arr_1 = [float(_v.split(')')[0]) for _v in _arr_1.split('+/-')]

		if same:
			mu = _arr_1[0]
			sigma = _arr_1[1]
		else:
			_arr_2 = cell_2.split(':')[1]
			_arr_2 = [float(_v.split(')')[0]) for _v in _arr_2.split('+/-')]

			# cell2 - cell1
			# mu = _arr_2[0] - _arr_1[0]      #
			# sigma = np.sqrt(_arr_2[1] ** 2 + _arr_1[1] ** 2)

			# X~ N(u_x, s_x**2), Y~ N(u_y, s_y**2), Z = X/Y
			# X and Y are independent,
			# E(X/Y) = E(X)E(1/Y) = E(X)/E(Y)
			# cell2 / cell1
			u_x = _arr_1[0]  # OCSVM
			u_y = _arr_2[0]  # other method
			mu = u_x / u_y
			s_x = _arr_1[1]
			s_y = _arr_2[1]
			# one approximation for Var(X/Y)
			sigma = np.sqrt(((u_x / u_y) ** 2) * ((s_x / u_x) ** 2 + (s_y / u_y) ** 2))

		diff = f"{mu:.3f} +/- {sigma:.3f}"

	except Exception as e:
		traceback.print_exc()
		diff = f"{-1:.3f} +/- {-1:.3f}"
	return diff


def get_all_aucs(raw_values):
	values = raw_values.split(':')[-1].split('-')
	values = [float(v) for v in values]
	# for i, v in enumerate(raw_values):
	#     if i == 0:
	#         values.append(float(v.split(':')[1]))
	#     elif i == len(values)-1:
	#         values.append(float(v.split(']')[0]))
	#     else:
	#         values.append(float(v))

	return values


def get_auc_ratio(ocsvm_aucs, gmm_aucs):
	""" speedup = gmm/ocsvm

	Parameters
	----------
	ocsvm_aucs
	gmm_aucs

	Returns
	-------

	"""
	mean_ocsvm = np.mean(ocsvm_aucs)
	mean_gmm = np.mean(gmm_aucs)

	mean_ratio = mean_gmm / mean_ocsvm

	ratios = [v / mean_ocsvm for v in gmm_aucs]
	std_ratio = np.std(ratios)

	diff = f"{mean_ratio:.2f} +/- {std_ratio:.2f}"

	return diff


def get_one_result(df, dataset, feature, header, model, tuning):
	for line in df.values:
		if dataset == line[0] and feature == line[1] and header == line[2] and model == line[3] and tuning == line[4]:
			return line

	return ''


def get_speedup(baseline, model, scale=100):
	def _get(data):
		delimiter = '-'
		return [float(v) for v in data.split(delimiter)]

	# get baseline results
	# train set
	n_train = 5720 if 'MAWI' in baseline[0] else 10000
	# lg.info(f'{baseline[0]}, n_train: {n_train}')
	baseline_train_times = [v / n_train * scale for v in _get(baseline[7])]
	baseline_train_aucs = _get(baseline[8])
	# skip val results
	# test set
	n_test = float(baseline[5].split('|')[-1])
	lg.info(f'{baseline[0]}, n_train: {n_train}, n_test: {n_test}')
	baseline_test_times = [v / n_test * scale for v in _get(baseline[11])]
	baseline_test_aucs = _get(baseline[12])
	baseline_spaces = _get(baseline[13])

	if baseline[3] == model[3]:
		# baseline
		baseline_train_times = [v * 1000 for v in baseline_train_times]  # here change second to ms (1s == 1000ms)
		train_time_speedup = f"{np.mean(baseline_train_times):.2f} +/- {np.std(baseline_train_times):.2f}"
		train_auc_speedup = f"{np.mean(baseline_train_aucs):.2f} +/- {np.std(baseline_train_aucs):.2f}"

		baseline_test_times = [v * 1000 for v in baseline_test_times]
		test_time_speedup = f"{np.mean(baseline_test_times):.2f} +/- {np.std(baseline_test_times):.2f}"
		test_auc_speedup = f"{np.mean(baseline_test_aucs):.2f} +/- {np.std(baseline_test_aucs):.2f}"

		space_reduction = f"{np.mean(baseline_spaces):.2f} +/- {np.std(baseline_spaces):.2f}"
	else:
		# get model results
		# train set
		model_train_times = [v / n_train * scale for v in _get(model[7])]
		model_train_aucs = _get(model[8])
		# skip val results
		# test set
		model_test_times = [v / n_test * scale for v in _get(model[11])]
		model_test_aucs = _get(model[12])
		model_spaces = _get(model[13])

		train_time_speedup = get_auc_ratio(model_train_times, baseline_train_times)  # baseline / model
		train_auc_speedup = get_auc_ratio(baseline_train_aucs, model_train_aucs)  # model / baseline

		test_time_speedup = get_auc_ratio(model_test_times, baseline_test_times)  # baseline / model
		test_auc_speedup = get_auc_ratio(baseline_test_aucs, model_test_aucs)  # model / baseline

		space_reduction = get_auc_ratio(model_spaces, baseline_spaces)  # baseline / model

	return (train_time_speedup, train_auc_speedup, test_time_speedup, test_auc_speedup, space_reduction)


def baseline_table(in_file, out_file, header=False, feature='IAT+SIZE'):
	df = pd.read_csv(in_file, header=None)
	model = 'OCSVM(rbf)'
	res = {}
	for tuning in TUNINGS:
		tmp = []
		for dataset in DATASETS:
			model_result = get_one_result(df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
			tmp.append(model_result)
		res[f'tuning_{tuning}'] = tmp

	with open(out_file, 'w') as f:
		for tuning, tmp in res.items():
			f.write(f'\n\n{tuning}, {feature}, header_{header}\n')

			# AUC
			f.write(' & '.join([vs[8] for vs in tmp]).replace('+/-', '$\pm$') + '\n')
			# Train time
			f.write(' & '.join([vs[5] for vs in tmp]).replace('+/-', '$\pm$') + '\n')
			# Test time (Neon)
			f.write(' & '.join([vs[7] for vs in tmp]).replace('+/-', '$\pm$') + '\n')
			# Space
			f.write(' & '.join([vs[9] for vs in tmp]).replace('+/-', '$\pm$') + '\n')


def speedup_table(in_file, out_file, header=False, feature='IAT+SIZE'):
	df = pd.read_csv(in_file, header=None)
	res = {}
	for tuning in TUNINGS:
		tmp = []
		for dataset in DATASETS:
			for model in MODELS:
				model_result = get_one_result(df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
				tmp.append(model_result)
		res[f'tuning_{tuning}'] = tmp

	with open(out_file, 'w') as f:
		for tuning, tmp in res.items():
			for covariance_type in ['full', 'diag']:
				f.write(f'\n\n\n***{tuning}, {covariance_type}, {feature}, header_{header}\n')
				# OC-KJL Table (i.e., KJL-GMM(''))
				f.write(f'{tuning}, {covariance_type}, {feature}, header_{header},  OC-KJL Table\n')
				# Test time speedup (Neon)
				f.write(' & '.join([vs[7] for vs in tmp if f'KJL-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')
				# Space reduction
				f.write(' & '.join([vs[9] for vs in tmp if f'KJL-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')

				# OC-Nystrom Table
				f.write(f'\n\n{tuning}, {covariance_type}, {feature}, header_{header}, OC-Nystrom Table\n')
				# Test time speedup (Neon)
				f.write(' & '.join([vs[7] for vs in tmp if f'Nystrom-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')
				# Space reduction
				f.write(' & '.join([vs[9] for vs in tmp if f'Nystrom-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')

				# OC-KJL, OC-KJL-QS vs OC-Nystrom, OC-Nystrom-QS
				f.write(f'\n\n{tuning}, {covariance_type}, {feature}, header_{header}, '
				        f'OC-KJL, OC-KJL-QS vs OC-Nystrom, OC-Nystrom-QS\n')
				# Test time speedup (Neon)
				f.write(' & '.join([vs[7] for vs in tmp if f'KJL-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')
				# Space reduction
				f.write(' & '.join([vs[9] for vs in tmp if f'KJL-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')

				f.write(' & '.join([vs[7] for vs in tmp if f'KJL-QS-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')
				# Space reduction
				f.write(' & '.join([vs[9] for vs in tmp if f'KJL-QS-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')

				f.write(' & '.join([vs[7] for vs in tmp if f'Nystrom-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')
				# Space reduction
				f.write(' & '.join([vs[9] for vs in tmp if f'Nystrom-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')

				f.write(' & '.join([vs[7] for vs in tmp if f'Nystrom-QS-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')
				# Space reduction
				f.write(' & '.join([vs[9] for vs in tmp if f'Nystrom-QS-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\n')


def main(in_file, FEATURES=['IAT+SIZE'], HEADERS=[False]):
	###################################################################################################
	# 1. Get speedup for train, test and space
	# # in_file = 'examples/offline/deployment/out/src_dst/results/2021-09-22/gather.csv'
	# in_file = 'examples/offline/deployment/out/src_dst/results-20210928/RSPI/IAT+SIZE-header_False/gather.csv'
	df = pd.read_csv(in_file, header=None)
	res = []
	for dataset, feature, header, model, tuning in list(
			itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS)):
		try:
			# baseline
			baseline = get_one_result(df, dataset, feature, f'header_{header}', 'OCSVM(rbf)', f'tuning_{tuning}')
			# another model result
			model_result = get_one_result(df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')

			data = get_speedup(baseline, model_result)
		except Exception as e:
			lg.error(e)
			traceback.print_exc()
			data = (0, 0, 0, 0, 0)
		line = ','.join([dataset, feature, f'header_{header}', model, f'tuning_{tuning}'] + [str(v) for v in data])
		lg.debug(line)
		res.append(line)

	out_file = os.path.splitext(in_file)[0] + '-speedup.csv'
	save2txt(res, out_file)
	lg.info(f'out_file: {out_file}')

	###################################################################################################
	# 2. format results
	in_file = out_file
	baseline_file = os.path.splitext(in_file)[0] + '-baseline.txt'
	baseline_table(in_file, baseline_file, header, feature)
	lg.debug(f'baseline: {baseline_file}')

	in_file = out_file
	speedup_file = os.path.splitext(in_file)[0] + '-speedup.txt'
	speedup_table(in_file, speedup_file, header, feature)
	lg.debug(f'speedup: {speedup_file}')


if __name__ == '__main__':
	# in_file = 'examples/offline/deployment/out/src_dst/results/2021-09-22/gather.csv'
	root_dir = 'examples/offline/deployment/report/out/src_dst/results-20210928'
	for device in ['Neon', 'RSPI', 'Nano']:
		lg.info(f'\n\n***{device}, {root_dir}')
		in_file = os.path.join(root_dir, f'{device}/IAT+SIZE-header_False/gather.csv')
		lg.debug(in_file)
		main(in_file, FEATURES=['IAT+SIZE'], HEADERS=[False])
		in_file = os.path.join(root_dir, f'{device}/STATS-header_True/gather.csv')
		lg.debug(in_file)
		main(in_file, FEATURES=['STATS'], HEADERS=[True])
