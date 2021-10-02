"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import itertools
import os
import pandas as pd

from examples.offline._constants import *
from kjl.utils.tool import timer, check_path


@timer
def gather(in_dir='src_dst', DATASETS=[], FEATURES=[], HEADERS=[], MODELS=[], TUNINGS=[]):
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

	return res


def _format_auc(vs):
	vs = vs.split('|')
	# return f'{float(vs[0]):.2f}|{float(vs[1]):.2f}'
	return f'{float(vs[0]):.2f} $\pm$ {float(vs[1]):.2f}'


def main():
	history = {}
	in_dir = 'examples/offline/out/src_dst'
	DATASETS = [
		# 'DUMMY',
		# Final datasets for the paper
		# 'UNB3_345',  # Combine UNB3, UNB3 and UNB5 attack data as attack data and only use UNB3's normal as normal data
		# 'CTU1',  # Two different abnormal data
		'MAWI1_2020',  # Two different normal data
		# 'MACCDC1',  # Two different normal data
		# 'SFRIG1_2020', #  Two different normal data
		'SFRIG1_2021',  # Two different normal data
		# 'AECHO1_2020',  # Two different normal data
		'DWSHR_AECHO_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
		# 'DWSHR_WSHR_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
		# 'DWSHR_WSHR_AECHO_2020'
	]
	for feature, header in [('IAT+SIZE', False), ('STATS', True), ('SAMP_SIZE', False)]:
		history[(feature, header)] = gather(in_dir, DATASETS, [feature], [header], ['OCSVM(rbf)'], [True])

	out_dir = 'examples/offline/report/out/src_dst/results-20210928/different_features'
	# Save all results to gather.csv
	out_file = os.path.join(out_dir, 'different_features.csv')
	check_path(out_file)
	with open(out_file, 'w') as f:
		f.write(f'\\toprule\n')
		f.write('Dataset & ' + ' & '.join(DATASETS) + '\\\\ \n')
		for (feature_, header_), res_ in history.items():
			tmp = [_format_auc(vs[-1]) for vs in res_]
			f.write(f'{feature_} & ' + ' & '.join(tmp) + '\\\\ \n')
			f.write(f'\\midrule\n')
		f.write(f'\\bottomrule\n')

	lg.info(f'{out_file}, {os.path.exists(out_file)}')
	return out_file


if __name__ == '__main__':
	main()
