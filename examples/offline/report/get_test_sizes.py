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
import itertools
import traceback

from examples.offline._constants import *
from kjl.utils.tool import load, dump, check_path, timer

RESULT_DIR = f'results/{START_TIME}'
# DATASETS = ['UNB3_345']  # Two different normal data
MODELS = ["OCSVM(rbf)"]
FEATURES = ['IAT+SIZE', 'STATS']
HEADERS = [False, True]
TUNINGS = [False, True]


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


def get_test_sizes(in_dir, out_dir, FEATURES=[], HEADERS=[]):
	# 1. reconstruct new models
	res = []
	for dataset, feature, header, model, tuning in list(
			itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS)):
		try:
			test_file = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
			                         f'test_set.dat')
			test_set = load(test_file)
			res.append((dataset, feature, f'header_{header}', model, f'tuning_{tuning}', get_space(test_set)))

		except Exception as e:
			lg.error(e)
			traceback.print_exc()

	out_file = os.path.join(out_dir, feature, f'header_{header}', f'tuning_{tuning}',
	                        f'test_size.csv')
	check_path(out_file)
	lg.info(out_file)
	with open(out_file, 'w') as f:
		for vs in res:
			f.write(','.join([str(v) for v in vs]) + '\n')


@timer
def main():
	model_in_dir = 'examples/offline/deployment/data/src_dst/models'

	# test_size
	for (feature, header) in [('IAT+SIZE', False), ('STATS', True)]:  #
		lg.debug(f'\n***Evaluate models, feature: {feature}, header: {header}')
		in_dir = model_in_dir
		out_dir = 'examples/offline/report/out/src_dst'
		get_test_sizes(in_dir, out_dir, FEATURES=[feature], HEADERS=[header])


if __name__ == '__main__':
	main()
