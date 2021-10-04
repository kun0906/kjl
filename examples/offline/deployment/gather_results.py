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
from kjl.models.ocsvm import _OCSVM
from kjl.projections.kjl import KJL
from kjl.projections.nystrom import Nystrom
from kjl.utils import pstats
from kjl.utils.tool import load, dump, check_path, timer
import pandas as pd

RESULT_DIR = f'results/{START_TIME}'
# DATASETS = ['UNB3_345']  # Two different normal data
MODELS = ["OCSVM(rbf)"]
FEATURES = ['IAT+SIZE', 'STATS']
HEADERS = [False, True]
TUNINGS = [False, True]


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

			test_times.append('')
			test_aucs.append('')
			spaces.append(vs['space'])
			if i == 0:
				args = vs['args']
				data_shape = "|".join(str(v) for v in [0, 0,0])
				dim = str(0)
				line = f'{delimiter}'.join([args.dataset, args.feature, f'header_{args.header}',
				                            args.model, f'tuning_{args.tuning}', data_shape, dim]) + delimiter

		line += f'-'.join([str(v) for v in train_times]) + delimiter + \
		        f'-'.join([str(v) for v in train_aucs]) + delimiter + \
		        f'-'.join([str(v) for v in val_times]) + delimiter + \
		        f'-'.join([str(v) for v in val_aucs]) + delimiter + \
		        f'-'.join([str(v) for v in test_times]) + delimiter + \
		        f'-'.join([str(v) for v in test_aucs]) + delimiter + \
		        f'-'.join([str(v) for v in spaces])

		v = delimiter + '|'.join(str(v) for v in ['shape'] + list(test_['shape_fit_']) + ['n_support'] + list(test_['_n_support']) +['gamma_' + str(test_['_gamma'])])
		line +=v
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
			# test_file = os.path.join(in_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
			#                          f'test_set.dat')
			# test_set = load(test_file)
			history = {}
			for i in range(n_repeats):
				model_params_file = os.path.join(in_dir, dataset, feature, f'header_{header}', model,
				                                 f'tuning_{tuning}',
				                                 f'model_params_{i}th.dat')
				params = load(model_params_file)
				out_ = params['model']
				args = Args(
					{'dataset': dataset, 'feature': feature, 'header': header, 'model': model, 'tuning': tuning})
				history[f'{i}th_repeat'] = {'args': args, 'train': params['extra_params']['train'],
				                            'val': '', 'test': out_,
				                            'space': params['space']}

			out_file = os.path.join(out_dir, dataset, feature, f'header_{header}', model, f'tuning_{tuning}',
			                        f'~res.dat')
			check_path(out_file)
			out_file = os.path.splitext(out_file)[0] + '-full_params.csv'
			save_dict2txt(history, out_file)
		except Exception as e:
			lg.error(e)
			traceback.print_exc()


@timer
def main():
	n_repeats = 5
	flg = False
	model_in_dir = 'examples/offline/deployment/data/src_dst/models'
	# if flg:
	# 	# 1. only extract needed parameters from each built model. It should be done before deploying.
	# 	lg.debug(f'\n***Extract needed parameters')
	# 	in_dir = 'examples/offline/out/src_dst'
	# 	extract_needed_parameters(in_dir, model_in_dir, n_repeats)
	# # 2. deployment: upload the models and test set to different servers using (scp)
	# # lg.debug(f'\n***Upload models')

	# 3. evaluate models
	for (feature, header) in [('IAT+SIZE', False), ('STATS', True)]:  #
		lg.debug(f'\n***Evaluate models, feature: {feature}, header: {header}')
		in_dir = model_in_dir
		out_dir = 'examples/offline/deployment/out/src_dst'
		n_test_repeats = 100
		evaluate(in_dir, out_dir, n_repeats, n_test_repeats, FEATURES=[feature], HEADERS=[header])



if __name__ == '__main__':
	main()
