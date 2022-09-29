""" Main function for the offline application

Main steps:
	1. Parse data and extract features
	2. Create and builds models
	3. Evaluate models on variate datasets

Command:
	current directory is project_root_dir (i.e., kjl/.)
	PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/_offline.py
"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
# # must set these before loading numpy:
# os.environ["OMP_NUM_THREADS"] = '1'  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = '1'  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = '1'  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = '1' # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = '1' # export NUMEXPR_NUM_THREADS=6
import traceback

import configargparse

from examples.offline._constants import *
from examples.offline.datasets.ctu import CTU
from examples.offline.datasets.dummy import DUMMY
from examples.offline.datasets.maccdc import MACCDC
from examples.offline.datasets.mawi import MAWI
from examples.offline.datasets.uchi import UCHI
from examples.offline.datasets.unb import UNB
from kjl.models.gmm import GMM
from kjl.models.ocsvm import OCSVM
from kjl.models.kde import KDE
from kjl.projections.kjl import KJL
from kjl.projections.nystrom import Nystrom
from kjl.utils.tool import timer, load


@timer
class Data:
	def __init__(self, name, verbose=10, overwrite=False, feature_name='IAT+SIZE', flow_direction='src_dst',
	             out_dir = None,
	             header=False, random_state=42):
		self.name = name
		self.out_dir = out_dir
		self.verbose = verbose
		self.overwrite = overwrite
		self.feature_name = feature_name
		self.flow_direction = flow_direction
		self.header = header
		self.random_state = random_state

		if name == 'DUMMY':
			self.data = DUMMY(dataset_name=name, out_dir=out_dir,
			                  feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                  random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'UNB3_345':
			self.data = UNB(dataset_name=name, out_dir=out_dir,
			                feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'CTU1':
			self.data = CTU(dataset_name=name, out_dir=out_dir,
			                feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'MAWI1_2020':
			self.data = MAWI(dataset_name=name, out_dir=out_dir,
			                 feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                 random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'MACCDC1':
			self.data = MACCDC(dataset_name=name, out_dir=out_dir,
			                   feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                   random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name in ['SFRIG1_2020', 'SFRIG1_2021', 'AECHO1_2020', 'DWSHR_2020', 'WSHR_2020',
		              'DWSHR_WSHR_2020', 'DWSHR_AECHO_2020', 'DWSHR_WSHR_AECHO_2020']:
			self.data = UCHI(dataset_name=name, out_dir=out_dir,
			                 feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                 random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		else:
			msg = name
			raise NotImplementedError(msg)

		self.X = None
		self.y = None

	def generate(self):
		# if self.name == 'UNB3_345' or self.name == 'SFRIG1_2020': # for different training sizes
		# 	# Xy_file = f'examples/offline/out/src_dst/{self.name}/IAT+SIZE/header_False/Xy.dat'
		# 	# meta = load(Xy_file)
		# 	# # print(data)
		# 	# self.X, self.y = meta['X'], meta['y']
		# 	Xy_file = f'examples/offline/out/src_dst/{self.name}/IAT+SIZE/header_False/Xy.txt'
		# 	if self.name == 'UNB3_345': dim = 43
		# 	if self.name == 'SFRIG1_2020': dim = 25
		# 	with open(Xy_file, 'r') as f:
		# 		data = f.readlines()
		# 		X, y = [], []
		# 		for line in data:
		# 			line = line.split(',')
		# 			X.append([float(v) for v in line[:dim]])
		# 			y.append(''.join(line[dim:]))
		# 	self.X, self.y = np.asarray(X), np.asarray(y)
		# else:
		self.X, self.y = self.data.generate()
		if 'SAMP' in self.feature_name:
			for key, vs in self.y.items():
				self.y[key] = np.asarray([0 if v.startswith('normal') else 1 for v in vs])
		else:
			self.y = np.asarray([0 if v.startswith('normal') else 1 for v in self.y])
		return self.X, self.y


class Projection:

	def __init__(self, name='KJL', d=5, n=10, m=10, q=0.3, overwrite=False, random_state=RANDOM_STATE):
		self.name = name
		self.d = d
		self.n = n
		self.m = m
		self.q = q
		self.overwrite = overwrite
		self.random_state = random_state

		if name == 'KJL':
			# self.proj = KJL(d=d, n=n, m=n, q=q, random_state=random_state)
			params = {'kjl_d': self.d, 'kjl_n': self.n, 'm': self.m, 'kjl_q': self.q, 'random_state': self.random_state}
			self.proj = KJL(params)
		elif name == 'NYSTROM':
			# self.proj = Nystrom(d=d, n=n, m=n, q=q, random_state=random_state)
			params = {'nystrom_d': self.d, 'nystrom_n': self.n, 'm': self.m, 'nystrom_q': self.q, 'random_state': self.random_state}
			self.proj = Nystrom(params)
		else:
			msg = name
			raise NotImplementedError(msg)

	def fit(self, X):
		self.proj.fit(X)
		self.sigma = self.proj.sigma

	def transform(self, X):
		return self.proj.transform(X)


class Model:

	def __init__(self, name='OCSVM', overwrite=OVERWRITE, random_state=RANDOM_STATE, **kwargs):
		self.name = name
		self.overwrite = overwrite
		self.random_state = random_state
		if 'OCSVM' in name:
			self.model = OCSVM(params = {'kernel': 'rbf', 'OCSVM_q': 0.3, 'random_state': random_state})
		elif 'GMM' in name:
			self.model = GMM(params = {'random_state': random_state})
		else:
			msg = name
			raise NotImplementedError(msg)

	def fit(self, X, y=None):
		self.model.fit(X)

	def eval(self, X, y):
		# y_score = self.model.predict_proba(X)
		try:
			res = self.model.eval(X, y)
		except Exception as e:
			res = self.model.test(X, y)

		return res


@timer
def parser():
	# p = configargparse.ArgParser(default_config_files=['/etc/app/conf.d/*.conf', '~/.my_settings'])
	# p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
	# p.add('--genome', required=True,
	#       help='path to genome file')  # this option can be set in a config file because it starts with '--'
	# p.add('-v', help='verbose', action='store_true')
	# p.add('-d', '--dbsnp', help='known variants .vcf',
	#       env_var='DBSNP_PATH')  # this option can be set in a config file because it starts with '--'
	# p.add('vcf', nargs='+', help='variant file(s)')

	p = configargparse.ArgParser()
	p.add_argument('-m', '--model', default='OCSVM', type=str, required=False, help='model name')
	p.add_argument('-d', '--dataset', default='UNB', type=str, help='dataset')
	p.add_argument('-v', '--verbose', default=10, type=int, help='verbose')
	p.add_argument('-or', '--overwrite', default=False, type=bool, help='overwrite')
	p.add_argument('-o', '--out_dir', default='applications/offline/out', type=str, help='output directory')

	args = p.parse_args()
	lg.debug(p.format_values())  # useful for logging where different settings came from
	return args


def _single_main(args, train_set, test_set):
	""" Get the result on given parameters

	Parameters
	----------
	args
	X
	y
	test

	Returns
	-------
	res: evalated results
	data: (train, val, test)
	"""
	lg.debug(args)
	X_train, y_train = train_set
	X_test, y_test = test_set

	if 'OCSVM' in args.model:
		model = OCSVM(params=args.params)
	elif 'GMM' in args.model:
		model = GMM(params=args.params)
	elif 'KDE' in args.model:
		model = KDE(params=args.params)
	else:
		msg = args.model
		raise NotImplementedError(msg)

	model.fit(X_train, y_train)

	res = model.test(X_test, y_test)

	data = (X_train, y_train, X_test, y_test)
	history = {'model': model, 'data': data, 'args': args, 'res': res}
	return history


@timer
def main(args=None, train_set=None, test_set=None):
	""" Get the result according to the given parameters

	Parameters
	----------
	args
	test: boolean
		if we evaluate the built model on val set or test set
	Returns
	-------
	history: dict
		Return the best result on 'SAMP' related feature. Otherwise, return the result
	"""
	try:
		lg.debug(args)
		###############################################################################################################
		""" 1.1 Parse data and extract features

		"""
		lg.info(f'\n--- 1.1 Parse data')
		res_ = _single_main(args, train_set, test_set)
		history = {'score': res_['res']['score'], 'model': res_}

	except Exception as e:
		traceback.print_exc()
		history = {'score': 0, 'model': {'model': None}}

	return history


if __name__ == '__main__':
	main(test=True)
