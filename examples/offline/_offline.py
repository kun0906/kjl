""" Main function for the offline application

Main steps:
	1. Parse data and extract features
	2. Create and builds models
	3. Evaluate models on variate datasets

Command:
	current directory is project_root_dir (i.e., kjl/.)
	PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 applications/offline/_offline.py
"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import copy
# # must set these before loading numpy:
# os.environ["OMP_NUM_THREADS"] = '1'  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = '1'  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = '1'  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = '1' # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = '1' # export NUMEXPR_NUM_THREADS=6
import traceback
from collections import Counter

import configargparse
from func_timeout import func_set_timeout
from sklearn.model_selection import train_test_split

from examples.offline._constants import *
from examples.offline.datasets.ctu import CTU
from examples.offline.datasets.dummy import DUMMY
from examples.offline.datasets.maccdc import MACCDC
from examples.offline.datasets.mawi import MAWI
from examples.offline.datasets.uchi import UCHI
from examples.offline.datasets.unb import UNB
from kjl.models.gmm import GMM
from kjl.models.ocsvm import OCSVM
from kjl.projections.kjl import KJL
from kjl.projections.nystrom import Nystrom
from kjl.utils.tool import timer, check_path, dump, remove_file, save2txt, get_test_rest, \
	get_train_val


@timer
class Data:
	def __init__(self, name, verbose=10, overwrite=False, feature_name='IAT+SIZE', flow_direction='src_dst',
	             header=False, random_state=42):
		self.name = name
		self.verbose = verbose
		self.overwrite = overwrite
		self.feature_name = feature_name
		self.flow_direction = flow_direction
		self.header = header
		self.random_state = random_state

		if name == 'DUMMY':
			self.data = DUMMY(dataset_name=name, out_dir=OUT_DIR,
			                  feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                  random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'UNB3_345':
			self.data = UNB(dataset_name=name, out_dir=OUT_DIR,
			                feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'CTU1':
			self.data = CTU(dataset_name=name, out_dir=OUT_DIR,
			                feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'MAWI1_2020':
			self.data = MAWI(dataset_name=name, out_dir=OUT_DIR,
			                 feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                 random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name == 'MACCDC1':
			self.data = MACCDC(dataset_name=name, out_dir=OUT_DIR,
			                   feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                   random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		elif name in ['SFRIG1_2020','SFRIG1_2021', 'AECHO1_2020', 'DWSHR_2020','WSHR_2020',
		              'DWSHR_WSHR_2020', 'DWSHR_AECHO_2020', 'DWSHR_WSHR_AECHO_2020']:
			self.data = UCHI(dataset_name=name, out_dir=OUT_DIR,
			                 feature_name=feature_name, flow_direction=self.flow_direction, header=self.header,
			                 random_state=self.random_state, verbose=self.verbose, overwrite=self.overwrite)
		else:
			msg = name
			raise NotImplementedError(msg)

		self.X = None
		self.y = None

	def generate(self):
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
			self.proj = KJL(d=d, n=n, m=n, q=q, random_state=random_state)
		elif name == 'NYSTROM':
			self.proj = Nystrom(d=d, n=n, m=n, q=q, random_state=random_state)
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


	def fit(self, X, y=None):
		self.model.fit(X)

	def eval(self, X, y):
		# y_score = self.model.predict_proba(X)
		res = self.model.eval(X, y)

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

def _single_main(args,train_set, test_set):
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
		model = OCSVM(params= args.params)
	elif 'GMM' in args.model:
		model = GMM(params=args.params)
	else:
		msg = args.model
		raise NotImplementedError(msg)

	model.fit(X_train, y_train)

	res = model.test(X_test, y_test)

	data = (X_train, y_train, X_test, y_test)
	history = {'model': model,'data': data, 'args': args, 'res': res}
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
