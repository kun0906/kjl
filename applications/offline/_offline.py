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
import os
# # must set these before loading numpy:
# os.environ["OMP_NUM_THREADS"] = '1'  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = '1'  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = '1'  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = '1' # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = '1' # export NUMEXPR_NUM_THREADS=6

from collections import Counter

import configargparse
import numpy as np
from loguru import logger as lg
from sklearn.model_selection import train_test_split

from applications.offline._constants import *
from applications.offline.data.unb import UNB
from applications.offline.data.ctu import CTU
from applications.offline.data.mawi import MAWI
from applications.offline.data.maccdc import MACCDC
from applications.offline.data.uchi import UCHI
from kjl.models.gmm import GMM
from kjl.models.ocsvm import OCSVM
from kjl.projections.kjl import KJL
from kjl.projections.nystrom import Nystrom
from kjl.utils.tool import timer, check_path, dump

@timer
class Data:
	def __init__(self, name, verbose=10, overwrite=False, feature_name='IAT+SIZE'):
		self.name = name
		self.verbose = verbose
		self.overwrite = overwrite
		self.feature_name = feature_name

		if name == 'UNB345_3':
			self.data = UNB(out_dir='applications/offline/out', feature_name=feature_name)
		elif name == 'CTU1':
			self.data = CTU(out_dir='applications/offline/out', feature_name=feature_name)
		elif name == 'MAWI1_2020':
			self.data = MAWI(out_dir='applications/offline/out', feature_name=feature_name)
		elif name == 'MACCDC':
			self.data = MACCDC(out_dir='applications/offline/out', feature_name=feature_name)
		elif name in ['SFRIG1_2020', 'AECHO1_2020', 'DWSHR_WSHR_2020']:
			self.data = UCHI(out_dir='applications/offline/out', feature_name=feature_name)
		else:
			msg = name
			raise NotImplementedError(msg)

		self.X = None
		self.y = None

	def generate(self):
		self.X, self.y = self.data.generate()
		self.y = np.asarray([0 if v.startswith('normal') else 1 for v in self.y])
		return self.X, self.y


class Projection:

	def __init__(self, name='KJL',d = 5, n=10, m=10, q=0.3,  overwrite=False, random_state=RANDOM_STATE):
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

		if name == 'OCSVM':
			self.model = OCSVM(params={'OCSVM_q': kwargs['q']}, random_state=random_state)
		elif name == 'GMM':
			self.model = GMM()
		else:
			msg = name
			raise NotImplementedError(msg)

	def fit(self, X, y=None):
		self.model.fit(X)

	def eval(self, X, y):
		# y_score = self.model.predict_proba(X)
		res  = self.model.eval(X, y)
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


@timer
def main(args= None):
	args = parser() if args is None else args
	lg.debug(args)

	""" 1.1 Parse data and extract features
		
	"""
	lg.info(f'\n--- 1.1 Parse data')
	data = Data(name=args.dataset, overwrite=args.overwrite)
	data.generate()

	""" 1.2 Split train and test set

	"""
	lg.info(f'\n--- 1.2 Split train and test set')
	X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.3,
	                                                    shuffle=True, random_state=RANDOM_STATE)
	lg.debug(f'X_train:{X_train.shape}, y_train: {Counter(y_train)}')
	lg.debug(f'X_test:{X_test.shape}, y_test: {Counter(y_test)}')

	""" 1.3 preprocessing
		projection
	"""
	lg.info(f'\n--- 1.3 Preprocessing')
	proj = Projection(name='KJL')
	proj.fit(X_train)
	X_train = proj.transform(X_train)

	""" 2.1 Build the model

	"""
	lg.info(f'\n--- 2.1 Build the model')
	model = Model(name=args.model, q  = proj.q, overwrite=args.overwrite, random_state=RANDOM_STATE)
	model.fit(X_train, y_train)

	""" 2.2 Evaluate the model

	"""
	lg.info(f'\n--- 2.2 Evaluate the model')
	X_test = proj.transform(X_test)
	res = model.eval(X_test, y_test)

	""" 3. Dump the result to disk

	"""
	lg.info(f'\n--- 3. Save the result')
	res_file = os.path.join(args.out_dir, f'{args.dataset}-{args.model}.dat')
	check_path(os.path.dirname(res_file))
	dump(res, out_file=res_file)
	lg.info(f'res_file: {res_file}')

	return res

if __name__ == '__main__':
	main()
