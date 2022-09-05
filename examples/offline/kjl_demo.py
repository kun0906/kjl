""" Main function for the offline application

Main steps:
	1. Parse data and extract features
	2. Create and builds models
	3. Evaluate models on variate datasets

Command:
	current directory is project_root_dir (i.e., kjl/.)
	PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/kjl_demo.py
"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
from collections import Counter

from sklearn.model_selection import train_test_split

from examples.offline._constants import *
from examples.offline._offline import Projection, Model
from examples.offline.datasets.ctu import CTU
from examples.offline.datasets.maccdc import MACCDC
from examples.offline.datasets.mawi import MAWI
from examples.offline.datasets.uchi import UCHI
from examples.offline.datasets.unb import UNB
from kjl.utils.tool import timer, check_path, dump, load

lg.remove()
lg.add(sys.stdout, format="{message}", level='INFO')


@timer
class Data:
	def __init__(self, name, verbose=10, overwrite=False, feature_name='IAT+SIZE', flow_direction='src_dst'):
		self.name = name
		self.verbose = verbose
		self.overwrite = overwrite
		self.feature_name = feature_name

		if name == 'UNB345_3':
			self.data = UNB(out_dir=OUT_DIR, feature_name=feature_name, flow_direction=flow_direction)
		elif name == 'CTU1':
			self.data = CTU(out_dir=OUT_DIR, feature_name=feature_name, flow_direction=flow_direction)
		elif name == 'MAWI1_2020':
			self.data = MAWI(out_dir=OUT_DIR, feature_name=feature_name, flow_direction=flow_direction)
		elif name == 'MACCDC':
			self.data = MACCDC(out_dir=OUT_DIR, feature_name=feature_name, flow_direction=flow_direction)
		elif name in ['SFRIG1_2020', 'AECHO1_2020', 'DWSHR_WSHR_2020']:
			self.data = UCHI(out_dir=OUT_DIR, feature_name=feature_name, flow_direction=flow_direction)
		else:
			msg = name
			raise NotImplementedError(msg)

		self.X = None
		self.y = None

	def generate(self):
		self.X, self.y = self.data.generate()
		self.y = np.asarray([0 if v.startswith('normal') else 1 for v in self.y])
		return self.X, self.y


@timer
def main(args=None):
	""" 1.1 Parse data and extract features

	"""
	lg.info(f'\n--- 1.1 Load data')
	# feat_file = f'{OUT_DIR}/DEMO_IAT+SIZE.dat'
	feat_file = f'{OUT_DIR}/src_dst/DUMMY/IAT+SIZE/Xy.dat'
	# X, y = load(feat_file)
	data = load(feat_file)
	X, y = data['X'], data['y']
	y = [0 if 'normal_0' == v else 1 for v in y]  # normal is 0 and abnormal is 1

	""" 1.2 Split train and test set

	"""
	lg.info(f'\n--- 1.2 Split train and test set')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
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
	model_name = 'OCSVM'
	model = Model(name=model_name, q=proj.q, overwrite=OVERWRITE, random_state=RANDOM_STATE)
	model.fit(X_train, y_train)

	""" 2.2 Evaluate the model

	"""
	lg.info(f'\n--- 2.2 Evaluate the model')
	X_test = proj.transform(X_test)
	res = model.eval(X_test, y_test)

	""" 3. Dump the result to disk

	"""
	lg.info(f'\n--- 3. Save the result')
	res_file = os.path.join(OUT_DIR, f'DEMO-KJL-{model_name}-results.dat')
	check_path(os.path.dirname(res_file))
	dump(res, out_file=res_file)
	lg.info(f'res_file: {res_file}')

	return res


if __name__ == '__main__':
	main()
