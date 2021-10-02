"""Useful tools includes 'data_info', 'dump_data', etc.

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

import inspect
import os
import pickle
import shutil
import subprocess
from datetime import datetime
from functools import wraps
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def timer(func):
	# This function shows the execution time of
	# the function object passed
	def wrap_func(*args, **kwargs):
		t1 = time()
		result = func(*args, **kwargs)
		t2 = time()
		print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
		return result

	return wrap_func


def load(in_file):
	"""load data from file

	Parameters
	----------
	in_file: str
		input file path

	Returns
	-------
	data:
		loaded data
	"""
	with open(in_file, 'rb') as f:
		data = pickle.load(f)

	return data


def dump(data, out_file=''):
	"""Save data to file

	Parameters
	----------
	data: any data

	out_file: str
		out file path
	Returns
	-------

	"""

	with open(out_file, 'wb') as f:
		pickle.dump(data, f)

	return out_file


def data_info(data=None, name='data'):
	"""Print data basic information

	Parameters
	----------
	data: array

	name: str
		data name

	Returns
	-------

	"""

	pd.set_option('display.max_rows', 500)
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 100)
	pd.set_option('display.float_format', lambda x: '%.3f' % x)  # without scientific notation

	columns = ['col_' + str(i) for i in range(data.shape[1])]
	dataset = pd.DataFrame(data=data, index=range(data.shape[0]), columns=columns)
	print(f'{name}.shape: {data.shape}')
	print(dataset.describe())
	print(dataset.info(verbose=True))


def check_path(file_path):
	"""Check if a path is existed or not.
	 If the path doesn't exist, then create it.

	Parameters
	----------
	file_path: str

	overwrite: boolean (default is True)
		if the path exists, delete all data in it and create a new one

	Returns
	-------

	"""
	dir_path = os.path.dirname(file_path)
	if os.path.exists(dir_path):
		return True
	else:
		os.makedirs(dir_path)
		return False


def timing(func):
	"""Calculate the execute time of the given func"""

	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		st = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
		print(f'\'{func.__name__}()\' starts at {st}')
		result = func(*args, **kwargs)
		end = time.time()
		ed = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
		tot_time = (end - start) / 60
		tot_time = float(f'{tot_time:.4f}')
		print(f'\'{func.__name__}()\' ends at {ed} and takes {tot_time} mins.')
		func.tot_time = tot_time  # add new variable to func
		return result, tot_time

	return wrapper


def merge_dicts(dict_1, dict_2):
	"""

	Parameters
	----------
	dict_1
	dict_2

	Returns
	-------

	"""
	for key in dict_2.keys():
		# if key not in dict_1.keys():
		dict_1[key] = dict_2[key]

	return dict_1


def func_args(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		_args = inspect.getfullargspec(func)  # get all paramters and the corresponding default values
		# # might not work for kwargs
		# _keys_values = dict(list(zip(_args.args, args)))  # it will fail when misses pass some (k,v) pairs
		_keys_values = list(args)  # args and kwargs only get passed values to the current func
		[_keys_values.append(f"{k}={v}") for k, v in kwargs.items() if k not in _keys_values]
		print(f"\'{func.__name__}'s arguments ( {_keys_values} )")
		result = func(*args, **kwargs)
		return result

	return wrapper


def execute_time(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		st = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
		print(f'\'{func.__name__}()\' starts at {st}')
		result = func(*args, **kwargs)
		end = time.time()
		ed = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
		tot_time = (end - start) / 60
		tot_time = float(f'{tot_time:.4f}')
		print(f'\'{func.__name__}()\' ends at {ed}. It takes {tot_time} mins')
		return result

	return wrapper


def time_func(func, *args, **kwargs):
	start = datetime.now()

	result = func(*args, **kwargs)

	end = datetime.now()
	total_time = (end - start).total_seconds()
	# print(f'{func} running time: {total_time}, and result: {result}')

	return result, total_time


def mprint(msg, verbose=10, level=1):
	if verbose >= level:
		print(msg)


def run(in_file, out_file):
	"""Save output to file by subprocess

		https://janakiev.com/blog/python-shell-commands/
		https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
		https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running

	Parameters
	----------
	in_file
	out_file

	Returns
	-------

	"""
	# cmd = f"python3.7 {in_file} > {out_file} 2>&1"
	cmd = f"python3.7 {in_file}"
	print(cmd)
	with open(out_file, 'wb') as f:
		# buffsize: 0 = unbuffered (default value); 1 = line buffered; N = approximate buffer size, ...
		# p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
		#                      bufsize=0, universal_newlines=True)
		p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		                     bufsize=1, universal_newlines=True)
		for line in p.stdout:
			# sys.stdout.write(line) # output to console
			f.write(line.encode())  # save to file
			print(line, end='')  # output to console

	return 0


def remove_file(in_file, overwrite=False):
	""" remove file

	Parameters
	----------
	in_file
	overwrite

	Returns
	-------

	"""
	if overwrite:
		if os.path.exists(in_file):
			os.remove(in_file)
		else:
			pass


def remove_dir(in_dir, overwrite=False):
	""" remove directory

	Parameters
	----------
	in_dir
	overwrite

	Returns
	-------

	"""
	if overwrite:
		if os.path.exists(in_dir):
			shutil.rmtree(in_dir)
		else:
			pass


def check_arr(X):
	""" Fill nan to 0

	Parameters
	----------
	X

	Returns
	-------
	X
	"""

	# X[np.isnan(X)] = 0
	# # X[np.isneginf(X)] = 0
	# # X[np.isinf(X)] = 0
	# X[np.isfinite(X)] = 0
	# print(X)
	# print(np.quantile(X, q=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]))
	# import pandas as pd
	# df = pd.DataFrame(X)
	# print(df.describe())
	#
	# assert_all_finite(X, allow_nan=False)
	# print(f'***{np.mean(X, axis=0)}, {np.any(np.isnan(X))}, '
	#       f'{np.any(np.isinf(X))}, {np.any(np.isneginf(X))}')
	#
	X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
	return X


def save2txt(data, out_file, delimiter=','):
	""" Save result to txt

	Parameters
	----------
	data: list

	out_file: path
	delimiter: ','

	Returns
	-------

	"""
	with open(out_file, 'w') as f:
		for vs in data:
			if type(vs) == list:
				line = f'{delimiter}'.join([str(v) for v in vs])
			else:
				line = str(vs)
			f.write(line + '\n')


def get_test_rest(X_normal, y_normal, X_abnormal, y_abnormal,
                  shuffle=True, random_state=42):
	"""

	Parameters
	----------
	X
	y
	shuffle
	random_state

	Returns
	-------
	n_normal: N
	n_abnormal: M

	n_train(normal): n
	n_test(abnormal): m

	# compute the number of abnormal test (m)
	n:m = 8:2
	(N-m):m = 8:2
	(N-m)*2 = 8*m
	N* 2 = 10m
	m = N * 2 // 10
	n = 8*m // 2
	"""
	print(f'normal:{len(y_normal)}, abnormal: {len(y_abnormal)}')
	n = 10000
	N = len(y_normal)
	m = len(y_abnormal)
	if N > 10000:
		n_train = 10000
	else:
		n_train = N

	n_abormal_max_test = n_train // 4
	if n_abormal_max_test < m:
		pass
	else:
		# (test: val = 4:1, here -100 is to make sure the validation set can have random data)
		n_abormal_max_test = m * 4 // 5 - 100

	print(f'***n_normal_max: {N}, n_abormal_max_test: {n_abormal_max_test}')

	if len(y_abnormal) > n_abormal_max_test:
		X_abnormal, X_abnormal_rest, y_abnormal, y_abnormal_rest = train_test_split(X_abnormal, y_abnormal,
		                                                                            train_size=n_abormal_max_test,
		                                                                            stratify=y_abnormal,
		                                                                            random_state=random_state)

	else:
		n_abormal_max_test = int(
			(len(y_abnormal) * 0.7))  # the left abnormal data (0.3) used for choosing random validation sets
		X_abnormal, X_abnormal_rest, y_abnormal, y_abnormal_rest = train_test_split(X_abnormal, y_abnormal,
		                                                                            train_size=n_abormal_max_test,
		                                                                            stratify=y_abnormal,
		                                                                            random_state=random_state)

	# get test set first
	X_normal, X_test, y_normal, y_test = train_test_split(X_normal, y_normal, test_size=len(y_abnormal),
	                                                      shuffle=shuffle, random_state=random_state)
	X_test = np.concatenate([X_test, X_abnormal], axis=0)
	y_test = np.concatenate([y_test, y_abnormal], axis=0)

	return X_test, y_test, X_abnormal_rest, y_abnormal_rest, X_normal, y_normal


def get_train_val(X_normal, y_normal, X_abnormal, y_abnormal, val_size=100, shuffle=True, random_state=42):
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

	n_normal_max_train = 10000
	X_train = X_normal[:n_normal_max_train]
	y_train = y_normal[:n_normal_max_train]
	return X_train, y_train, X_val, y_val


def split_train_val_test(X, y, shuffle=True, random_state=42):
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
	# split normal and abnormal
	X_normal = X[y == 0]
	y_normal = y[y == 0]
	X_abnormal = X[y == 1]
	y_abnormal = y[y == 1]
	# if len(y_normal) > 20000:
	# 	X_normal, _, y_normal, _ = train_test_split(X_normal, y_normal, train_size= 20000,
	# 	                                                stratify=y_normal, random_state=random_state)
	if len(y_abnormal) > 4000:
		X_abnormal, _, y_abnormal, _ = train_test_split(X_abnormal, y_abnormal, train_size=4000,
		                                                stratify=y_abnormal, random_state=random_state)

	# get test set first
	X_normal, X_test, y_normal, y_test = train_test_split(X_normal, y_normal, test_size=len(y_abnormal),
	                                                      shuffle=shuffle, random_state=random_state)
	X_test = np.concatenate([X_test, X_abnormal], axis=0)
	y_test = np.concatenate([y_test, y_abnormal], axis=0)

	# select a part of val_data from test_set.
	X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=int(len(y_test) * 0.2),
	                                                stratify=y_test, random_state=random_state)
	X_train = X_normal[:10000]
	y_train = y_normal[:10000]
	return X_train, y_train, X_val, y_val, X_test, y_test


def normalize(X_train, y_train, X_val, y_val, X_test, y_test):
	""" Normalize data

	Parameters
	----------
	X_train
	y_train
	X_val
	y_val
	X_test
	y_test

	Returns
	-------

	"""
	ss = StandardScaler()
	ss.fit(X_train)

	X_train = ss.transform(X_train)
	X_val = ss.transform(X_val)
	X_test = ss.transform(X_test)
	return ss, X_train, y_train, X_val, y_val, X_test, y_test


def short_parmas(params):
	return str(params)[:10]
