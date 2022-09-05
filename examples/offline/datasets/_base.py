""" Base class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os
from collections import Counter

import numpy as np
import sklearn
from loguru import logger as lg
from odet.pparser.parser import _get_flow_duration, _get_split_interval, _flows2subflows, _get_IAT_SIZE, _pcap2flows

from examples.offline.datasets._generate import _subflows2featutes
from kjl.utils.tool import check_path, load, dump, time_func, data_info, save2txt, remove_file


class Base:

	def __init__(self, Xy_file=None, feature_name='IAT+SIZE', header=False, verbose=0):
		self.Xy_file = Xy_file
		self.feature_name = feature_name
		self.header = header
		self.verbose = verbose

	def pcap2flows(self, pcap_file=None):
		return _pcap2flows(pcap_file, flow_pkts_thres=2)

	def flow2subflows(self, flows=None, interval=None, labels=None):
		return _flows2subflows(flows, interval=interval, labels=labels)

	def flow2features(self, flows=None, name='IAT+SIZE'):
		if name == 'IAT+SIZE':
			features, fids = _get_IAT_SIZE(flows)
		else:
			msg = f'{name}'
			raise NotImplementedError(msg)
		return features, fids

	def fix_feature(self, features, dim=None):
		# fix each flow to the same feature dimension (cut off the flow or append 0 to it)
		features = [v[:dim] if len(v) > dim else v + [0] * (dim - len(v)) for v in features]
		return np.asarray(features)

	def generate(self):
		q_interval = 0.9
		# pcap to flows
		flows = self.pcap2flows(self.pcap_file)

		# flows to subflow
		labels = [1] * len(flows)
		durations = [_get_flow_duration(pkts) for fid, pkts in flows]
		interval = _get_split_interval(durations, q_interval=q_interval)
		subflows, labels = self.flow2subflows(flows, interval=interval, labels=labels)

		# get dimension
		normal_flows = subflows
		num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
		dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
		lg.info(f'dim={dim}')

		# flows to features
		features, fids = self.flow2features(subflows, name=self.feature_name)

		# fixed the feature size
		features = self.fix_feature(features, dim=dim)

		self.X = features
		self.y = np.asarray([0] * len(features))

		# save data to disk
		check_path(os.path.dirname(self.Xy_file))
		dump((self.X, self.y), out_file=self.Xy_file)
		msg = 'Base.generate()'
		raise NotImplementedError(msg)

	def flows2features(self, normal_files, abnormal_files, q_interval=0.9):
		lg.debug(f'normal_files: {normal_files}')
		lg.debug(f'abnormal_files: {abnormal_files}')
		normal_durations = []
		normal_flows = []
		normal_labels = []
		for i, f in enumerate(normal_files):
			(flows, labels), load_time = time_func(load, f)
			normal_flows.extend(flows)
			lg.debug(f'i: {i}, load_time: {load_time} s.')
			normal_labels.extend([f'normal_{i}'] * len(labels))
			data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in flows]).reshape(-1, 1),
			          name=f'durations_{i}')
			normal_durations.extend([_get_flow_duration(pkts) for fid, pkts in flows])

		# 1. get interval from all normal flows
		data_info(np.asarray(normal_durations).reshape(-1, 1), name='durations')
		interval = _get_split_interval(normal_durations, q_interval=q_interval)
		lg.debug(f'interval {interval} when q_interval: {q_interval}')

		abnormal_flows = []
		abnormal_labels = []
		for i, f in enumerate(abnormal_files):
			flows, labels = load(f)
			abnormal_flows.extend(flows)
			abnormal_labels.extend([f'abnormal_{i}'] * len(labels))
		lg.debug(f'fullflows: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')

		self.raw_normal_flows = normal_flows
		self.raw_abnormal_flows = abnormal_flows

		# 2. flows2subflows
		# flow_durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
		normal_flows, normal_labels = _flows2subflows(normal_flows, interval=interval, labels=normal_labels,
		                                              flow_pkts_thres=2,
		                                              verbose=1)
		abnormal_flows, abnormal_labels = _flows2subflows(abnormal_flows, interval=interval, labels=abnormal_labels,
		                                                  flow_pkts_thres=2, verbose=1)
		lg.debug(f'subflows: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')

		# 3. subflows2features
		num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
		# dim is for SIZE features
		dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension

		if self.feature_name.startswith('SAMP'):
			X = {}
			y = {}
			for q_samp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
				# get sampling_rate on normal_flows first
				# lg.debug(f'np.quantile(flows_durations): {np.quantile(flow_durations, q=[0.1, 0.2, 0.3, 0.9, 0.95])}')
				sampling_rate = _get_split_interval(normal_durations, q_interval=q_samp)
				if sampling_rate <= 0.0: continue
				X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
				                                        feat_type=self.feature_name, sampling_rate=sampling_rate,
				                                        header=self.header, verbose=self.verbose)

				X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
				                                            feat_type=self.feature_name, sampling_rate=sampling_rate,
				                                            header=self.header,
				                                            verbose=self.verbose)
				lg.debug(
					f'q_samp: {q_samp}, subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
				self.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}

				X[q_samp] = np.concatenate([X_normal, X_abnormal], axis=0)
				y[q_samp] = np.concatenate([y_normal, y_abnormal], axis=0)

		else:
			X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
			                                        feat_type=self.feature_name, header=self.header,
			                                        verbose=self.verbose)
			X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
			                                            feat_type=self.feature_name, header=self.header,
			                                            verbose=self.verbose)
			lg.debug(
				f'subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
			self.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
			X = np.concatenate([X_normal, X_abnormal], axis=0)
			y = np.concatenate([y_normal, y_abnormal], axis=0)

		meta = {'normal_files': normal_files, 'abnormal_files': abnormal_files,  # raw flows
		        'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,  # subflows
		        'X': X, 'y': y, 'dim': dim,
		        'q_interval': q_interval}

		lg.debug(f'Xy_file: {self.Xy_file}')
		if self.overwrite:
			remove_file(self.Xy_file)
		check_path(self.Xy_file)
		dump(meta, out_file=self.Xy_file)
		if 'SAMP' not in self.feature_name:
			save2txt([list(x_) + list(y_) for x_, y_ in zip(X, y)], out_file=os.path.splitext(self.Xy_file)[0] + '.txt')

		lg.debug(f'Xy_file: {self.Xy_file}')

		return meta

	def _flows2features_seperate(self, normal_files, abnormal_files, q_interval=0.9):
		""" dataset1 and dataset2 use different interval and will get different dimension
			then append 0 to the smaller dimension to make both has the same dimension

		Parameters
		----------
		normal_files
		abnormal_files
		q_interval

		Returns
		-------

		"""

		lg.debug(f'normal_files: {normal_files}')
		lg.debug(f'abnormal_files: {abnormal_files}')

		X = []
		y = []
		for i, (f1, f2) in enumerate(zip(normal_files, abnormal_files)):
			(normal_flows, labels), load_time = time_func(load, f1)
			normal_labels = [f'normal_{i}'] * len(labels)
			lg.debug(f'i: {i}, load_time: {load_time} s.')

			# 1. get durations
			data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in normal_flows]).reshape(-1, 1),
			          name=f'durations_{i}')
			durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]

			interval = _get_split_interval(durations, q_interval=q_interval)
			lg.debug(f'interval {interval} when q_interval: {q_interval}')

			# 2. flows2subflows
			normal_flows, normal_labels = _flows2subflows(normal_flows, interval=interval, labels=normal_labels,
			                                              flow_pkts_thres=2,
			                                              verbose=1)
			# 3. subflows2features
			num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
			data_info(np.asarray(num_pkts).reshape(-1, 1), name='num_ptks_for_flows')
			dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
			lg.debug(f'i: {i}, dim: {dim}')
			X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
			                                        verbose=self.verbose)
			n_samples = 15000
			if len(y_normal) > n_samples:
				X_normal, y_normal = sklearn.utils.resample(X_normal, y_normal, n_samples=n_samples, replace=False,
				                                            random_state=42)
			else:
				X_normal, y_normal = sklearn.utils.resample(X_normal, y_normal, n_samples=60000, replace=True,
				                                            random_state=42)
			X.extend(X_normal.tolist())
			y.extend(y_normal)

			# for abnormal flows
			(abnormal_flows, labels), load_time = time_func(load, f2)
			abnormal_labels = [f'abnormal_{i}'] * len(labels)
			abnormal_flows, abnormal_labels = _flows2subflows(abnormal_flows, interval=interval,
			                                                  labels=abnormal_labels,
			                                                  flow_pkts_thres=2, verbose=1)
			X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
			                                            verbose=self.verbose)
			n_samples = 15000
			if len(y_abnormal) > n_samples:
				X_abnormal, y_abnormal = sklearn.utils.resample(X_abnormal, y_abnormal, n_samples=n_samples,
				                                                replace=False,
				                                                random_state=42)
			else:  #
				X_abnormal, y_abnormal = sklearn.utils.resample(X_abnormal, y_abnormal, n_samples=200, replace=True,
				                                                random_state=42)

			X.extend(X_abnormal.tolist())
			y.extend(y_abnormal)
			lg.debug(
				f'subflows (before sampling): normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
			lg.debug(
				f'after resampling: normal_labels: {Counter(y_normal)}, abnormal_labels: {Counter(y_abnormal)}')
		# break
		max_dim = max([len(v) for v in X])
		lg.debug(f'===max_dim: {max_dim}')
		new_X = []
		for v in X:
			v = v + (max_dim - len(v)) * [0]
			new_X.append(np.asarray(v, dtype=float))

		X = np.asarray(new_X, dtype=float)
		y = np.asarray(y, dtype=str)
		self.Xy_file = os.path.join(self.out_dir, 'Xy-normal-abnormal.dat')
		dump((X, y), out_file=self.Xy_file)
		lg.debug(f'Xy_file: {self.Xy_file}')
