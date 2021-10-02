"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
from kjl.utils.tool import dump

RANDOM_STATE = 42

import os
from odet.pparser.parser import PCAP


class FEATURES(PCAP):

	def __init__(self, pcap_file='data/misc.pcap', label_file='misc.csv', label=None,
	             feat_type='IAT_SIZE',
	             interval=0, q_interval=0.9,
	             fft=False, header=False, out_dir='.', flow_pkts_thres=2,
	             verbose=10, random_state=100):
		if not os.path.exists(pcap_file): print(f'{pcap_file} does not exist.')

		(super, FEATURES).__init__(pcap_file=pcap_file)

		self.pcap_file = pcap_file
		self.label_file = label_file
		self.label = label
		self.feat_type = feat_type
		self.q_interval = q_interval
		self.interval = interval
		self.fft = fft
		self.header = header
		self.out_dir = out_dir
		self.flow_pkts_thres = flow_pkts_thres
		self.verbose = verbose
		self.random_state = random_state

	def pcap2features(self):
		# extract flows from pcap
		self.pcap2flows()
		# label each flow with a file or label
		self.label_flows(label_file=self.label_file, label=self.label)
		out_file = f'{self.out_dir}/flows.dat'
		print('raw_flows+labels: ', out_file)
		dump((self.flows, self.labels), out_file)

		self.flows2subflows(interval=self.interval, q_interval=self.q_interval)
		out_file = f'{self.out_dir}/subflows-q_interval:{self.q_interval}.dat'
		print('subflows+labels: ', out_file)
		dump((self.flows, self.labels), out_file)

		# extract features from each flow given feat_type
		self.flow2features(self.feat_type, fft=self.fft, header=self.header)
		out_file = f'{self.out_dir}/features-q_interval:{self.q_interval}.dat'
		print('features+labels: ', out_file)
		dump((self.features, self.labels), out_file)

		print(self.features.shape, self.pcap2flows.tot_time, self.flows2subflows.tot_time, self.flow2features.tot_time)

		return self.features
