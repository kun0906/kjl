""" MACCDC class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os
import os.path as pth

from loguru import logger as lg

from examples.offline._constants import ORIG_DIR, OUT_DIR
from examples.offline.datasets._base import Base
from examples.offline.datasets._generate import keep_ip, _pcap2fullflows
from kjl.utils.tool import load, remove_file, check_path, dump


def _get_maccdc_flows(original_dir, out_dir, data_name, direction):
	if data_name == 'MACCDC/MACCDC_2012/pc_192.168.202.79':
		# normal and abormal are independent
		# pth_normal = pth.join(in_dir, data_name, 'maccdc2012_00000-srcIP_192.168.229.153.pcap')
		# the result does beat OCSVM.
		pth_normal = pth.join(out_dir, direction, data_name, 'maccdc2012_00000-pc_192.168.202.79.pcap')
		pth_abnormal = pth.join(out_dir, direction, data_name, 'maccdc2012_00000-pc_192.168.202.76.pcap')
		if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
			# normal
			in_file = pth.join(original_dir, 'MACCDC/MACCDC_2012', 'maccdc2012_00000.pcap')
			keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.202.79'], direction=direction)
			# abnormal
			in_file = pth.join(original_dir, 'MACCDC/MACCDC_2012', 'maccdc2012_00000.pcap')
			keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.202.76'], direction=direction)
	else:
		msg = f'{data_name}'
		raise NotImplementedError(msg)

	##############################################################################################################
	# step 2: pcap 2 flows
	normal_flows, normal_labels, _, _ = _pcap2fullflows(pcap_file=pth_normal,
	                                                    label_file=None, label='normal')
	_, _, abnormal_flows, abnormal_labels = _pcap2fullflows(pcap_file=pth_abnormal,
	                                                        label_file=None, label='abnormal')

	normal_file = os.path.join(out_dir, direction, data_name, 'normal_flows_labels.dat')
	check_path(normal_file)
	dump((normal_flows, normal_labels), out_file=normal_file)

	abnormal_file = os.path.join(out_dir, direction, data_name, 'abnormal_flows_labels.dat')
	check_path(abnormal_file)
	dump((abnormal_flows, abnormal_labels), out_file=abnormal_file)

	return normal_file, abnormal_file


def get_maccdc_flows(original_dir='../Datasets',
                     out_dir='examples/offline/out',
                     data_name='',
                     direction='src_dst',
                     ):
	lg.debug(get_maccdc_flows.__dict__)

	if data_name == 'MACCDC1':
		subdatasets = ('MACCDC/MACCDC_2012/pc_192.168.202.79',)
	else:
		msg = f'{data_name}'
		raise NotImplementedError(msg)

	# get normal and abnormal (not subflows)
	normal_files = []
	abnormal_files = []
	for data_name in subdatasets:
		normal, abnormal = _get_maccdc_flows(original_dir, out_dir, data_name, direction)
		normal_files.append(normal)
		abnormal_files.append(abnormal)

	return normal_files, abnormal_files


class MACCDC(Base):

	def __init__(self, dataset_name='',
	             out_dir=OUT_DIR, feature_name='', flow_direction='src_dst',
	             q_interval=0.9, header=False, verbose=0,
	             overwrite=False, random_state=42):
		self.X = None
		self.y = None
		self.overwrite = overwrite
		self.out_dir = out_dir
		self.feature_name = feature_name
		self.dataset_name = dataset_name
		self.flow_direction = flow_direction
		self.q_interval = q_interval
		self.header = header
		self.random_state = random_state
		self.verbose = verbose

		self.Xy_file = os.path.join(self.out_dir, self.flow_direction, self.dataset_name, self.feature_name,
		                            f'header_{self.header}', 'Xy.dat')
		lg.info(f'{self.Xy_file}')

	def generate(self):
		if self.overwrite:
			remove_file(self.Xy_file)
		else:
			pass

		if os.path.exists(self.Xy_file):
			meta = load(self.Xy_file)
		else:
			normal_files, abnormal_files = get_maccdc_flows(original_dir=ORIG_DIR,
			                                                out_dir=self.out_dir,
			                                                data_name=self.dataset_name,
			                                                direction=self.flow_direction)
			meta = self.flows2features(normal_files, abnormal_files, q_interval=self.q_interval)
			lg.debug(f'meta: {meta.keys()}')
		self.X, self.y = meta['X'], meta['y']

		return self.X, self.y
