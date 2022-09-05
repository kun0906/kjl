""" UNB class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os.path as pth

from examples.offline._constants import *
from examples.offline.datasets._base import Base
from examples.offline.datasets._generate import keep_ip, merge_csvs, keep_csv_ip, \
	_pcap2fullflows
from kjl.utils.tool import load, remove_file, dump, check_path


def _get_unb_flows(original_dir, out_dir, data_name, direction):
	##############################################################################################################
	# step 1: get path
	# get label_file
	in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
		'Friday-WorkingHours-Morning.pcap_ISCX.csv',
		'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
		'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
	raw_label_file = pth.join(out_dir, direction, data_name, 'Friday_labels.csv')
	merge_csvs(in_files, raw_label_file)

	if data_name == 'UNB/CICIDS_2017/pc_192.168.10.5':
		# normal and abormal are mixed together
		pcap_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.5.pcap')
		label_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.5.csv')
		if not os.path.exists(pcap_file) or not os.path.exists(label_file):
			in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
			keep_ip(in_file, out_file=pcap_file, kept_ips=['192.168.10.5'], direction=direction)
			# label_file
			keep_csv_ip(raw_label_file, label_file, ips=['192.168.10.5'], direction=direction, keep_original=True,
			            verbose=10)
	elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.8':
		# normal and abormal are mixed together
		pcap_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.8.pcap')
		label_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.8.csv')
		if not os.path.exists(pcap_file) or not os.path.exists(label_file):
			in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
			keep_ip(in_file, out_file=pcap_file, kept_ips=['192.168.10.8'], direction=direction)
			# label_file
			keep_csv_ip(raw_label_file, label_file, ips=['192.168.10.8'], direction=direction, keep_original=True,
			            verbose=10)

	elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.9':
		# normal and abormal are mixed together
		pcap_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.9.pcap')
		label_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.9.csv')
		if not os.path.exists(pcap_file) or not os.path.exists(label_file):
			in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
			keep_ip(in_file, out_file=pcap_file, kept_ips=['192.168.10.9'], direction=direction)
			# label_file
			keep_csv_ip(raw_label_file, label_file, ips=['192.168.10.9'], direction=direction, keep_original=True,
			            verbose=10)

	elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.14':
		# normal and abormal are mixed together
		pcap_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.14.pcap')
		label_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.14.csv')
		if not os.path.exists(pcap_file) or not os.path.exists(label_file):
			in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
			keep_ip(in_file, out_file=pcap_file, kept_ips=['192.168.10.14'], direction=direction)
			keep_csv_ip(raw_label_file, label_file, ips=['192.168.10.14'], direction=direction, keep_original=True,
			            verbose=10)

	elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.15':
		# normal and abormal are mixed together
		pcap_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.15.pcap')
		label_file = pth.join(out_dir, direction, data_name, 'pc_192.168.10.15.csv')
		if not os.path.exists(pcap_file) or not os.path.exists(label_file):
			in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
			keep_ip(in_file, out_file=pcap_file, kept_ips=['192.168.10.15'], direction=direction)
			# label_file
			keep_csv_ip(raw_label_file, label_file, ips=['192.168.10.15'], direction=direction, keep_original=True,
			            verbose=10)
	else:
		msg = f'{data_name} does not found.'
		raise ValueError(msg)

	##############################################################################################################
	# step 2:  pcap 2 flows
	normal_flows, normal_labels, abnormal_flows, abnormal_labels = _pcap2fullflows(pcap_file=pcap_file,
	                                                                               label_file=label_file)

	normal_file = os.path.join(out_dir, direction, data_name, 'normal_flows_labels.dat')
	check_path(normal_file)
	dump((normal_flows, normal_labels), out_file=normal_file)

	abnormal_file = os.path.join(out_dir, direction, data_name, 'abnormal_flows_labels.dat')
	check_path(abnormal_file)
	dump((abnormal_flows, abnormal_labels), out_file=abnormal_file)

	return normal_file, abnormal_file


def get_unb_flows(original_dir='../Datasets',
                  out_dir='examples/offline/out',
                  data_name='',
                  direction='src_dst',
                  ):
	lg.debug(get_unb_flows.__dict__)
	if data_name == 'UNB3_345':
		subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.9',
		               'UNB/CICIDS_2017/pc_192.168.10.14',
		               'UNB/CICIDS_2017/pc_192.168.10.15',)  # each_data has normal and abnormal
	else:
		msg = f'{data_name}'
		raise NotImplementedError(msg)

	# get normal and abnormal (not subflows)
	normal_files = []
	abnormal_files = []
	for data_name in subdatasets:
		normal, abnormal = _get_unb_flows(original_dir, out_dir, data_name, direction)
		normal_files.append(normal)
		abnormal_files.append(abnormal)

	# only use UNB(PC3) as normal flows
	normal_files = [normal_files[0]]

	return normal_files, abnormal_files


class UNB(Base):

	def __init__(self, dataset_name='UNB',
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
			normal_files, abnormal_files = get_unb_flows(original_dir=ORIG_DIR,
			                                             out_dir=self.out_dir,
			                                             data_name=self.dataset_name,
			                                             direction=self.flow_direction)
			meta = self.flows2features(normal_files, abnormal_files, q_interval=self.q_interval)
			lg.debug(f'meta: {meta.keys()}')
		self.X, self.y = meta['X'], meta['y']

		return self.X, self.y
