""" UCHI class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import glob
import os
import os.path as pth
import subprocess

from loguru import logger as lg
from odet.pparser.parser import keep_mac_address, _pcap2flows

from examples.offline._constants import OUT_DIR, ORIG_DIR
from examples.offline.datasets._base import Base
from examples.offline.datasets._generate import keep_ip
from kjl.datasets.uchicago import split_by_activity
from kjl.utils.tool import load, remove_file, check_path, dump


def keep_mac_address(pcap_file, kept_ips=[], out_file='', direction='src'):
	if out_file == '':
		out_file = os.path.splitext(pcap_file)[0] + 'kept_mac.pcap'  # Split a path in root and extension.

	if direction == 'src':
		# filter by mac srcIP address
		srcIP_str = " or ".join([f'eth.src=={srcIP}' for srcIP in kept_ips])
		cmd = f"tshark -r \"{pcap_file}\" -w \"{out_file}\" {srcIP_str}"
	elif direction == 'src_dst':
		# filter by mac srcIP address
		srcIP_str = " or ".join([f'eth.addr=={srcIP}' for srcIP in kept_ips])
		cmd = f"tshark -r \"{pcap_file}\" -w \"{out_file}\" {srcIP_str}"
	else:
		msg = f'{direction}'
		raise NotImplementedError(msg)

	print(f'{cmd}')
	try:
		result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
	except Exception as e:
		print(f'{e}, {result}')
		return -1

	return out_file


def get_iot2021_flows(in_dir=f'../Datasets/UCHI/IOT_2021/data-clean/refrigerator',
                      out_dir='',
                      direction='src'):
	""" Hard coding in_dir and pcap paths
	# extract from data-clean.zip (collected at 2021 for human activity recognition: contains pcap and videos)
	Note:
		1) refrigerator IP changes over time (dynamic ip), so here we filter with mac address.
		2) please don't merge all pcaps first and then obtain flows.
	Parameters
	----------
	in_dir
	direction
	Returns
	-------
	"""
	ip2device = {'192.168.143.152': 'refrigerator', }
	device2ip = {'refrigerator': '192.168.143.43', 'nestcam': '192.168.143.104', 'alexa': '192.168.143.74'}
	# #
	device2mac = {'refrigerator': '70:2c:1f:39:25:6e', 'nestcam': '18:b4:30:8a:9f:b2',
	              'alexa': '4c:ef:c0:0b:91:b3'}
	normal_pcaps = list(glob.iglob(in_dir + '/no_interaction/**/*.' + 'pcap', recursive=True))
	normal_pcaps.append(in_dir + '/idle_1629935923.pcap')
	normal_pcaps.append(in_dir + '/idle_1630275254.pcap')
	normal_pcaps = sorted(normal_pcaps)
	normal_flows = []
	for f in normal_pcaps:
		filter_f = f'{out_dir}/~tmp.pcap'
		check_path(filter_f)
		keep_mac_address(f, kept_ips=[device2mac['refrigerator']], out_file=filter_f, direction=direction)
		flows = _pcap2flows(filter_f, verbose=10)  # normal  flows
		normal_flows.extend(flows)
	lg.debug(f'total normal pcaps: {len(normal_pcaps)} and total flows: {len(normal_flows)}')

	# get abnormal flows
	abnormal_pcaps = list(glob.iglob(in_dir + '/open_close_fridge/**/*.' + 'pcap', recursive=True)) + \
	                 list(glob.iglob(in_dir + '/put_back_item/**/*.' + 'pcap', recursive=True)) + \
	                 list(glob.iglob(in_dir + '/screen_interaction/**/*.' + 'pcap', recursive=True)) + \
	                 list(glob.iglob(in_dir + '/take_out_item/**/*.' + 'pcap', recursive=True))
	abnormal_pcaps = sorted(abnormal_pcaps)

	abnormal_flows = []
	for f in abnormal_pcaps:
		filter_f = f'{out_dir}/~tmp.pcap'
		check_path(filter_f)
		keep_mac_address(f, kept_ips=[device2mac['refrigerator']], out_file=filter_f, direction=direction)
		flows = _pcap2flows(filter_f, verbose=10)  # normal  flows
		abnormal_flows.extend(flows)
	lg.debug(f'total abnormal pcaps: {len(abnormal_pcaps)} and total flows: {len(abnormal_flows)}')

	meta = {'normal_flows': normal_flows, 'abnormal_flows': abnormal_flows,
	        'normal_pcaps': normal_pcaps, 'abnormal_pcaps': abnormal_pcaps,
	        'device2mac': device2mac, 'filter_mac': device2mac['refrigerator'],
	        'direction': direction, 'in_dir': in_dir}
	return meta


def _get_uchi_flows(original_dir, out_dir, data_name, direction):
	if data_name == 'UCHI/IOT_2020/aecho_192.168.143.74':
		# normal
		normal_flows = []
		tmp_dir = pth.join(out_dir, direction, data_name)
		for v in sorted(os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle'))):
			if not v.startswith('.') and v.endswith('.pcap'):
				filter_f = os.path.join(tmp_dir, f'filtered/{v}.pcap')
				# filter pcap
				check_path(filter_f)
				keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
				        out_file=filter_f, kept_ips=['192.168.143.74'], direction=direction)
				# extract flows from pcap
				flows = _pcap2flows(filter_f, verbose=10)  # normal flows
				normal_flows.extend(flows)

		# abnormal
		# Can not use the whole abnormal pcap directly because when we split it to subpcap,
		# one flow will split to multi-flows.
		activity = 'shop'
		whole_abnormal = pth.join(original_dir, data_name, f'echo_{activity}.pcap')
		num = split_by_activity(whole_abnormal, out_dir=os.path.dirname(whole_abnormal), activity=activity)
		tmp_dir = pth.join(out_dir, direction, data_name)
		abnormal_flows = []
		for i in range(num):
			v = f'{activity}/capture{i}.seq/deeplens_{activity}_{i}.pcap'
			if not v.startswith('.') and v.endswith('.pcap'):
				filter_f = pth.join(tmp_dir, v)
				# filter pcap
				check_path(filter_f)
				keep_ip(pth.join(original_dir, data_name, v),
				        out_file=filter_f, kept_ips=['192.168.143.74'], direction=direction)
				# extract flows from pcaps
				flows = _pcap2flows(filter_f, verbose=10)  # normal flows
				abnormal_flows.extend(flows)

	elif data_name == 'UCHI/IOT_2020/sfrig_192.168.143.43':
		# normal
		normal_flows = []
		tmp_dir = pth.join(out_dir, direction, data_name)
		for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')):
			if not v.startswith('.') and v.endswith('.pcap'):
				filter_f = os.path.join(tmp_dir, f'filtered/{v}.pcap')
				# filter pcap
				check_path(filter_f)
				keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
				        out_file=filter_f,
				        kept_ips=['192.168.143.43'], direction=direction)
				# extract flows from pcaps
				flows = _pcap2flows(filter_f, verbose=10)  # normal flows
				normal_flows.extend(flows)

		# abnormal
		abnormal_flows = []
		num = 9
		tmp_dir = pth.join(out_dir, direction, data_name)
		for i in range(num):
			v = f'open_shut/capture{i}.seq/deeplens_open_shut_fridge_batch_{i}.pcap_filtered.pcap'
			filter_f = pth.join(tmp_dir, v)
			# filter pcap
			check_path(filter_f)
			keep_ip(pth.join(original_dir, data_name, v),
			        out_file=filter_f,
			        kept_ips=['192.168.143.43'], direction=direction)
			# extract flows from pcaps
			flows = _pcap2flows(filter_f, verbose=10)  # normal flows
			abnormal_flows.extend(flows)

	elif data_name == 'UCHI/IOT_2021/sfrig_mac_70:2c:1f:39:25:6e':
		# IP = 'mac_70:2c:1f:39:25:6e'  # IP for the new data changes over time, so here use mac address instead
		# hard coding (is not a good idea)
		# # extract from data-clean.zip (collected at 2021 for human activity recognition: contains pcap and videos)
		meta = get_iot2021_flows(in_dir=f'{original_dir}/UCHI/IOT_2021/data-clean/refrigerator',
		                         out_dir=out_dir,
		                         direction=direction)
		normal_flows = meta['normal_flows']
		abnormal_flows = meta['abnormal_flows']

	elif data_name == 'UCHI/IOT_2020/wshr_192.168.143.100':
		# normal
		normal_flows = []
		tmp_dir = pth.join(out_dir, direction, data_name)
		for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')):
			if not v.startswith('.') and v.endswith('.pcap'):
				filter_f = os.path.join(tmp_dir, f'filtered/{v}.pcap')
				# filter pcap
				check_path(filter_f)
				keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
				        out_file=filter_f,
				        kept_ips=['192.168.143.100'], direction=direction)
				# extract flows from pcaps
				flows = _pcap2flows(filter_f, verbose=10)  # normal flows
				normal_flows.extend(flows)

		# abnormal
		tmp_dir = pth.join(out_dir, direction, data_name)
		abnormal_flows = []
		num = 31
		for i in range(num):
			v = f'open_wshr/capture{i}.seq/deeplens_open_washer_{i}.pcap'
			filter_f = pth.join(tmp_dir, v)
			# filter pcap
			check_path(filter_f)
			keep_ip(pth.join(original_dir, data_name, v),
			        out_file=filter_f,
			        kept_ips=['192.168.143.43'], direction=direction)
			# extract flows from pcaps
			flows = _pcap2flows(filter_f, verbose=10)  # normal flows
			abnormal_flows.extend(flows)

	elif data_name == 'UCHI/IOT_2020/dwshr_192.168.143.76':
		# normal
		tmp_dir = pth.join(out_dir, direction, data_name)
		normal_flows = []
		for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')):
			if not v.startswith('.') and v.endswith('.pcap'):
				filter_f = os.path.join(tmp_dir, f'filtered/{v}.pcap')
				# filter pcap
				check_path(filter_f)
				keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
				        out_file=filter_f,
				        kept_ips=['192.168.143.76'], direction=direction)
				# extract flows from pcaps
				flows = _pcap2flows(filter_f, verbose=10)  # normal flows
				normal_flows.extend(flows)

		# abnormal
		tmp_dir = pth.join(out_dir, direction, data_name)
		abnormal_flows = []
		num = 31
		for i in range(num):
			v = f'open_dwshr/capture{i}.seq/deeplens_open_dishwasher_{i}.pcap_filtered.pcap'
			filter_f = pth.join(tmp_dir, v)
			# filter pcap
			check_path(filter_f)
			keep_ip(pth.join(original_dir, data_name, v),
			        out_file=filter_f,
			        kept_ips=['192.168.143.76'], direction=direction)
			# extract flows from pcaps
			flows = _pcap2flows(filter_f, verbose=10)  # normal flows
			abnormal_flows.extend(flows)

	# elif data_name == 'UCHI/IOT_2020/ghome_192.168.143.20':
	# 	# normal and abormal are independent
	# 	pth_normal = pth.join(out_dir, data_name, 'google_home-2daysactiv-src_192.168.143.20-normal.pcap')
	# 	pth_abnormal = pth.join(out_dir, data_name,
	# 	                        'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap')
	else:
		msg = f'{data_name}'
		raise NotImplementedError(msg)

	##############################################################################################################
	# step 2: pcap 2 flows

	normal_file = os.path.join(out_dir, direction, data_name, 'normal_flows_labels.dat')
	check_path(normal_file)
	normal_labels = ['normal'] * len(normal_flows)
	dump((normal_flows, normal_labels), out_file=normal_file)

	abnormal_file = os.path.join(out_dir, direction, data_name, 'abnormal_flows_labels.dat')
	check_path(abnormal_file)
	abnormal_labels = ['abnormal'] * len(abnormal_flows)
	dump((abnormal_flows, abnormal_labels), out_file=abnormal_file)

	return normal_file, abnormal_file


def get_uchi_flows(original_dir='../Datasets',
                   out_dir='examples/offline/out',
                   data_name='',
                   direction='src_dst',
                   ):
	lg.debug(get_uchi_flows.__dict__)
	if data_name == 'SFRIG1_2020':  # normal >10000, abnormal: ~900
		subdatasets = ('UCHI/IOT_2020/sfrig_192.168.143.43',)
	elif data_name == 'SFRIG1_2021':  # normal >10000, abnormal: ~10000
		# extract from data-clean.zip (collected at 2021 for human activity recognition: contains pcap and videos)
		subdatasets = ('UCHI/IOT_2021/sfrig_mac_70:2c:1f:39:25:6e',)
	elif data_name == 'AECHO1_2020':  # normal >10000, abnormal: ~800
		subdatasets = ('UCHI/IOT_2020/aecho_192.168.143.74',)
	elif data_name == 'WSHR_2020':  # normal ~4000, abnormal: ~80
		subdatasets = ('UCHI/IOT_2020/wshr_192.168.143.100',)
	elif data_name == 'DWSHR_2020':  # normal > 10000, abnormal: ~102
		subdatasets = ('UCHI/IOT_2020/dwshr_192.168.143.76',)
	elif data_name == 'DWSHR_WSHR_2020':  # only use DWSHR normal data
		subdatasets = ('UCHI/IOT_2020/dwshr_192.168.143.76', 'UCHI/IOT_2020/wshr_192.168.143.100')
		# get normal and abnormal (not subflows)
		normal_files = []
		abnormal_files = []
		for data_name in subdatasets:
			normal, abnormal = _get_uchi_flows(original_dir, out_dir, data_name, direction)
			if data_name == 'UCHI/IOT_2020/dwshr_192.168.143.76':
				normal_files.append(normal)
			abnormal_files.append(abnormal)

		return normal_files, abnormal_files
	elif data_name == 'DWSHR_WSHR_AECHO_2020':
		subdatasets = ('UCHI/IOT_2020/dwshr_192.168.143.76', 'UCHI/IOT_2020/wshr_192.168.143.100',
		               'UCHI/IOT_2020/aecho_192.168.143.74')
		# get normal and abnormal (not subflows)
		normal_files = []
		abnormal_files = []
		for data_name in subdatasets:
			normal, abnormal = _get_uchi_flows(original_dir, out_dir, data_name, direction)
			if data_name == 'UCHI/IOT_2020/dwshr_192.168.143.76':  # only use DWSHR normal data
				normal_files.append(normal)
			abnormal_files.append(abnormal)

		return normal_files, abnormal_files
	elif data_name == 'DWSHR_AECHO_2020':
		subdatasets = ('UCHI/IOT_2020/dwshr_192.168.143.76',
		               'UCHI/IOT_2020/aecho_192.168.143.74')
		# get normal and abnormal (not subflows)
		normal_files = []
		abnormal_files = []
		for data_name in subdatasets:
			normal, abnormal = _get_uchi_flows(original_dir, out_dir, data_name, direction)
			if data_name == 'UCHI/IOT_2020/dwshr_192.168.143.76':  # only use DWSHR normal data
				normal_files.append(normal)
			abnormal_files.append(abnormal)

		return normal_files, abnormal_files
	else:
		msg = f'{data_name}'
		raise NotImplementedError(msg)

	# get normal and abnormal (not subflows)
	normal_files = []
	abnormal_files = []
	for data_name in subdatasets:
		normal, abnormal = _get_uchi_flows(original_dir, out_dir, data_name, direction)
		normal_files.append(normal)
		abnormal_files.append(abnormal)

	return normal_files, abnormal_files


class UCHI(Base):

	def __init__(self, dataset_name='CTU1',
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
			normal_files, abnormal_files = get_uchi_flows(original_dir=ORIG_DIR,
			                                              out_dir=self.out_dir,
			                                              data_name=self.dataset_name,
			                                              direction=self.flow_direction)
			meta = self.flows2features(normal_files, abnormal_files, q_interval=self.q_interval)
			lg.debug(f'meta: {meta.keys()}')
		self.X, self.y = meta['X'], meta['y']

		return self.X, self.y
