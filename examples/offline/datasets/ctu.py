""" CTU class for dataset

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os.path as pth

from examples.offline._constants import *
from examples.offline.datasets._base import Base
from examples.offline.datasets._generate import keep_ip, _pcap2fullflows
from kjl.utils.tool import load, remove_file, check_path, dump


def _get_ctu_flows(original_dir, out_dir, data_name, direction):
	if data_name == 'CTU/IOT_2017/pc_192.168.1.196':
		"""
		Datasets: Malware on IoT Dataset
			https://www.stratosphereips.org/datasets-iot
		
		2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap:
			https://mcfp.felk.cvut.cz/publicDatasets/IoTDatasets/CTU-IoT-Malware-Capture-41-1/
		2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap:
			https://mcfp.felk.cvut.cz/publicDatasets/IoTDatasets/CTU-IoT-Malware-Capture-34-1/
		"""
		#
		#
		# normal and abormal are independent
		pth_normal = pth.join(out_dir, direction, data_name,
		                      '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap')
		pth_abnormal = pth.join(out_dir, direction, data_name,
		                        '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
		if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
			# normal
			in_file = pth.join(original_dir, 'CTU/IOT_2017',
			                   'CTU-IoT-Malware-Capture-41-1_2019-01-09-22-46-52-192.168.1.196.pcap')
			keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.1.196'], direction=direction)
			# abnormal
			in_file = pth.join(original_dir, 'CTU/IOT_2017',
			                   'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
			keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)

	elif data_name == 'CTU/IOT_2017/pc_10.0.2.15_192.168.1.195':
		"""
		normal_traffic:
			https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/ (around 1100 flows)
			https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-22/ 
		"""
		# normal and abormal are independent
		pth_normal = pth.join(out_dir, direction, data_name, '2017-04-30_CTU-win-normal.pcap')
		pth_abnormal = pth.join(out_dir, direction, data_name,
		                        '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
		if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
			# normal
			in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-04-30_win-normal.pcap')
			keep_ip(in_file, out_file=pth_normal, kept_ips=['10.0.2.15'], direction=direction)
			# abnormal
			in_file = pth.join(original_dir, 'CTU/IOT_2017',
			                   'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
			keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)

	elif data_name == 'CTU/IOT_2017/pc_10.0.2.15_192.168.1.196':
		"""
					normal_traffic:
						https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/
					"""
		# normal and abormal are independent
		pth_normal = pth.join(out_dir, direction, data_name, '2017-04-30_CTU-win-normal.pcap')
		pth_abnormal = pth.join(out_dir, direction, data_name,
		                        '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap')
		if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
			# normal
			in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-04-30_win-normal.pcap')
			keep_ip(in_file, out_file=pth_normal, kept_ips=['10.0.2.15'], direction=direction)
			# abnormal
			in_file = pth.join(original_dir, 'CTU/IOT_2017',
			                   '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap.pcap')
			keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.196'], direction=direction)


	elif data_name == 'CTU/IOT_2017/pc_192.168.1.191_192.168.1.195':
		"""
		normal_traffic:
			https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/ (around 1100 flows)
			https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-22/ 
		"""
		## editcap -c 500000 2017-05-02_kali.pcap 2017-05-02_kali.pcap

		# normal and abormal are independent
		pth_normal = pth.join(out_dir, direction, data_name, '2017-05-02_CTU-kali-normal.pcap')
		pth_abnormal = pth.join(out_dir, direction, data_name,
		                        '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
		if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
			# normal
			in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-05-02_kali_00000_20170502082205.pcap')
			keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.1.191'], direction=direction)
			# abnormal
			in_file = pth.join(original_dir, 'CTU/IOT_2017',
			                   'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
			keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)


	elif data_name == 'CTU/IOT_2017/pc_192.168.1.191_192.168.1.196':
		"""
		normal_traffic:
			https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/
		"""
		# normal and abormal are independent
		pth_normal = pth.join(out_dir, direction, data_name, '2017-05-02_CTU-kali-normal.pcap')
		pth_abnormal = pth.join(out_dir, direction, data_name,
		                        '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_abnormal.pcap')
		if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
			# normal
			# in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-05-02_kali_00000_20170502072205.pcap')   # for CALUMENT
			in_file = pth.join(original_dir, 'CTU/IOT_2017',
			                   '2017-05-02_kali_00000_20170502082205.pcap')  # for NEON

			keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.1.191'], direction=direction)
			# abnormal
			in_file = pth.join(original_dir, 'CTU/IOT_2017',
			                   'CTU-IoT-Malware-Capture-41-1_2019-01-09-22-46-52-192.168.1.196.pcap')
			keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.196'], direction=direction)

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


def get_ctu_flows(original_dir='../Datasets',
                  out_dir='examples/offline/out',
                  data_name='',
                  direction='src_dst',
                  ):
	lg.debug(get_ctu_flows.__dict__)

	if data_name == 'CTU1':
		subdatasets = ('CTU/IOT_2017/pc_192.168.1.196',)
	elif data_name == 'CTU21':
		subdatasets = ('CTU/IOT_2017/pc_10.0.2.15_192.168.1.195',)
	elif data_name == 'CTU22':
		subdatasets = ('CTU/IOT_2017/pc_10.0.2.15_192.168.1.196',)
	elif data_name == 'CTU31':
		subdatasets = ('CTU/IOT_2017/pc_192.168.1.191_192.168.1.195',)
	elif data_name == 'CTU32':
		subdatasets = ('CTU/IOT_2017/pc_192.168.1.191_192.168.1.196',)
	else:
		msg = f'{data_name}'
		raise NotImplementedError(msg)

	# get normal and abnormal (not subflows)
	normal_files = []
	abnormal_files = []
	for data_name in subdatasets:
		normal, abnormal = _get_ctu_flows(original_dir, out_dir, data_name, direction)
		normal_files.append(normal)
		abnormal_files.append(abnormal)

	return normal_files, abnormal_files


class CTU(Base):

	def __init__(self, dataset_name='CTU1',
	             out_dir=OUT_DIR, feature_name="", flow_direction='src_dst',
	             q_interval=0.9, header=False, verbose=0,
	             overwrite=OVERWRITE, random_state=RANDOM_STATE):
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
			normal_files, abnormal_files = get_ctu_flows(original_dir=ORIG_DIR,
			                                             out_dir=self.out_dir,
			                                             data_name=self.dataset_name,
			                                             direction=self.flow_direction)
			meta = self.flows2features(normal_files, abnormal_files, q_interval=self.q_interval)
			lg.debug(f'meta: {meta.keys()}')
		self.X, self.y = meta['X'], meta['y']

		return self.X, self.y
