""" Get features

"""
import os
import os.path as pth
import traceback
import numpy as np
import sklearn
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from kjl.dataset import uchicago
from kjl.dataset.uchicago import UChicago
from kjl.utils.data import load_data, dump_data
from kjl.utils.tool import execute_time, time_func, mprint, data_info
from odet.pparser.parser import _pcap2flows, PCAP, _get_flow_duration, _get_split_interval, _flows2subflows
from matplotlib import pyplot as plt, cm
from collections import Counter
from config import *

RANDOM_STATE = 42


def _pcap2fullflows(pcap_file='', label_file=None, label='normal'):
    pp = PCAP(pcap_file=pcap_file)
    pp.pcap2flows()
    if label_file is None:
        pp.labels = [label] * len(pp.flows)
    else:
        pp.label_flows(label_file, label=label)
    flows = pp.flows
    labels = pp.labels
    print(f'labels: {Counter(labels)}')

    if label_file is not None:
        normal_flows = []
        normal_labels = []
        abnormal_flows = []
        abnormal_labels = []
        for (f, l) in zip(flows, labels):
            # print(l, 'normal' in l, 'abnormal' in l)
            if l.startswith('normal'):  # 'normal' in l and 'abnormal' in l both will return True
                normal_flows.append(f)
                normal_labels.append(l)
            else:
                # print(l)
                abnormal_flows.append(f)
                abnormal_labels.append(l)
    else:
        if label == 'normal':  # 'normal' in label:
            normal_flows = flows
            normal_labels = labels
            abnormal_flows = None
            abnormal_labels = None
        else:
            normal_flows = None
            normal_labels = None
            abnormal_flows = flows
            abnormal_labels = labels

    return normal_flows, normal_labels, abnormal_flows, abnormal_labels


def _get_path(dir_in, data_name, overwrite=False):
    if 'UNB/CIC_IDS_2017' in data_name:
        ##############################################################################################################
        # step 1: get path
        if data_name == 'UNB/CIC_IDS_2017/pc_192.168.10.5':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'UNB/CIC_IDS_2017/pc_192.168.10.8':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.8_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.8_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'UNB/CIC_IDS_2017/pc_192.168.10.9':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.9_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.9_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None


        elif data_name == 'UNB/CIC_IDS_2017/pc_192.168.10.14':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.14_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.14_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'UNB/CIC_IDS_2017/pc_192.168.10.15':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.15_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.15_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None
        else:
            msg = f'{data_name} does not found.'
            raise ValueError(msg)

        ##############################################################################################################
        # step 2:  pcap 2 flows
        normal_file = os.path.dirname(pth_pcap_mixed) + '/normal_flows_labels.dat'
        abnormal_file = os.path.dirname(pth_pcap_mixed) + '/abnormal_flows_labels.dat'
        if overwrite:
            if os.path.exists(normal_file): os.remove(normal_file)
            if os.path.exists(abnormal_file): os.remove(abnormal_file)
        if not os.path.exists(normal_file) or not os.path.exists(abnormal_file):
            normal_flows, normal_labels, abnormal_flows, abnormal_labels = _pcap2fullflows(pcap_file=pth_pcap_mixed,
                                                                                           label_file=pth_labels_mixed)

            dump_data((normal_flows, normal_labels), out_file=normal_file)
            dump_data((abnormal_flows, abnormal_labels), out_file=abnormal_file)

    else:
        ##############################################################################################################
        # step 1: get path
        if data_name == 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'srcIP_10.42.0.1_normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name, 'srcIP_10.42.0.119_anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS30_OCS_IoT/DS31-srcIP_192.168.0.13':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'MIRAI/benign-dec-EZVIZ-ip_src-192.168.0.13-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'MIRAI/mirai-udpflooding-1-dec-EZVIZ-ip_src-192.168.0.13-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name,
                                  '2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU/IOT_2017/pc_192.168.1.196':
            # normal and abormal are independent
            # pth_normal = pth.join(dir_in, data_name,
            #                       '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
            pth_normal = pth.join(dir_in, data_name,
                                  '2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_202.171.168.50':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '201912071400-10000000pkts_00000_src_202_171_168_50_normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '201912071400-10000000pkts_00000_src_202_4_27_109_anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_203.78.7.165':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_185.8.54.240.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_203.78.4.32':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.4.32.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_203.78.7.165.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_222.117.214.171':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_222.117.214.171.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_101.27.14.204':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_101.27.14.204.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_18.178.219.109':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.4.32.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_18.178.219.109.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2019/ghome_192.168.143.20':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'google_home-2daysactiv-src_192.168.143.20-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2019/scam_192.168.143.42':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'samsung_camera-2daysactiv-src_192.168.143.42-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'samsung_camera-2daysactiv-src_192.168.143.42-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'samsung_fridge-2daysactiv-src_192.168.143.43-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'samsung_fridge-2daysactiv-src_192.168.143.43-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'bose_soundtouch-2daysactiv-src_192.168.143.48-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS60_UChi_IoT/iotlab_open_shut_fridge_192.168.143.43':
            # normal and abormal are independent
            'idle'
            'iotlab_open_shut_fridge_192.168.143.43/open_shut'
            pth_normal = pth.join(dir_in, data_name, 'bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'bose_soundtouch-2daysactiv-src_192.168.143.48-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'WRCCDC/2020-03-20':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'wrccdc.2020-03-20.174351000000002-172.16.16.30-normal.pcap')
            # pth_abnormal = pth.join(dir_in, data_name,
            #                         'wrccdc.2020-03-20.174351000000002-172.16.16.16.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'wrccdc.2020-03-20.174351000000002-10.183.250.172-abnormal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None
        elif data_name == 'DEFCON/ctf26':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'DEFCON26ctf_packet_captures-src_10.0.0.2-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'DEFCON26ctf_packet_captures-src_10.13.37.23-abnormal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'ISTS/ISTS_2015':
            # normal and abormal are independent
            # pth_normal = pth.join(dir_in, data_name, 'snort.log.1425741051-src_10.128.0.13-normal.pcap')
            # pth_normal = pth.join(dir_in, data_name, 'snort.log.1425823409-src_10.2.1.80.pcap')
            # pth_normal = pth.join(dir_in, data_name, 'snort.log.1425824560-src_129.21.3.17.pcap')
            # pth_normal = pth.join(dir_in, data_name,
            #                       'snort.log-merged-srcIP_10.128.0.13-10.0.1.51-10.0.1.4-10.2.12.40.pcap')
            #
            # pth_abnormal = pth.join(dir_in, data_name,
            #                         'snort.log-merged-srcIP_10.2.12.50.pcap')
            pth_normal = pth.join(dir_in, data_name, 'snort.log-merged-3pcaps.pcap'
                                  )
            # pth_abnormal = pth.join(dir_in, data_name, 'snort.log.1425824164.pcap')
            pth_abnormal = pth.join(dir_in, data_name, 'snort.log-merged-srcIP_10.2.4.30.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MACCDC/MACCDC_2012/pc_192.168.202.79':
            # normal and abormal are independent
            # pth_normal = pth.join(dir_in, data_name, 'maccdc2012_00000-srcIP_192.168.229.153.pcap')   # the result does beat OCSVM.
            pth_normal = pth.join(dir_in, data_name, 'maccdc2012_00000-srcIP_192.168.202.79.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'maccdc2012_00000-srcIP_192.168.202.76.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU_IOT23/CTU-IoT-Malware-Capture-7-1':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, '2018-07-20-17-31-20-192.168.100.108.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name,
                                        'CTU-IoT-Malware-Capture-7-1-conn.log.labeled.txt.csv-src_192.168.100.108.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        else:
            print('debug')
            data_name = 'DEMO_IDS/DS-srcIP_192.168.10.5'
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        ##############################################################################################################
        # step 2: pcap 2 flows
        normal_file = os.path.dirname(pth_normal) + '/normal_flows_labels.dat'
        abnormal_file = os.path.dirname(pth_normal) + '/abnormal_flows_labels.dat'
        if overwrite:
            if os.path.exists(normal_file): os.remove(normal_file)
            if os.path.exists(abnormal_file): os.remove(abnormal_file)
        if not os.path.exists(normal_file) or not os.path.exists(abnormal_file):
            normal_flows, normal_labels, _, _ = _pcap2fullflows(pcap_file=pth_normal,
                                                                label_file=None, label='normal')
            _, _, abnormal_flows, abnormal_labels = _pcap2fullflows(pcap_file=pth_abnormal,
                                                                    label_file=None, label='abnormal')

            dump_data((normal_flows, normal_labels), out_file=normal_file)
            dump_data((abnormal_flows, abnormal_labels), out_file=abnormal_file)

    print(f'normal_file: {normal_file}')
    print(f'abnormal_file: {abnormal_file}')

    return normal_file, abnormal_file


def _subflows2featutes(flows, labels, dim=10, verbose=10):
    # extract features from each flow given feat_type
    feat_type = 'IAT_SIZE'
    pp = PCAP()
    pp.flows = flows
    pp.labels = labels
    pp.flow2features(feat_type, fft=False, header=False, dim=dim)
    # out_file = f'{out_dir}/features-q_interval:{q_interval}.dat'
    # print('features+labels: ', out_file)
    # features = pp.features
    # labels = pp.labels
    # dump_data((features, labels), out_file)

    return pp.features, pp.labels


class PCAP2FEATURES():

    def __init__(self, out_dir='', random_state=100, overwrite=False):
        self.out_dir = out_dir
        self.verbose = 10
        self.random_state = random_state
        self.overwrite = overwrite

        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)

    def get_path(self, datasets, in_dir):
        normal_files = []
        abnormal_files = []
        for _idx, _name in enumerate(datasets):
            normal_file, abnormal_file = _get_path(in_dir, data_name=_name, overwrite=self.overwrite)
            normal_files.append(normal_file)
            abnormal_files.append(abnormal_file)

        return normal_files, abnormal_files

    @execute_time
    def flows2features(self, normal_files, abnormal_files, q_interval=0.9):
        is_same_duration = True
        if is_same_duration:
            self._flows2features(normal_files, abnormal_files, q_interval=q_interval)
        else:
            self._flows2features_seperate(normal_files, abnormal_files, q_interval=q_interval)

    def _flows2features(self, normal_files, abnormal_files, q_interval=0.9):
        print(f'normal_files: {normal_files}')
        print(f'abnormal_files: {abnormal_files}')
        durations = []
        normal_flows = []
        normal_labels = []
        for i, f in enumerate(normal_files):
            (flows, labels), load_time = time_func(load_data, f)
            normal_flows.extend(flows)
            print(f'i: {i}, load_time: {load_time} s.')
            normal_labels.extend([f'normal_{i}'] * len(labels))
            data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in flows]).reshape(-1, 1),
                      name=f'durations_{i}')
            durations.extend([_get_flow_duration(pkts) for fid, pkts in flows])

        # 1. get interval from all normal flows
        data_info(np.asarray(durations).reshape(-1, 1), name='durations')
        interval = _get_split_interval(durations)
        print(f'interval {interval} when q_interval: {q_interval}')

        abnormal_flows = []
        abnormal_labels = []
        for i, f in enumerate(abnormal_files):
            flows, labels = load_data(f)
            abnormal_flows.extend(flows)
            abnormal_labels.extend([f'abnormal_{i}'] * len(labels))
        print(f'fullflows: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')

        # 2. flows2subflows
        normal_flows, normal_labels = _flows2subflows(normal_flows, interval=interval, labels=normal_labels,
                                                      flow_pkts_thres=2,
                                                      verbose=1)
        abnormal_flows, abnormal_labels = _flows2subflows(abnormal_flows, interval=interval, labels=abnormal_labels,
                                                          flow_pkts_thres=2, verbose=1)
        print(f'subflows: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')

        # 3. subflows2features
        num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
        dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
        X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
                                                verbose=self.verbose)
        X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
                                                    verbose=self.verbose)
        print(f'subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')

        self.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
        X = np.concatenate([X_normal, X_abnormal], axis=0)
        y = np.concatenate([y_normal, y_abnormal], axis=0)
        self.Xy_file = os.path.join(self.out_dir, 'Xy-normal-abnormal.dat')
        dump_data((X, y), out_file=self.Xy_file)
        print(f'Xy_file: {self.Xy_file}')

    def _flows2features_seperate(self, normal_files, abnormal_files, q_interval=0.9):
        print(f'normal_files: {normal_files}')
        print(f'abnormal_files: {abnormal_files}')

        X = []
        y = []
        for i, (f1, f2) in enumerate(zip(normal_files, abnormal_files)):
            (normal_flows, labels), load_time = time_func(load_data, f1)
            normal_labels = [f'normal_{i}'] * len(labels)
            print(f'i: {i}, load_time: {load_time} s.')

            # 1. get durations
            data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in normal_flows]).reshape(-1, 1),
                      name=f'durations_{i}')
            durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]

            interval = _get_split_interval(durations, q_interval=q_interval)
            print(f'interval {interval} when q_interval: {q_interval}')

            # 2. flows2subflows
            normal_flows, normal_labels = _flows2subflows(normal_flows, interval=interval, labels=normal_labels,
                                                          flow_pkts_thres=2,
                                                          verbose=1)
            # 3. subflows2features
            num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
            data_info(np.asarray(num_pkts).reshape(-1, 1), name='num_ptks_for_flows')
            dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
            print(f'i: {i}, dim: {dim}')
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
            (abnormal_flows, labels), load_time = time_func(load_data, f2)
            abnormal_labels = [f'abnormal_{i}'] * len(labels)
            abnormal_flows, abnormal_labels = _flows2subflows(abnormal_flows, interval=interval, labels=abnormal_labels,
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
            print(
                f'subflows (before sampling): normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
            print(f'after resampling: normal_labels: {Counter(y_normal)}, abnormal_labels: {Counter(y_abnormal)}')
            # break
        max_dim = max([len(v) for v in X])
        print(f'===max_dim: {max_dim}')
        new_X = []
        for v in X:
            v = v + (max_dim - len(v)) * [0]
            new_X.append(np.asarray(v, dtype=float))

        X = np.asarray(new_X, dtype=float)
        y = np.asarray(y, dtype=str)
        self.Xy_file = os.path.join(self.out_dir, 'Xy-normal-abnormal.dat')
        dump_data((X, y), out_file=self.Xy_file)
        print(f'Xy_file: {self.Xy_file}')


def split_train_arrival_test(X, y, params):
    """

    Parameters
    ----------
    normal_arr
    abnormal_arr
    random_state

    Returns
    -------

    """
    random_state = params.random_state
    verbose = params.verbose
    # Step 1. Shuffle data first
    X, y = shuffle(X, y, random_state=random_state)
    if verbose >= DEBUG: data_info(X, name='X')

    n_init_train = params.n_init_train  # 5000
    n_init_test_abnm_0 = 50
    n_arrival = 500    # params.n_init_train # 5000
    n_test_abnm_0 = 100

    idx_nm_0 = y == 'normal_0'
    X_nm_0, y_nm_0 = X[idx_nm_0], y[idx_nm_0]
    idx_abnm_0 = y == 'abnormal_0'
    X_abnm_0, y_abnm_0 = X[idx_abnm_0], y[idx_abnm_0]

    if params.data_type == 'one_dataset':
        X_nm_1, y_nm_1 = X_nm_0, y_nm_0
        X_abnm_1, y_abnm_1 = X_abnm_0, y_abnm_0
    elif params.data_type == 'two_datasets':
        idx_nm_1 = y == 'normal_1'
        X_nm_1, y_nm_1 = X[idx_nm_1], y[idx_nm_1]
        idx_abnm_1 = y == 'abnormal_1'
        X_abnm_1, y_abnm_1 = X[idx_abnm_1], y[idx_abnm_1]
    else:
        raise NotImplementedError()

    N1 = int(round(params.percent_first_init * n_init_train)) + n_init_test_abnm_0 + n_test_abnm_0 + \
         int(round((1 - params.percent_first_init) * n_arrival))
    N2 = int(round((1 - params.percent_first_init) * n_init_train)) + n_init_test_abnm_0 + n_test_abnm_0 + int(
        round(params.percent_first_init * n_arrival))

    AN1 = n_init_test_abnm_0 + n_test_abnm_0
    is_resample = True
    if is_resample:
        print(
            f'before reampling, y_nm_0: {Counter(y_nm_0)}, y_abnm_0: {Counter(y_abnm_0)}, y_nm_1: {Counter(y_nm_1)}, y_abnm_1: {Counter(y_abnm_1)}')
        if len(y_nm_0) < N1:
            X_nm_0, y_nm_0 = sklearn.utils.resample(X_nm_0, y_nm_0, n_samples=N1, replace=True,
                                                    random_state=42)
        if len(y_nm_1) < N2:
            X_nm_1, y_nm_1 = sklearn.utils.resample(X_nm_1, y_nm_1, n_samples=N2, replace=True,
                                                    random_state=42)
        if len(y_abnm_0) < AN1:
            X_abnm_0, y_abnm_0 = sklearn.utils.resample(X_abnm_0, y_abnm_0, n_samples=AN1, replace=True,
                                                        random_state=42)
        if len(y_abnm_1) < AN1:
            X_abnm_1, y_abnm_1 = sklearn.utils.resample(X_abnm_1, y_abnm_1, n_samples=AN1, replace=True,
                                                        random_state=42)

        print(
            f'after reampling, y_nm_0: {Counter(y_nm_0)}, y_abnm_0: {Counter(y_abnm_0)}, y_nm_1: {Counter(y_nm_1)}, y_abnm_1: {Counter(y_abnm_1)}')
    X_normal = np.concatenate([X_nm_0, X_nm_1], axis=0)
    X_abnormal = np.concatenate([X_abnm_0, X_abnm_1], axis=0)
    if verbose >= DEBUG: data_info(X_normal, name='X_normal')
    if verbose >= DEBUG:   data_info(X_abnormal, name='X_abnormal')

    def random_select(X, y, n=100, random_state=100):
        X, y = shuffle(X, y, random_state=random_state)
        X0 = X[:n, :]
        y0 = y[:n]

        rest_X = X[n:, :]
        rest_y = y[n:]
        # X_nm_1, y_nm_1 = sklearn.utils.resample(X, y, n_samples=n, replace=False,
        #                                         random_state=random_state)
        # if n <=0:
        #     _, dim = X.shape
        #     X0, rest_X, y0, rest_y = np.empty((0, dim)), X, np.empty((0,)), y
        # else:
        #     X0, rest_X, y0, rest_y = train_test_split(X, y, train_size=n, random_state=random_state, shuffle=True)
        return X0, y0, rest_X, rest_y

    ########################################################################################################
    # Step 2.1. Get init_set
    # 1) get init_train: normal
    X_init_train_nm_0, y_init_train_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0,
                                                                         n=int(round(
                                                                             params.percent_first_init * n_init_train,
                                                                             0)), random_state=random_state)
    X_init_train_nm_1, y_init_train_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1,
                                                                         n=int(round((1 - params.percent_first_init) *
                                                                                     n_init_train, 0)),
                                                                         random_state=random_state)
    X_init_train = np.concatenate([X_init_train_nm_0, X_init_train_nm_1], axis=0)
    y_init_train = np.concatenate([y_init_train_nm_0, y_init_train_nm_1], axis=0)

    # 2) get init_test: normal + abnormal
    X_init_test_nm_0, y_init_test_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0, n=n_init_test_abnm_0,
                                                                       random_state=random_state)
    X_init_test_nm_1, y_init_test_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1, n=n_init_test_abnm_0,
                                                                       random_state=random_state)
    X_init_test_abnm_0, y_init_test_abnm_0, X_abnm_0, y_abnm_0 = random_select(X_abnm_0, y_abnm_0,
                                                                               n=n_init_test_abnm_0,
                                                                               random_state=random_state)
    X_init_test_abnm_1, y_init_test_abnm_1, X_abnm_1, y_abnm_1 = random_select(X_abnm_1, y_abnm_1,
                                                                               n=n_init_test_abnm_0,
                                                                               random_state=random_state)
    X_init_test = np.concatenate([X_init_test_nm_0, X_init_test_nm_1,
                                  X_init_test_abnm_0, X_init_test_abnm_1,
                                  ], axis=0)
    y_init_test = np.concatenate([y_init_test_nm_0, y_init_test_nm_1,
                                  y_init_test_abnm_0, y_init_test_abnm_1,
                                  ], axis=0)

    ########################################################################################################
    # Step 2.2. Get arrival_set: normal
    X_arrival_nm_0, y_arrival_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0,
                                                                   n=int(
                                                                       round(
                                                                           (1 - params.percent_first_init) * n_arrival,
                                                                           0)), random_state=random_state)
    X_arrival_nm_1, y_arrival_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1,
                                                                   n=int(round(params.percent_first_init *
                                                                               n_arrival, 0)),
                                                                   random_state=random_state)
    X_arrival = np.concatenate([X_arrival_nm_0, X_arrival_nm_1], axis=0)
    y_arrival = np.concatenate([y_arrival_nm_0, y_arrival_nm_1], axis=0)

    ########################################################################################################
    # Step 2.3. Get test_set
    # get test_set: normal + abnormal
    X_test_nm_0, y_test_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0, n=n_test_abnm_0,
                                                             random_state=random_state)
    X_test_nm_1, y_test_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1, n=n_test_abnm_0,
                                                             random_state=random_state)
    X_test_abnm_0, y_test_abnm_0, X_abnm_0, y_abnm_0 = random_select(X_abnm_0, y_abnm_0, n=n_test_abnm_0,
                                                                     random_state=random_state)
    X_test_abnm_1, y_test_abnm_1, X_abnm_1, y_abnm_1 = random_select(X_abnm_1, y_abnm_1, n=n_test_abnm_0,
                                                                     random_state=random_state)
    X_test = np.concatenate([X_test_nm_0, X_test_nm_1, X_test_abnm_0, X_test_abnm_1], axis=0)
    y_test = np.concatenate([y_test_nm_0, y_test_nm_1, y_test_abnm_0, y_test_abnm_1], axis=0)

    X_init_train, y_init_train = shuffle(X_init_train, y_init_train, random_state=random_state)
    X_init_test, y_init_test = shuffle(X_init_test, y_init_test, random_state=random_state)
    X_arrival, y_arrival = shuffle(X_arrival, y_arrival, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

    mprint(f'X_init_train: {X_init_train.shape}, in which, y_init_train is {Counter(y_init_train)}', verbose, INFO)
    mprint(f'X_init_test: {X_init_test.shape}, in which, y_init_test is {Counter(y_init_test)}', verbose, INFO)
    mprint(f'X_arrival: {X_arrival.shape}, in which, y_arrival is {Counter(y_arrival)}', verbose, INFO)
    mprint(f'X_test: {X_test.shape}, in which, y_test is {Counter(y_test)}', verbose, INFO)

    if verbose >= INFO:
        data_info(X_init_train, name='X_init_train')
        data_info(X_init_test, name='X_init_test')
        data_info(X_arrival, name='X_arrival')
        data_info(X_test, name='X_test')

    return X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test, y_test


def plot_data(X, y):
    plt.figure()
    y_unique = np.unique(y)
    colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        plt.scatter(this_X[:, 0], this_X[:, 1], s=50,
                    c=color[np.newaxis, :],
                    alpha=0.5, edgecolor='k',
                    label="Class %s" % this_y)
    plt.legend(loc="best")
    plt.title("Data")
    plt.show()


def _generate_mimic_data(data_type='', random_state=42, out_file=''):
    if data_type == 'two_datasets':
        X, y = make_blobs(n_samples=[12000, 200, 12000, 200],
                          centers=[(-1, -2), (0, 0), (5, 5), (7.5, 7.5)], cluster_std=[(1, 1), (1, 1), (1, 1), (1, 1)],
                          # cluster_std=[(2, 10), (1,1), (2,3)
                          n_features=2,
                          random_state=random_state)  # generate data from multi-variables normal distribution

        y = np.asarray(y, dtype=str)
        y[y == '0'] = 'normal_0'
        y[y == '1'] = 'abnormal_0'
        y[y == '2'] = 'normal_1'
        y[y == '3'] = 'abnormal_1'

    else:
        msg = out_file
        raise NotImplementedError(msg)

    # plt.scatter(X[:, 0], X[:, 1])
    plot_data(X, y)
    dump_data((X, y), out_file)

    return out_file


def generate_data(data_name, data_type='two_datasets', out_file='.dat', overwrite=False, random_state=42):
    if overwrite and pth.exists(out_file): os.remove(out_file)

    in_dir = f'../../Datasets'
    if data_name == 'mimic_GMM':
        out_file = _generate_mimic_data(data_type=data_type, random_state=random_state, out_file=out_file)
    elif data_name in ['UNB1_UNB2', 'UNB1_UNB3', 'UNB1_UNB4', 'UNB1_UNB5',
                       'UNB2_UNB1', 'UNB2_UNB3']:  # mix UNB1 and UNB2
        # pcaps and flows directory
        # in_dir = f'./data/data_reprst/pcaps'
        if data_name == 'UNB1_UNB2':
            subdatasets = (
            'UNB/CIC_IDS_2017/pc_192.168.10.5', 'UNB/CIC_IDS_2017/pc_192.168.10.8')  # each_data has normal and abnormal
        elif data_name == 'UNB1_UNB3':
            subdatasets = (
            'UNB/CIC_IDS_2017/pc_192.168.10.5', 'UNB/CIC_IDS_2017/pc_192.168.10.9')  # each_data has normal and abnormal
        elif data_name == 'UNB1_UNB4':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.5',
                           'UNB/CIC_IDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB1_UNB5':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.5',
                           'UNB/CIC_IDS_2017/pc_192.168.10.15')  # each_data has normal and abnormal
        elif data_name == 'UNB2_UNB3':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.8', 'UNB/CIC_IDS_2017/pc_192.168.10.9')
        elif data_name == 'UNB3_UNB4':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.9', 'UNB/CIC_IDS_2017/pc_192.168.10.14')
        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file
    elif data_name in ['UNB1_CTU1', 'UNB1_MAWI1', 'CTU1_MAWI1',
                       'CTU1_UNB1', 'MAWI1_UNB1', 'MAWI1_CTU1',
                       'MACCDC1_UNB1', 'MACCDC1_CTU1', 'MACCDC1_MAWI1',
                       'UNB1_SCAM1', 'CTU1_SCAM1', 'MAWI1_SCAM1',
                       'UNB2_CTU1', 'UNB2_MAWI1', 'UNB1_ISTS1']:  # mix UNB1 and others
        if data_name == 'UNB1_CTU1':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.5', 'CTU/IOT_2017/pc_192.168.1.196')
        elif data_name == 'UNB1_MAWI1':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.5', 'MAWI/WIDE_2019/pc_203.78.7.165')
        elif data_name == 'CTU1_MAWI1':
            subdatasets = ('CTU/IOT_2017/pc_192.168.1.196', 'MAWI/WIDE_2019/pc_203.78.7.165')
        elif data_name == 'CTU1_UNB1':
            subdatasets = ('CTU/IOT_2017/pc_192.168.1.196', 'UNB/CIC_IDS_2017/pc_192.168.10.5',)
        elif data_name == 'MAWI1_UNB1':
            subdatasets = ('MAWI/WIDE_2019/pc_203.78.7.165', 'UNB/CIC_IDS_2017/pc_192.168.10.5')
        elif data_name == 'MAWI1_CTU1':
            subdatasets = ('MAWI/WIDE_2019/pc_203.78.7.165', 'CTU/IOT_2017/pc_192.168.1.196')
        elif data_name == 'MACCDC1_UNB1':
            subdatasets = ('MACCDC/MACCDC_2012/pc_192.168.202.79', 'UNB/CIC_IDS_2017/pc_192.168.10.5')
        elif data_name == 'MACCDC1_CTU1':
            subdatasets = ('MACCDC/MACCDC_2012/pc_192.168.202.79', 'CTU/IOT_2017/pc_192.168.1.196')
        elif data_name == 'MACCDC1_MAWI1':
            subdatasets = ('MACCDC/MACCDC_2012/pc_192.168.202.79', 'MAWI/WIDE_2019/pc_203.78.7.165')
        elif data_name == 'UNB1_SCAM1':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.5', 'UCHI/IOT_2019/scam_192.168.143.42')
        elif data_name == 'CTU1_SCAM1':
            subdatasets = ('CTU/IOT_2017/pc_192.168.1.196', 'UCHI/IOT_2019/scam_192.168.143.42')
        elif data_name == 'MAWI1_SCAM1':
            subdatasets = ('MAWI/WIDE_2019/pc_203.78.7.165', 'UCHI/IOT_2019/scam_192.168.143.42')
        elif data_name == 'UNB2_CTU1':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.8', 'CTU/IOT_2017/pc_192.168.1.196')
        elif data_name == 'UNB2_MAWI1':
            subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.8', 'MAWI/WIDE_2019/pc_203.78.7.165')
        # elif data_name == 'UNB1_ISTS1':
        #     subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.8', 'MAWI/WIDE_2019/pc_203.78.7.165')
        # elif data_name == 'UNB1_ISTS2':
        #     subdatasets = ('UNB/CIC_IDS_2017/pc_192.168.10.8', 'MAWI/WIDE_2019/pc_203.78.7.165')

        # elif data_name == 'CTU1_MAWI1':
        #     subdatasets = ('CTU/IOT_2017/pc_192.168.1.196', 'MAWI/WIDE_2019/pc_203.78.7.165')
        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file

    elif data_name in ['FRIG_IDLE12', 'FRIG1_OPEN_BROWSE', 'FRIG1_BROWSE_OPEN']:

        if pth.exists(out_file):
            # pass
            return
        normal1 = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
        abnormal1 = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'

        ips = ['192.168.143.43']
        uchicago.filter_ips(in_dir=pth.join(in_dir, normal1), out_dir=pth.join(in_dir, normal1), ips=ips,
                            direction='both', keep_original=False)
        uchicago.filter_ips(in_dir=pth.join(in_dir, abnormal1), out_dir=pth.join(in_dir, abnormal1), ips=ips,
                            direction='both', keep_original=False)

        normal2 = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle1'
        abnormal2 = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
        ips = ['192.168.143.43']
        uchicago.filter_ips(in_dir=pth.join(in_dir, normal2), out_dir=pth.join(in_dir, normal2), ips=ips,
                            direction='both', keep_original=False)
        uchicago.filter_ips(in_dir=pth.join(in_dir, abnormal2), out_dir=pth.join(in_dir, abnormal2), ips=ips,
                            direction='both', keep_original=False)

        if data_name == 'FRIG_IDLE12':
            # # # Fridge: (normal1: idle1, and normal2: idle2) (abnormal1: open_shut, and abnormal2: browse)
            subdatasets1 = (normal1, abnormal1)  # normal1 and abnormal_1
            subdatasets2 = (normal2, abnormal2)  # normal2 and abnormal_2
        elif data_name == 'FRIG1_OPEN_BROWSE':
            # # # Fridge: (abnormal1: open_shut, and abnormal2: browse) (normal1: idle1, and normal2: idle2)
            subdatasets1 = (abnormal1, normal1)
            subdatasets2 = (abnormal2, normal2)
        elif data_name == 'FRIG1_BROWSE_OPEN':
            # # # Fridge: (abnormal: browse, and abnormal2: open_shut ) abnormal: (normal1: idle2, and normal2: idle1)
            subdatasets1 = (abnormal2, normal2)
            subdatasets2 = (abnormal1, normal1)
        else:
            raise NotImplementedError(data_name)

        subdatasets = (subdatasets1,subdatasets2)
        out_dir = 'data/feats'
        normal_files, abnormal_files = uchicago.get_flows(in_dir, subdatasets, out_dir)
        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file


    elif data_name in ['UNB1_FRIG1', 'CTU1_FRIG1', 'MAWI1_FRIG1',
                       'FRIG1_UNB1', 'FRIG1_CTU1', 'FRIG1_MAWI1',
                       'UNB1_FRIG2',
                       'CTU1_UNB2','CTU1_FRIG2',
                       'MAWI1_UNB2','MAWI1_FRIG2',
                       'FRIG2_UNB1', 'FRIG2_CTU1', 'FRIG2_MAWI1',
                       'SCAM1_FRIG1', 'SCAM1_FRIG2',
                       'FRIG1_SCAM1', 'FRIG2_SCAM1'
                                      'UNB2_FRIG1', 'UNB2_FRIG2',
                       'UNB1_DRYER1', 'DRYER1_UNB1',
                       'UNB1_DWSHR1', 'DWSHR1_UNB1',
                       'UNB1_WSHR1', 'WSHR1_UNB1',
                       'FRIG1_DWSHR1', 'FRIG2_DWSHR1'
                                       'CTU1_DWSHR1',
                       'MAWI1_DWSHR1', 'FRIG2_DWSHR1',
                       'MACCDC1_UNB1', 'MACCDC1_CTU1', 'MACCDC1_MAWI1', 'MACCDC1_DWSHR1'

                       ]:
        if pth.exists(out_file):
            pass
        else:
            out_dir = 'data/feats'
            if data_name in ['UNB1_FRIG1', 'FRIG1_UNB1']:  # Fridge: idle and open_shut
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.5')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB1_FRIG1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['UNB2_FRIG1', 'FRIG1_UNB2']:  # Fridge: idle and open_shut
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.8')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB2_FRIG1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1


            elif data_name in ['CTU1_FRIG1', 'FRIG1_CTU1']:
                subdatasets1 = ('CTU/IOT_2017/pc_192.168.1.196')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'CTU1_FRIG1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['MAWI1_FRIG1', 'FRIG1_MAWI1']:
                subdatasets1 = ('MAWI/WIDE_2019/pc_203.78.7.165')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'MAWI1_FRIG1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['MAWI1_UNB2', 'UNB2_MAWI1']:  # Fridge: idle and open_shut
                subdatasets1 = ('MAWI/WIDE_2019/pc_203.78.7.165')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                subdatasets2 = ('UNB/CIC_IDS_2017/pc_192.168.10.8')
                subdatasets = (subdatasets2,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
                normal_files2, abnormal_files2 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                if data_name == 'MAWI1_UNB2':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1


            elif data_name in ['UNB1_FRIG2', 'FRIG2_UNB1']:  # Fridge: idle and open_shut
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.5')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB1_FRIG2':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['CTU1_UNB2', 'UNB2_CTU1']:  # Fridge: idle and open_shut
                subdatasets1 = ('CTU/IOT_2017/pc_192.168.1.196')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                subdatasets2 = ('UNB/CIC_IDS_2017/pc_192.168.10.8')
                subdatasets = (subdatasets2,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
                normal_files2, abnormal_files2 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                if data_name == 'MAWI1_UNB2':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['CTU1_FRIG2', 'FRIG2_CTU1']:
                subdatasets1 = ('CTU/IOT_2017/pc_192.168.1.196')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'CTU1_FRIG2':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1


            elif data_name in ['MAWI1_FRIG2', 'FRIG2_MAWI1']:
                subdatasets1 = ('MAWI/WIDE_2019/pc_203.78.7.165')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'MAWI1_FRIG2':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['SCAM1_FRIG1', 'FRIG1_SCAM1']:
                subdatasets1 = ('UCHI/IOT_2019/scam_192.168.143.42')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat
                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'SCAM1_FRIG1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['SCAM1_FRIG2', 'FRIG2_SCAM1']:
                subdatasets1 = ('UCHI/IOT_2019/scam_192.168.143.42')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat
                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'SCAM1_FRIG1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1



            elif data_name in ['SCAM1_FRIG2', 'FRIG2_SCAM1']:
                subdatasets1 = ('UCHI/IOT_2019/scam_192.168.143.42')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat
                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'SCAM1_FRIG2':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['UNB2_FRIG1', 'FRIG1_UNB2']:
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.8')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets, in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB2_FRIG1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['UNB2_FRIG2', 'FRIG2_UNB2']:
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.8')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat
                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB2_FRIG2':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['UNB1_DRYER1', 'DRYER1_UNB1']:
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.5')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/dryer_192.168.143.99/idle'
                activity = 'UCHI/IOT_2020/dryer_192.168.143.99/open_dryer'
                ips = ['192.168.143.99']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB1_DRYER1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1


            elif data_name in ['UNB1_DWSHR1', 'DWSHR1_UNB1']:
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.5')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/dwshr_192.168.143.76/idle'
                activity = 'UCHI/IOT_2020/dwshr_192.168.143.76/open_dwshr'
                ips = ['192.168.143.76']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB1_DWSHR1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1


            elif data_name in ['UNB1_WSHR1', 'WSHR1_UNB1']:
                subdatasets1 = ('UNB/CIC_IDS_2017/pc_192.168.10.5')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/wshr_192.168.143.100/idle'
                activity = 'UCHI/IOT_2020/wshr_192.168.143.100/open_wshr'
                ips = ['192.168.143.100']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'UNB1_WSHR1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1


            elif data_name in ['FRIG1_DWSHR1', '']:

                normal = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                abnormal = 'UCHI/IOT_2020/sfrig_192.168.143.43/open_shut'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, normal), out_dir=pth.join(in_dir, normal), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, abnormal), out_dir=pth.join(in_dir, abnormal), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets1 = (normal, abnormal)  # normal and abnormal_0
                subdatasets = (subdatasets1,)
                normal_files1, abnormal_files1 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                activity = 'UCHI/IOT_2020/dwshr_192.168.143.76/open_dwshr'
                idle = 'UCHI/IOT_2020/dwshr_192.168.143.76/idle'
                ips = ['192.168.143.76']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'FRIG1_DWSHR1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['FRIG2_DWSHR1', '']:

                idle = 'UCHI/IOT_2020/sfrig_192.168.143.43/idle'
                activity = 'UCHI/IOT_2020/sfrig_192.168.143.43/browse'
                ips = ['192.168.143.43']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets1 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets1,)
                normal_files1, abnormal_files1 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                idle = 'UCHI/IOT_2020/dwshr_192.168.143.76/idle'
                activity = 'UCHI/IOT_2020/dwshr_192.168.143.76/open_dwshr'
                ips = ['192.168.143.76']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'FRIG2_DWSHR1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['CTU1_DWSHR1', '']:
                subdatasets1 = ('CTU/IOT_2017/pc_192.168.1.196')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat
                idle = 'UCHI/IOT_2020/dwshr_192.168.143.76/idle'
                activity = 'UCHI/IOT_2020/dwshr_192.168.143.76/open_dwshr'
                ips = ['192.168.143.76']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'CTU1_DWSHR1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1


            elif data_name in ['MAWI1_DWSHR1', '']:
                subdatasets1 = ('MAWI/WIDE_2019/pc_203.78.7.165')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/dwshr_192.168.143.76/idle'
                activity = 'UCHI/IOT_2020/dwshr_192.168.143.76/open_dwshr'
                ips = ['192.168.143.76']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'MAWI1_DWSHR1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            elif data_name in ['MACCDC1_DWSHR1', '']:
                subdatasets1 = ('MACCDC/MACCDC_2012/pc_192.168.202.79')
                subdatasets = (subdatasets1,)
                pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state,
                                   overwrite=overwrite)
                normal_files1, abnormal_files1 = pf.get_path(subdatasets,
                                                             in_dir)  # pcap to xxx_flows_labels.dat.dat

                idle = 'UCHI/IOT_2020/dwshr_192.168.143.76/idle'
                activity = 'UCHI/IOT_2020/dwshr_192.168.143.76/open_dwshr'
                ips = ['192.168.143.76']
                uchicago.filter_ips(in_dir=pth.join(in_dir, idle), out_dir=pth.join(in_dir, idle), ips=ips,
                                    direction='both', keep_original=False)
                uchicago.filter_ips(in_dir=pth.join(in_dir, activity), out_dir=pth.join(in_dir, activity), ips=ips,
                                    direction='both', keep_original=False)
                subdatasets2 = (idle, activity)  # normal and abnormal_0
                subdatasets = (subdatasets2,)
                normal_files2, abnormal_files2 = uchicago.get_flows(in_dir, subdatasets, out_dir)

                if data_name == 'MACCDC1_DWSHR1':
                    normal_files = normal_files1 + normal_files2
                    abnormal_files = abnormal_files1 + abnormal_files2
                else:  # data_name == 'FRIG1_UNB1':  # Fridge: idle and open_shut
                    normal_files = normal_files2 + normal_files1
                    abnormal_files = abnormal_files2 + abnormal_files1

            pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
            pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
            out_file = pf.Xy_file

    else:
        msg = data_name
        raise NotImplementedError(msg)

    return out_file


@execute_time
def main(random_state, n_jobs=-1, single=False, verbose=10):
    """

    Parameters
    ----------
    random_state
    n_jobs

    Returns
    -------

    """
    feat_type = 'IAT_SIZE'
    fft = False
    header = False
    q_interval = 0.9
    verbose = 10

    if single:
        datasets = [
            # 'DEMO_IDS/DS-srcIP_192.168.10.5',
            # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',  # data_name is unique
            'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
            # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
            # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
            # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
            #
            # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
            #
            # # 'DS30_OCS_IoT/DS31-srcIP_192.168.0.13',

            # 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
            # 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196',
            # #
            # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
            # 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
            # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
            # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
            # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
            # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',

            # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
            # 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
            # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
            # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'

            # 'WRCCDC/2020-03-20',
            # 'DEFCON/ctf26',
            # 'ISTS/2015',
            'MACCDC/2012',
            'CTU_IOT23/CTU-IoT-Malware-Capture-7-1',
        ]

    else:
        datasets = [  # 'DEMO_IDS/DS-srcIP_192.168.10.5',
            # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',  # data_name is unique
            # ('DS10_UNB_IDS/DS13-srcIP_192.168.10.8', 'DS10_UNB_IDS/DS14-srcIP_192.168.10.9'),   # demo
            ('DS10_UNB_IDS/DS12-srcIP_192.168.10.8', 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9'),
            ('DS10_UNB_IDS/DS13-srcIP_192.168.10.9', 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9'),
            # # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
            # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
            # #
            # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
            # #
            # # # 'DS30_OCS_IoT/DS31-srcIP_192.168.0.13',
            #
            # 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
            # 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196',
            # #
            # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
            # 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
            # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
            # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
            # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
            # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
            #
            # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
            # 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
            # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
            # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'

            # 'WRCCDC/2020-03-20',
            # 'DEFCON/ctf26',
            # 'ISTS/2015',
            # 'MACCDC/2012',
            # 'CTU_IOT23/CTU-IoT-Malware-Capture-7-1',
        ]
    #
    # # in_dir = 'data/data_kjl'
    # # dir_in = f'data_{model}'
    # obs_dir = '../../IoT_feature_sets_comparison_20190822/examples/'
    # in_dir = f'{obs_dir}data/data_reprst/pcaps'
    # out_dir = 'out/'

    for subdatasets in datasets:
        obs_dir = '../../IoT_feature_sets_comparison_20190822/examples/'
        in_dir = f'{obs_dir}data/data_reprst/pcaps'
        pf = PCAP2FEATURES(out_dir=os.path.dirname(Xy_file), random_state=random_state)
        normal_files, abnormal_files = pf.get_path(subdatasets, in_dir, out_dir='out/')
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        data, Xy_file = pf.data, pf.Xy_file


if __name__ == '__main__':
    # main(random_state=RANDOM_STATE, n_jobs=1, single=False)

    _generate_mimic_data(data_type='two_datasets', random_state=42, out_file='./data/demo.dat')
