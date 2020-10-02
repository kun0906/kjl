""" Get features

"""
import os
import os.path as pth
import traceback
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle

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
        if 'normal' in label:
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
    if 'UNB_IDS' in data_name:
        ##############################################################################################################
        # step 1: get path
        if data_name == 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.8_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.8_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.9_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.9_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None


        elif data_name == 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.14_AGMT.pcap')
            pth_labels_mixed = pth.join(dir_in, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.14_AGMT.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15':
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
            pth_normal = pth.join(dir_in, data_name, '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name,
                                  '2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '201912071400-10000000pkts_00000_src_202_171_168_50_normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '201912071400-10000000pkts_00000_src_202_4_27_109_anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_185.8.54.240.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.4.32.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_203.78.7.165.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_222.117.214.171.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_101.27.14.204.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, '202007011400-srcIP_203.78.4.32.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    '202007011400-srcIP_18.178.219.109.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20':
            # normal and abormal are independent
            pth_normal = pth.join(dir_in, data_name, 'google_home-2daysactiv-src_192.168.143.20-normal.pcap')
            pth_abnormal = pth.join(dir_in, data_name,
                                    'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42':
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

        elif data_name == 'ISTS/2015':
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

        elif data_name == 'MACCDC/2012':
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
            durations.extend([_get_flow_duration(pkts) for fid, pkts in flows])

        # 1. get interval from all normal flows
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


#
# def get_feats(datasets, in_dir, out_dir, q_interval = 0.9, feat_type='IAT_SIZE', fft=False, header=False,
#               random_state = 100, verbose = 10, single=True):
#
#     print(f'{datasets}')
#     try:
#         if single:
#             normal_pcap_file, abnormal_pcap_file, normal_label_file, abnormal_label_file = _get_path(in_dir, datasets)
#             expand_out_dir = os.path.join(out_dir, os.path.dirname(normal_pcap_file.split('examples/')[-1]))
#             if not os.path.exists(expand_out_dir): os.makedirs(expand_out_dir)
#             print(expand_out_dir)
#             data,Xy_file = get_normal_abnormal_featrues(normal_file=(normal_pcap_file, normal_label_file),
#                                                 abnormal_file=(abnormal_pcap_file, abnormal_label_file),
#                                                 feat_type= feat_type,
#                                                 q_interval=q_interval, fft=fft, header=header, out_dir = expand_out_dir,
#                                                 verbose=verbose, random_state=random_state)
#         else:
#             normal_files = []
#             abnormal_files = []
#             for _idx, _name in enumerate(datasets):
#                 normal_pcap_file, abnormal_pcap_file, normal_label_file, abnormal_label_file = _get_path(
#                     in_dir, data_name=_name)
#                 if _idx == 0:
#                     expand_out_dir = os.path.join(out_dir, os.path.dirname(
#                         normal_pcap_file.split('examples/')[-1].replace(_name, '-'.join(datasets).replace('/', '-'))))
#                 normal_files.append((normal_pcap_file, normal_label_file))
#                 abnormal_files.append((abnormal_pcap_file, abnormal_label_file))
#
#             if not os.path.exists(expand_out_dir): os.makedirs(expand_out_dir)
#             print(expand_out_dir)
#             data, Xy_file = get_mutli_nomral_abnormal_features(normal_files, abnormal_files,
#                                                       feat_type=feat_type,
#                                                       q_interval=q_interval,
#                                                       fft=fft, header=header, out_dir = expand_out_dir,
#                                                       verbose=verbose, random_state=random_state)
#
#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#
#     return data, Xy_file
#


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

    n_init_train = 5000
    n_init_test_abnm_0 = 50
    n_arrival = 5000
    n_test_abnm_0 = 50

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

    if verbose >= DEBUG:
        data_info(X_init_train, name='X_init_train')
        data_info(X_init_test, name='X_init_test')
        data_info(X_arrival, name='X_arrival')
        data_info(X_test, name='X_test')

    return X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test, y_test


#
# def split_train_arrival_test_back_up(X, y, params):
#     """
#
#     Parameters
#     ----------
#     normal_arr
#     abnormal_arr
#     random_state
#
#     Returns
#     -------
#
#     """
#     random_state = params.random_state
#     percent_first_init = params.percent_first_init
#     one_data_flg = False if params.data_type == 'two_datasets' else True
#     # dataset 1
#     idx = y == 'normal_0'
#     normal_X_0 = X[idx]
#     normal_y_0 = y[idx]
#     idx = y == 'normal_1'
#     normal_X_1 = X[idx]
#     normal_y_1 = y[idx]
#     normal_X = np.concatenate([normal_X_0, normal_X_1], axis=0)
#     normal_y = np.concatenate([normal_y_0, normal_y_1], axis=0)
#     data_info(normal_X, name='normal_X')
#
#     idx = y == 'abnormal_0'
#     abnormal_X_0 = X[idx]
#     abnormal_y_0 = y[idx]
#     idx = y == 'abnormal_1'
#     abnormal_X_1 = X[idx]
#     abnormal_y_1 = y[idx]
#     abnormal_X = np.concatenate([abnormal_X_0, abnormal_X_1], axis=0)
#     abnormal_y = np.concatenate([abnormal_y_0, abnormal_y_1], axis=0)
#     data_info(abnormal_X, name='abnormal_X')
#
#     data_info(X, name='X')
#     init_train_n = 5000
#     init_test_n = 100
#
#     arrival_n = 5000
#     # test_n = 5000
#
#     normal_y = np.asarray(normal_y)
#     abnormal_y = np.asarray(abnormal_y)
#
#     def random_select(X, y, n=100, random_state=100):
#         X, y = shuffle(X, y, random_state=random_state)
#
#         X0 = X[:n, :]
#         y0 = y[:n]
#
#         rest_X = X[n:, :]
#         rest_y = y[n:]
#         return X0, y0, rest_X, rest_y
#
#     # normal_X, normal_y = shuffle(normal_X, normal_y, random_state=random_state)
#
#     ##################################################################################################################
#     # dataset 1
#     idx = normal_y == 'normal_0'
#     normal_X_1 = normal_X[idx]
#     normal_y_1 = normal_y[idx]
#     idx = abnormal_y == 'abnormal_0'
#     abnormal_X_1 = abnormal_X[idx]
#     abnormal_y_1 = abnormal_y[idx]
#
#     # percent_first_init = 1     # 0.9, 0.95, 1.00
#     # init_set: train and test set
#     # init_train_set: data1/data2=9:1
#     # init_test_set: data1/data=9:1
#     init_test_abnormal1_size = 50
#     test_abnormal1_size = 50
#     init_train_normal_X_1, init_train_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1, normal_y_1,
#                                                                                          n=int(round(
#                                                                                              init_train_n * percent_first_init,
#                                                                                              0)),
#                                                                                          random_state=random_state)
#
#     init_test_normal_X_1, init_test_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1, normal_y_1,
#                                                                                        n=init_test_abnormal1_size,
#                                                                                        random_state=random_state)
#     init_test_abnormal_X_1, init_test_abnormal_y_1, abnormal_X_1, abnormal_y_1 = random_select(abnormal_X_1,
#                                                                                                abnormal_y_1,
#                                                                                                n=init_test_abnormal1_size,
#                                                                                                random_state=random_state)
#
#     # arrival set: data1/data2 = 1:9
#     arrival_train_normal_X_1, arrival_train_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1,
#                                                                                                normal_y_1,
#                                                                                                n=int(round(
#                                                                                                    init_train_n * (
#                                                                                                            1.0 - percent_first_init),
#                                                                                                    0)),
#                                                                                                random_state=random_state)
#
#     # test set: data1/data2 = 1:1
#     test_normal_X_1, test_normal_y_1, normal_X_1, normal_y_1 = random_select(normal_X_1, normal_y_1,
#                                                                              n=test_abnormal1_size,
#                                                                              random_state=random_state)
#
#     test_abnormal_X_1, test_abnormal_y_1, abnormal_X_1, abnormal_y_1 = random_select(abnormal_X_1, abnormal_y_1,
#                                                                                      n=test_abnormal1_size,
#                                                                                      random_state=random_state)
#
#     ##################################################################################################################
#     # dataset 2
#     # arrival set: data1/data2 = 1:9
#     if one_data_flg:  # use the same dataset
#         normal_X_2 = normal_X_1
#         normal_y_2 = normal_y_1
#         abnormal_X_2 = abnormal_X_1
#         abnormal_y_2 = abnormal_y_1
#
#     else:
#         idx = normal_y == 'normal_1'
#         normal_X_2 = normal_X[idx]
#         normal_y_2 = normal_y[idx]
#         idx = abnormal_y == 'abnormal_1'
#         abnormal_X_2 = abnormal_X[idx]
#         abnormal_y_2 = abnormal_y[idx]
#
#     # init_set: train and test set
#     init_train_normal_X_2, init_train_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
#                                                                                          n=int(round(init_train_n * (
#                                                                                                  1 - percent_first_init),
#                                                                                                      0)),
#                                                                                          random_state=random_state)
#
#     init_test_normal_X_2, init_test_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
#                                                                                        n=init_test_abnormal1_size,
#                                                                                        random_state=random_state)
#     init_test_abnormal_X_2, init_test_abnormal_y_2, abnormal_X_2, abnormal_y_2 = random_select(abnormal_X_2,
#                                                                                                abnormal_y_2,
#                                                                                                n=init_test_abnormal1_size,
#                                                                                                random_state=random_state)
#
#     arrival_train_normal_X_2, arrival_train_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
#                                                                                                n=int(round(
#                                                                                                    init_train_n * percent_first_init,
#                                                                                                    0)),
#                                                                                                random_state=random_state)
#
#     test_normal_X_2, test_normal_y_2, normal_X_2, normal_y_2 = random_select(normal_X_2, normal_y_2,
#                                                                              n=test_abnormal1_size,
#                                                                              random_state=random_state)
#
#     test_abnormal_X_2, test_abnormal_y_2, abnormal_X_2, abnormal_y_2 = random_select(abnormal_X_2, abnormal_y_2,
#                                                                                      n=test_abnormal1_size,
#                                                                                      random_state=random_state)
#
#     # init_set: train and test set
#     X_init_train = np.concatenate([init_train_normal_X_1, init_train_normal_X_2], axis=0)
#     y_init_train = np.concatenate([init_train_normal_y_1, init_train_normal_y_2], axis=0)
#
#     X_init_test = np.concatenate(
#         [init_test_normal_X_1, init_test_normal_X_2, init_test_abnormal_X_1, init_test_abnormal_X_2], axis=0)
#     y_init_test = np.concatenate(
#         [init_test_normal_y_1, init_test_normal_y_2, init_test_abnormal_y_1, init_test_abnormal_y_2], axis=0)
#
#     # arrival set: only train set
#     X_arrival = np.concatenate([arrival_train_normal_X_1, arrival_train_normal_X_2], axis=0)
#     y_arrival = np.concatenate([arrival_train_normal_y_1, arrival_train_normal_y_2], axis=0)
#
#     # test set
#     X_test = np.concatenate([test_normal_X_1, test_normal_X_2, test_abnormal_X_1, test_abnormal_X_2], axis=0)
#     y_test = np.concatenate([test_normal_y_1, test_normal_y_2, test_abnormal_y_1, test_abnormal_y_2], axis=0)
#
#     print(f'X_init_train: {X_init_train.shape}, in which, y_init_train is {Counter(y_init_train)}')
#     print(f'X_init_test: {X_init_test.shape}, in which, y_init_test is {Counter(y_init_test)}')
#     print(f'X_arrival_train: {X_arrival.shape}, in which, y_arrival is {Counter(y_arrival)}')
#     print(f'X_test: {X_test.shape}, in which, y_test is {Counter(y_test)}')
#
#     X_init_train, y_init_train = shuffle(X_init_train, y_init_train, random_state=random_state)
#     X_init_test, y_init_test = shuffle(X_init_test, y_init_test, random_state=random_state)
#     X_arrival, y_arrival = shuffle(X_arrival, y_arrival, random_state=random_state)
#     X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
#
#     if params.verbose >= 1:
#         data_info(X_init_train, name='X_init_train')
#         data_info(X_init_test, name='X_init_test')
#         data_info(X_arrival, name='X_arrival')
#         data_info(X_test, name='X_test')
#
#     return X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test, y_test

#
# def get_normal_abnormal(normal_file, abnormal_file, random_state=42):
#     """Get normal and abnormal data
#
#     Parameters
#     ----------
#     normal_file
#     abnormal_file
#     random_state
#
#     Returns
#     -------
#         normal data
#         abnormal data
#
#     """
#     if not pth.exists(normal_file) or not pth.exists(abnormal_file):
#         _normal_file = pth.splitext(normal_file)[0] + '.csv'
#         _abnormal_file = pth.splitext(abnormal_file)[0] + '.csv'
#         # extract data from csv file
#         normal_data, abnormal_data = extract_data(_normal_file, _abnormal_file,
#                                                   meta_data={'idxs_feat': [0, -1],
#                                                              'train_size': -1,
#                                                              'test_size': -1})
#         # transform data format
#         dump_data(normal_data, normal_file)
#         dump_data(abnormal_data, abnormal_file)
#     else:
#         normal_data = load_data(normal_file)
#         abnormal_data = load_data(abnormal_file)
#
#     return normal_data, abnormal_data


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


#
#
# def mimic_data(name='', random_state=43, single_device=False):
#     # two classes: one has 1000 and another has 200 datapoints
#
#     if single_device:
#         # if 'GMM' in name:
#         #     gmm16 = GaussianMixture(n_components=3, covariance_type='diag', random_state=random_state)
#         #     X, y  = gmm16.sample(400)
#         # else:
#         X, y = make_blobs(n_samples=[12000, 200],
#                           centers=[(-1, -2), (0, 0)], cluster_std=[(1, 1), (1, 1)],
#                           # cluster_std=[(2, 10), (1,1), (2,3)
#                           n_features=2,
#                           random_state=random_state)  # generate data from multi-variables normal distribution
#
#         # plt.scatter(X[:, 0], X[:, 1])
#         plot_data(X, y)
#
#         idx = y == 0
#         X_normal = X[idx]
#         y_normal = ['normal_0'] * X_normal.shape[0]
#
#         abnormal_idx = y != 0
#         X_abnormal = X[abnormal_idx]
#         y_abnormal = ['abnormal_0' if v == 1 else 'abnormal_1' for v in y[abnormal_idx]]
#         data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
#     else:
#         X, y = make_blobs(n_samples=[12000, 200, 12000, 200],
#                           centers=[(-1, -2), (0, 0), (5, 5), (7.5, 7.5)], cluster_std=[(1, 1), (1, 1), (1, 1), (1, 1)],
#                           # cluster_std=[(2, 10), (1,1), (2,3)
#                           n_features=2,
#                           random_state=random_state)  # generate data from multi-variables normal distribution
#         # plt.scatter(X[:, 0], X[:, 1])
#         plot_data(X, y)
#
#         idx = y == 0
#         X_normal_0 = X[idx]
#         y_normal_0 = ['normal_0'] * X_normal_0.shape[0]
#
#         idx = y == 2
#         X_normal_1 = X[idx]
#         y_normal_1 = ['normal_1'] * X_normal_1.shape[0]
#         X_normal = np.concatenate([X_normal_0, X_normal_1], axis=0)
#         y_normal = y_normal_0 + y_normal_1
#
#         idx = y == 1
#         X_abnormal_0 = X[idx]
#         y_abnormal_0 = ['abnormal_0'] * X_abnormal_0.shape[0]
#
#         idx = y == 3
#         X_abnormal_1 = X[idx]
#         y_abnormal_1 = ['abnormal_1'] * X_abnormal_0.shape[0]
#         X_abnormal = np.concatenate([X_abnormal_0, X_abnormal_1], axis=0)
#         y_abnormal = y_abnormal_0 + y_abnormal_1
#
#         data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
#
#     Xy_file = f'out/data/data_reprst/pcaps/{name}/Xy-normal-abnormal.dat'
#     print(Xy_file)
#     dump_data(data, out_file=Xy_file)
#
#     return data, Xy_file
#

# def artifical_data():
#     datasets = ['mimic_GMM_dataset',
#                 ]
#
#     for dataset in datasets:
#         data = mimic_data(dataset, random_state=100)


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

    if data_name == 'mimic_GMM':
        out_file = _generate_mimic_data(data_type=data_type, random_state=random_state, out_file=out_file)
    elif data_name in ['UNB1_UNB2', 'UNB2_UNB3']:  # mix UNB1 and UNB2
        # pcaps and flows directory
        # in_dir = f'./data/data_reprst/pcaps'
        in_dir = f'../../IoT_feature_sets_comparison_20190822/examples/data/data_reprst/pcaps'
        if data_name == 'UNB1_UNB2':
            subdatasets = ('DS10_UNB_IDS/DS11-srcIP_192.168.10.5', 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8')
        elif data_name == 'UNB2_UNB3':
            subdatasets = ('DS10_UNB_IDS/DS12-srcIP_192.168.10.8', 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9')
        elif data_name == 'UNB3_UNB4':
            subdatasets = ('DS10_UNB_IDS/DS13-srcIP_192.168.10.9', 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14')

        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, in_dir)
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
