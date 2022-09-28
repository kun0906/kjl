"""
    get ioT_lab_data info
"""

import os
import os.path as pth
import subprocess
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
from odet.pparser.parser import PCAP, _pcap2flows, _get_flow_duration, _get_split_interval, _flows2subflows
from kjl.utils.data import dump_data, data_info, split_train_test
from kjl.utils.tool import load_data
from kjl.utils.utils import time_func
from collections import Counter


def _get_path(dir_in, data_name):
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

    elif data_name == 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1':
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
        pth_abnormal = pth.join(dir_in, data_name, '2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
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

    print(pth_normal)
    print(pth_abnormal)
    print(pth_labels_normal)
    print(pth_labels_abnormal)

    return pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal


def label_flows(flows, pth_label='xxx.csv'):
    """
    1. The number of flows in pth_label is more than flows in pcap, is it possible?
    2. Some flow appears in pth_label, but not in flows, or vice versa, is it possible?

    Parameters
    ----------
    flows
    pth_label

    Returns
    -------

    """
    NORMAL_LABELS = [v.upper() for v in ['benign', 'normal']]
    # ANOMALY_LABELS = [v.upper() for v in ['ANOMALY', 'Malicious', 'FTP-PATATOR', 'SSH-PATATOR',
    #                                       'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye',
    #                                       'Heartbleed',
    #                                       'Web Attack – Brute Force', 'Web Attack – XSS',
    #                                       'Web Attack – Sql Injection', 'Infiltration',
    #                                       'Bot', 'PortScan', 'DDoS']]

    NORMAL = 'normal'.upper()
    ABNORMAL = 'abnormal'.upper()

    # load CSV with pandas
    csv = pd.read_csv(pth_label)

    labels = {}  # {fid:(1, 0)} # 'normal':1, 'abnormal':0
    cnt_anomaly = 0
    cnt_nomral = 0

    for i, r in enumerate(csv.index):
        if i % 10000 == 0:
            print("Label CSV row {}".format(i))
        row = csv.loc[r]
        # parse 5-tuple flow ID
        # When you merge two csvs with headers, the file includes 'LABEL' means this line is the header
        # so just skip it
        if 'LABEL' in row[" Label"].upper():
            continue
        fid = (str(row[" Source IP"]), str(row[" Destination IP"]), int(row[" Source Port"]),
               int(row[" Destination Port"]), int(row[" Protocol"]))
        # ensure all 5-tuple flows have same label
        label_i = row[" Label"].upper()
        if label_i in NORMAL_LABELS:
            label_i = NORMAL
            cnt_nomral += 1
        else:
            label_i = ABNORMAL
            cnt_anomaly += 1

        if fid in labels.keys():
            labels[fid][label_i] += 1  # labels = {fid: {'normal':1, 'abnormal': 1}}
        else:
            v = 1 if label_i == NORMAL else 0
            labels[fid] = {NORMAL: v, ABNORMAL: 1 - v}

    # decide the true label of each fid
    conflicts = {}
    mislabels = {NORMAL: 0, ABNORMAL: 0}
    for fid, value in labels.items():
        if value[ABNORMAL] > 0 and value[NORMAL] > 0:
            conflicts[fid] = value

        if value[NORMAL] > value[ABNORMAL]:
            labels[fid] = NORMAL
            mislabels[NORMAL] += value[ABNORMAL]  # label 'abnormal' as 'normal'
        else:
            labels[fid] = ABNORMAL
            mislabels[ABNORMAL] += value[NORMAL]  # label 'normal' as 'abnormal'

    # for debug
    an = 0
    na = 0
    for fid, value in conflicts.items():
        if value[NORMAL] > value[ABNORMAL]:
            an += value[ABNORMAL]
        else:
            na += value[NORMAL]

    print(f'label_csv: cnt_normal: {cnt_nomral}, cnt_anomaly: {cnt_anomaly}, Unique labels: {len(labels.keys())}, '
          f'Counter(labels.values()),{Counter(labels.values())}, conflicts: {len(conflicts.keys())}'
          f', mislabels = {mislabels},  abnormal labeled as normal: {an}, normal labeled as abnormal: {na}')

    # obtain the labels of the corresponding features
    new_labels = []
    not_existed_fids = []
    new_fids = []
    for i, (fid, pkt) in enumerate(flows):
        if i == 0:
            print(f'i=0: fid: {fid}, list(labels.keys())[0]: {list(labels.keys())[0]}')
        if fid in labels.keys():
            new_labels.append(labels[fid])
            new_fids.append(fid)
        else:
            not_existed_fids.append(fid)
            new_fids.append('None')
            new_labels.append('None')  # the fid does not exist in labels.csv

    print(f'***{len(not_existed_fids)} (unique fids: {len(set(not_existed_fids))}) flows do not exist in {pth_label},'
          f'Counter(not_existed_fids)[:10]{list(Counter(not_existed_fids))[:10]}')
    print(f'len(new_labels): {len(new_labels)}, unique labels of new_labels: {Counter(new_labels)}')
    return (new_fids, new_labels)


def get_flows(dir_in, data_name, verbose=5):
    if 'DS10_UNB_IDS' in data_name or 'DS-srcIP_192.168.10.' in data_name:
        # normal and abnormal packets are mixed into one pcap
        # 1) get flows mixed
        pcap_mixed, _, pth_labels_mixed, _ = _get_path(dir_in, data_name)
        flows_mixed, parser_time = time_func(_pcap2flows, pcap_mixed, verbose=verbose)
        print(f'{pcap_mixed} parse time: {parser_time} s.')

        # 2) use labels to seperate normal and abnormal flows
        fids, labels = label_flows(flows_mixed, pth_label=pth_labels_mixed)
        flows_normal = [(fid_flow, pkts) for (fid_flow, pkts), (fid, label) in
                        zip(flows_mixed, zip(fids, labels)) if fid_flow == fid and label == "normal".upper()]
        flows_abnormal = [(fid_flow, pkts) for (fid_flow, pkts), (fid, label) in
                          zip(flows_mixed, zip(fids, labels)) if fid_flow == fid and label == "abnormal".upper()]
    else:
        # 1) get full flows: normal and abnormal flows
        pcap_normal, pcap_abnormal, pth_label_normal, pth_label_abnormal = _get_path(dir_in, data_name)
        flows_normal, parser_time = time_func(_pcap2flows, pcap_normal, verbose=verbose)
        print(f'{flows_normal} parse time: {parser_time} s.')
        flows_abnormal, parser_time = time_func(_pcap2flows, pcap_abnormal, verbose=verbose)
        print(f'{flows_abnormal} parse time: {parser_time} s.')
        # 2) no need to separate

    # dump all flows, not subflows
    pth_flows_normal = pth.join(dir_in, data_name, 'raw_normal_flows.dat')
    pth_flows_abnormal = pth.join(dir_in, data_name, 'raw_abnormal_flows.dat')
    print(pth_flows_normal, pth_flows_abnormal)
    if not pth.exists(pth.dirname(pth_flows_normal)): os.makedirs(pth.dirname(pth_flows_normal))
    dump_data(flows_normal, pth_flows_normal)
    dump_data(flows_abnormal, pth_flows_abnormal)

    return flows_normal, flows_abnormal


def choose_flows(flows, num=10000, random_state=42):
    num_flows = len(flows)
    num = num_flows if num > num_flows else num
    # idxs = np.random.choice(num_flows, size=num, replace=False)
    # abnormal_test_idx = np.in1d(range(abnormal_data.shape[0]), abnormal_test_idx)
    flows = shuffle(flows, random_state=random_state)

    return flows[:num]


class FEATURES(PCAP):

    def __init__(self, feat_type='IAT_SIZE', fft=False, header=False, q_interval=0.9, verbose=10):
        self.feat_type = feat_type
        self.fft = fft
        self.header = header
        self.q_interval = q_interval
        self.verbose = verbose

    def flow2feats(self, flows, dim=None):
        self.flows = flows
        # self.dim = dim
        self.flow2features(feat_type=self.feat_type, fft=self.fft, header=self.header, dim=dim)

#
# def get_subflows(normal_flows, abnormal_flows,
#                  normal_subflows_file, abnormal_subflows_file,
#                  q_interval):
#     subflow_flg = True  # get subflows
#     if subflow_flg:
#         durations = [_get_flow_duration(flow[1]) for flow in normal_flows]
#         interval = _get_split_interval(durations, q_interval=q_interval)
#         data_info(np.asarray(durations).reshape(-1, 1), name='durations (normal)')
#         print(f'interval: {interval}, q_interval: {q_interval}')
#         normal_subflows = _flows2subflows(normal_flows, interval)
#         dump_data(normal_subflows, normal_subflows_file)
#
#         # abnormal_durations = [_get_flow_duration(flow[1]) for flow in abnormal_flows]
#         # data_info(np.asarray(abnormal_durations).reshape(-1, 1), name='durations (abnormal)')
#         abnormal_subflows = _flows2subflows(abnormal_flows, interval)  # here the interval equals normal interval
#         dump_data(abnormal_subflows, abnormal_subflows_file)
#
#         #  only choose random 10,000 normal flows as train set, test_set equals 2 times of num_abnormal
#         # num = 600 if 'DS60_UChi_IoT' in data_name else 400  # ?
#         num_abnormal = len(abnormal_subflows) if len(
#             abnormal_subflows) < 400 else 400  # test set will be 2 * num_abnormal
#         normal_subflows = choose_flows(normal_subflows, num=10000 + num_abnormal)
#         abnormal_subflows = choose_flows(abnormal_subflows, num=num_abnormal)
#
#     return normal_subflows, abnormal_subflows, interval


def split_train_arrival_test(normal_arr, abnormal_arr, random_state=42):
    """

    Parameters
    ----------
    normal_arr
    abnormal_arr
    random_state

    Returns
    -------

    """
    from fractions import Fraction

    n_feats = min([data.shape[1] for data in normal_arr])
    for i in range(len(normal_arr)):
        normal_arr[i] = normal_arr[i][:, :n_feats]
        abnormal_arr[i] = abnormal_arr[i][:, :n_feats]

    ##########################################################################################
    # dataset1
    X_train1, y_train1, X_test1, y_test1 = split_train_test(normal_arr[0], abnormal_arr[0], train_size=0.8,
                                                            test_size=-1, random_state=random_state)
    # dataset2
    X_train2, y_train2, X_test2, y_test2 = split_train_test(normal_arr[1], abnormal_arr[1], train_size=0.8,
                                                            test_size=-1, random_state=random_state)

    X_test = np.concatenate([X_test1, X_test2], axis=0)
    y_test = np.concatenate([y_test1, y_test2], axis=0)

    ##########################################################################################
    # dataset1: Split train set into two subsets: initial set (init_set) and new arrival set (arvl_set)
    # with ratio 1:1.
    # in the init_set: X_train1 / X_train2= 9:1
    X_train1, X_arrival1, y_train1, y_arrival1 = train_test_split(X_train1, y_train1, train_size=0.9,
                                                                  random_state=random_state)
    # in the arrival set: X_arrival1 / X_arrival = 1:9
    X_train2, X_arrival2, y_train2, y_arrival2 = train_test_split(X_train2, y_train2, train_size=0.1,
                                                                  random_state=random_state)
    X_train = np.concatenate([X_train1, X_train2], axis=0)
    y_train = np.concatenate([y_train1, y_train2], axis=0)

    X_arrival = np.concatenate([X_arrival1, X_arrival2], axis=0)
    y_arrival = np.concatenate([y_arrival1, y_arrival2], axis=0)

    print(f'X_train: {X_train.shape}, in which, X_train1/X_train2 is {Fraction(X_train1.shape[0], X_train2.shape[0])}')
    print(f'X_arrival: {X_arrival.shape} in which, X_arrival1/X_arrival2 is '
          f'{Fraction(X_arrival1.shape[0], X_arrival2.shape[0])}')
    print(
        f'X_test: {X_test.shape},in which, X_test1/X_test2 {Fraction(X_test1.shape[0], X_test2.shape[0], _normalize=False)}')

    return X_train, y_train, X_arrival, y_arrival, X_test, y_test



def get_combined_feats(subdatasets, in_dir, out_dir, random_state = 100):
        expand_dir = pth.join(out_dir, '-'.join(list(subdatasets)).replace('/', '-'), 'iat_size', 'header:False')
        combined_flows_file = os.path.join(expand_dir, 'combined_raw_flows.dat')
        overwrite = False
        if overwrite:
            os.remove(combined_flows_file)
        if not os.path.exists(combined_flows_file):
            ##########################################################################################
            # 1. get all combined flows
            combined_flows = []
            combined_normal_flows = []

            for data_name in subdatasets:
                normal_flows, abnormal_flows = get_flows(in_dir, data_name, verbose=10)
                print(f'len(flows_normal): {len(normal_flows)}, len(flows_abnormal): {len(abnormal_flows)}')
                combined_flows.append((normal_flows, abnormal_flows))
                combined_normal_flows.extend(normal_flows)

            if not os.path.exists(expand_dir): os.makedirs(expand_dir)
            print(combined_flows_file)
            dump_data(combined_flows, out_file=combined_flows_file)
        else:
            print('load data')
            combined_flows, load_time = time_func(load_data, combined_flows_file)
            print(f'load {combined_flows_file} takes {load_time} s.')
            combined_normal_flows = []
            for normal_flows, _ in combined_flows:
                combined_normal_flows.extend(normal_flows)

        ##########################################################################################
        # 2. get interval on all normal flows
        q_interval = 0.9
        durations = [_get_flow_duration(flow[1]) for flow in combined_normal_flows]
        interval = _get_split_interval(durations, q_interval=q_interval)
        data_info(np.asarray(durations).reshape(-1, 1), name='durations (normal)')
        print(f'interval: {interval}, q_interval: {q_interval}')

        ##########################################################################################
        # 3. get subflows by interval
        combined_subflows = []
        combined_normal_subflows = []
        for (normal_flows, abnormal_flows) in combined_flows:
            normal_flows = _flows2subflows(normal_flows, interval)
            abnormal_flows = _flows2subflows(abnormal_flows, interval)
            combined_subflows.append((normal_flows, abnormal_flows))
            combined_normal_subflows.extend(normal_flows)

        combined_subflows_file = os.path.join(os.path.dirname(combined_flows_file),
                                              f'combined_subflows-q_{q_interval}.dat')
        print(combined_subflows_file)
        dump_data(combined_subflows, combined_subflows_file)

        ##########################################################################################
        # 4. get features from subflows
        feat_type = 'IAT_SIZE'
        num_pkts = [len(pkts) for fid, pkts in combined_normal_subflows]
        dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
        print(f'{feat_type}, dim: {dim}, q_interval: {q_interval}')
        if feat_type == 'IAT_SIZE': iat_size_dim = 2*dim -1
        combined_features_file = os.path.join(os.path.dirname(combined_flows_file),
                                              f'combined_feats-q_{q_interval}-{feat_type}-dim_{iat_size_dim}.dat')
        if not os.path.exists(combined_features_file):
            combined_features = []
            ##########################################################################################
            # subflows to features
            # print('load data...')
            # combined_subflows, load_time = time_func(load_data, combined_subflows_file)
            # print(f'finish loading subflows and it takes {load_time} seconds')
            ft = FEATURES(feat_type, fft=False, header=False)

            for (normal_flows, abnormal_flows) in combined_subflows:
                # data_info(np.asarray([len(pkts) for fid, pkts in normal_flows])[:, np.newaxis],
                #           name=f'packets of {type} flows')
                ft.flow2feats(normal_flows, dim)
                normal_feats = ft.features
                ft.flow2feats(abnormal_flows, dim)
                abnormal_feats = ft.features

                combined_features.append((normal_feats, abnormal_feats))

            # only save fixed number for normal and abnormal data
            normal_lst = []
            abnormal_lst = []
            for (normal, abnormal) in combined_features:
                shuffle(normal, replace=True, random_state=random_state)
                normal = normal[: 10000]
                normal_lst.append(normal)

                shuffle(abnormal, replace=True, random_state=random_state)
                abnormal = abnormal[: 400]
                abnormal_lst.append(abnormal)

            print(combined_features_file)
            dump_data(combined_subflows, combined_features_file)
        else:
            print('load data')
            combined_features, load_time = time_func(load_data, combined_features_file)
            print(f'load {combined_features_file} takes {load_time} s.')

        #
        # X_train, y_train, X_arrival, y_arrival, X_test, y_test = split_train_arrival_test(normal_lst, abnormal_lst,
        #                                                                                   random_state=random_state)

        return combined_features, combined_features_file


def main(random_state= 100):
    """Get results on all data with the current parameters(header, model, gs, kjl)

        Parameters
        ----------
        header

        Returns
        -------
            0: succeed
            otherwise: failed.
        """

    # dir_in = f'data_{model}'
    obs_dir = '../../IoT_feature_sets_comparison_20190822/examples/'
    in_dir = f'{obs_dir}data/data_reprst/pcaps'
    out_dir = f'data/data_kjl'  # (normal and abnormal)
    datasets = [  # 'DEMO_IDS/DS-srcIP_192.168.10.5',
        # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',  # data_name is unique
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

    for i, subdatasets in enumerate(datasets):
        print(f'{i}: {subdatasets}')
        combined_feats, out_file = get_combined_feats(subdatasets, in_dir, out_dir, random_state)

if __name__ == '__main__':
    main(random_state=100)
