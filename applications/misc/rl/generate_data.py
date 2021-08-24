

"""Detect abnormal flows progressively using LSTM

run in the command under the applications directory
   PYTHONPATH=../:./ python3.7 applications/rnn_main_single.py > rnn_main_single.txt 2>&1 &
"""
# Author: kun.bj@outlook.com
# license: xxx
import os
import random
import os.path as pth
import numpy as np
import sklearn
import torch
from odet.utils.tool import dump_data, load_data

from rl._pcap_parser import PCAP_PKTS, _pcap2flows
from offline.generate_data import generate_data_speed_up, _pcap2fullflows

RANDOM_STATE = 100

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




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


def set_random_state(random_state=100):
    """To make reproducible results

    Returns
    -------

    """

    random.seed(random_state)
    np.random.seed(random_state)

    torch.manual_seed(random_state)
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


set_random_state(random_state=RANDOM_STATE)


def raw2features(raw_features, data_type=True, MTU=1500, normalize=True):
    """Extract features for the detection models

    Parameters
    ----------
    raw_features:
        each flow := fid, [feat_0, feat_1, feat_2, ..., ]

    Returns
    -------
    features:
        each flow := [feat, ... ]
    """

    def normalize_bytes(flow):
        return [[v / 255 for v in pkt] for pkt in flow]

    X = []
    for i, (fid, v_lst) in enumerate(raw_features):
        feat_0 = v_lst[0]
        feat_i_lst = v_lst[1:]

        if data_type == 'header':
            tmp_v = [v['header'] for v in feat_i_lst]
            tmp_v = [v + [0] * (40 - len(v)) if len(v) < 40 else v[:40] for v in tmp_v]
        elif data_type == 'header_payload':
            tmp_v = [v['header'] + v['payload'] for v in feat_i_lst]
            tmp_v = [v + [0] * (MTU - len(v)) if len(v) < MTU else v[:MTU] for v in tmp_v]
        else:  # data_type=='payload':
            payload_len = MTU - 40
            tmp_v = [v['payload'] for v in feat_i_lst]
            tmp_v = [v + [0] * (payload_len - len(v)) if len(v) < payload_len else v[:payload_len] for v in tmp_v]

        if normalize:
            tmp_v = normalize_bytes(tmp_v)

        X.append(tmp_v)

    return X




def _generate_datasets(overwrite=False):


    datasets = {}
    for data_name, data_path in dataname_path_mappings.items():
        if overwrite:
            if pth.exists(data_path): os.remove(data_path)

        if not pth.exists(data_path):
            data_path = generate_data_speed_up(data_name, out_file=data_path, overwrite=overwrite)
        X, y = load_data(data_path)
        datasets[(data_name, data_path)] = (X, y)

    datasets = list(datasets.items())

    return datasets



def generate_data(data_name, data_type='two_datasets', out_file='.dat', overwrite=False, random_state=42):
    if overwrite and pth.exists(out_file): os.remove(out_file)

    in_dir = f'../../Datasets'
    if data_name == 'mimic_GMM':
        # out_file = _generate_mimic_data(data_type=data_type, random_state=random_state, out_file=out_file)
        pass
    elif data_name in ['UNB1', 'UNB2', 'UNB1_UNB4', 'UNB1_UNB5',
                       'UNB2_UNB1', 'UNB2_UNB3']:  # mix UNB1 and UNB2
        # pcaps and flows directory
        # in_dir = f'./data/data_reprst/pcaps'
        if data_name == 'UNB1':
            subdatasets = (
                'UNB/CIC_IDS_2017/pc_192.168.10.5',
                )  # each_data has normal and abnormal
        elif data_name == 'UNB2':
            subdatasets = (
                'UNB/CIC_IDS_2017/pc_192.168.10.8',
                )  # each_data has normal and abnormal
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


    else:
        msg = data_name
        raise NotImplementedError(msg)

    return out_file



def load_flow_data(overwrite=False, random_state=100, full_flow=True, data_type='header'):
    """Get raw features from PCAP and store them into disk

    Parameters
    ----------
    overwrite: boolean
    full_flow: boolean
        use full flows, not subflows
    Returns
    -------
        X_norm, y_norm, X_abnorm, y_abnorm
    """
    feat = 'iat_size'
    in_dir = f'rl/data/{feat}'
    dataname_path_mappings = {
        # 'mimic_GMM': f'{in_dir}/mimic_GMM/Xy-normal-abnormal.dat',
        # 'mimic_GMM1': f'{in_dir}/mimic_GMM1/Xy-normal-abnormal.dat',
        'UNB1': f'{in_dir}/UNB1/Xy-normal-abnormal.dat',
        'UNB2': f'{in_dir}/UNB2/Xy-normal-abnormal.dat',
            # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
            # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
            # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
            # # # # #
            # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
            # # # # # #
            # 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
        # 'CTU1': f'{in_dir}/CTU1/Xy-normal-abnormal.dat',
        #     # # #
        #     # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        #     # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        # 'MAWI1':  f'{in_dir}/MAWI1/Xy-normal-abnormal.dat',
        #     # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
        #     # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
        #     # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
        #     # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
        #     # #
        #     # # #
        #     # 'WRCCDC/2020-03-20',
        #     # 'DEFCON/ctf26',
        # 'ISTS1':  f'{in_dir}/ISTS1/Xy-normal-abnormal.dat',
        # 'MACCDC1': f'{in_dir}/MACCDC1/Xy-normal-abnormal.dat',

        #     # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
        # 'SCAM1': f'{in_dir}/SCAM1/Xy-normal-abnormal.dat',
        # # #     # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
        # # #     # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
        # # #
    }

    print(f'len(dataname_path_mappings): {len(dataname_path_mappings)}')

    dataset_name = 'CTU/IOT_2017/DS41-srcIP_10.0.2.15'
    dataset_name = 'CTU/IOT_2017/pc_192.168.1.196'


    print(f'dataset: {dataset_name}')
    in_dir = '../../Datasets/'
    if dataset_name == 'CTU/IOT_2017/pc_192.168.1.196':
        in_norm_file = f'{in_dir}/{dataset_name}/2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/2018-12-21-15-50-14-src_192.168.1.195-CTU_IoT_Mirai_normal.pcap'
    elif dataset_name == 'CTU/IOT_2017/DS41-srcIP_10.0.2.15':
        in_norm_file = f'{in_dir}/{dataset_name}/2017-05-02_CTU_Normal_32-src_10.0.2.15_normal.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/2019-01-09-22-46-52-src_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap'
    elif dataset_name == 'ISTS/2015':
        in_norm_file = f'{in_dir}/{dataset_name}/snort.log-merged-3pcaps.pcap'
        # in_norm_file = f'{in_dir}/{dataset_name}/snort.log-merged-srcIP_10.2.4.30.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/snort.log-merged-srcIP_10.2.4.30.pcap'
    elif dataset_name == 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42':
        in_norm_file = f'{in_dir}/{dataset_name}/samsung_camera-2daysactiv-src_192.168.143.42-normal.pcap'
        in_abnorm_file = f'{in_dir}/{dataset_name}/samsung_camera-2daysactiv-src_192.168.143.42-anomaly.pcap'
    else:
        in_norm_file = 'data/lstm/demo_normal.pcap'
        in_abnorm_file = 'data/lstm/demo_abnormal.pcap'

    out_norm_file = in_norm_file + '-raw_normal_features.dat'
    out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'

    if overwrite or not os.path.exists(out_norm_file) or not os.path.exists(out_abnorm_file):
        # note: this work uses full flows, not subflows
        norm_pp = PCAP_PKTS(pcap_file=in_norm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
        if full_flow:
            flows = _pcap2flows(norm_pp.pcap_file, norm_pp.flow_ptks_thres,
                                verbose=norm_pp.verbose)
            norm_pp.flows = flows
        else:
            norm_pp.pcap2flows()

        norm_pp.flows2bytes()
        out_norm_file = in_norm_file + '-raw_normal_features.dat'
        dump_data(norm_pp.features, out_norm_file)

        abnorm_pp = PCAP_PKTS(pcap_file=in_abnorm_file, flow_ptks_thres=2, verbose=10, random_state=RANDOM_STATE)
        if full_flow:
            flows = _pcap2flows(abnorm_pp.pcap_file, abnorm_pp.flow_ptks_thres,
                                verbose=abnorm_pp.verbose)
            abnorm_pp.flows = flows
        else:
            abnorm_pp.pcap2flows(interval=norm_pp.interval)
        abnorm_pp.flows2bytes()

        out_abnorm_file = in_abnorm_file + '-raw_abnormal_features.dat'
        dump_data(abnorm_pp.features, out_abnorm_file)

    X_norm = raw2features(load_data(out_norm_file), data_type=data_type)
    y_norm = [0] * len(X_norm)
    X_abnorm = raw2features(load_data(out_abnorm_file), data_type=data_type)
    y_abnorm = [1] * len(X_abnorm)

    X = X_norm + X_abnorm
    y = y_norm + y_abnorm
    # return split_train_test(X_norm, y_norm, X_abnorm, y_abnorm, random_state)
    return X, y

def balance_data(X,y):

    X_norm = []
    y_norm = []
    X_abnorm= []
    y_abnorm = []
    for i, (x_, y_) in enumerate(zip(X,y)):
        if y_==0:
            X_norm.append(x_)
            y_norm.append(y_)
        else:
            X_abnorm.append(x_)
            y_abnorm.append(y_)

    n = min(len(y_norm), len(y_abnorm))
    # new_X, new_y = sklearn.utils.resample(X, y, n_samples=n, stratify=y, random_state=42)
    if n == len(y_norm):
        X_abnorm, y_abnorm = sklearn.utils.resample(X_abnorm, y_abnorm, n_samples=n,  random_state=42)
    else:
        X_norm, y_norm = sklearn.utils.resample(X_norm, y_norm, n_samples=n, random_state=42)

    new_X = X_norm + X_abnorm
    new_y = y_norm + y_abnorm

    return new_X, new_y

def split_train_test(X_norm, y_norm, X_abnorm, y_abnorm, random_state=100):
    """Split train and test set

    Parameters
    ----------
    X_norm
    y_norm
    X_abnorm
    y_abnorm
    random_state

    Returns
    -------

    """

    # X_norm = sklearn.utils.shuffle(X_norm, random_state)
    random.Random(random_state).shuffle(X_norm)  #注意此处打乱数据的作用
    # size = int(len(y_norm) // 2) if len(y_norm) <= len(y_abnorm) else min(400, len(y_abnorm))
    size = min(400, len(y_abnorm))
    X_test = X_norm[-size:] + X_abnorm[:size]
    y_test = y_norm[-size:] + y_abnorm[:size]
    # X_train = X_norm[:-size]
    # y_train = y_norm[:-size]
    X_train = X_norm[:size]
    y_train = y_norm[:size]
    print(f'X_train: {len(X_train)}, X_test {len(X_test)}')

    return X_train, y_train, X_test, y_test


def main(random_state=100):
    data_type = 'payload'  # header, header_payload
    X_train, y_train, X_test, y_test = load_flow_data(random_state=random_state, data_type=data_type)
    in_dim = len(X_train[0][0])
    # rnn = RNN(n_epochs=100, in_dim=in_dim, out_dim=10, n_layers=1, lr=1e-3, bias=False, random_state=random_state)
    #
    # rnn.train(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, split=True)
    #
    # rnn.test(X_test=X_test, y_test=y_test, split=True)


if __name__ == '__main__':
    main(random_state=RANDOM_STATE)


