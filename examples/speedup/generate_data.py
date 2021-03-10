""" Get features

"""
import os.path as pth
from collections import Counter
from shutil import copyfile

import sklearn
from matplotlib import pyplot as plt, cm
from odet.pparser.parser import PCAP, _get_flow_duration, _get_split_interval, _flows2subflows, _get_IAT, _get_SIZE, \
    _get_IAT_SIZE, _get_STATS, _get_SAMP_NUM, _get_SAMP_SIZE, _get_FFT_data, _get_header_features
from odet.utils.tool import timing
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

from kjl.dataset.uchicago import split_by_activity
from kjl.utils.data import load_data, dump_data
from kjl.utils.tool import execute_time, time_func, mprint, data_info
# from online.config import *

RANDOM_STATE = 42

"""Get basic info of pcap


Split pcap:
    editcap -A “2017-07-07 9:00” -B ”2017-07-07 12:00” Friday-WorkingHours.pcap Friday-WorkingHours_09_00-12_00.pcap
    editcap -A "2017-07-04 09:02:00" -B "2017-07-04 09:05:00" AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap
    # only save the first 10000 packets
    editcap -r AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap 0-10000

filter:
   cmd = f"tshark -r {in_file} -w {out_file} {srcIP_str}"
"""
import os
import subprocess
import numpy as np
import pickle


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    return data


def keep_ip(in_file, out_file='', kept_ips=[''], direction='src_dst'):
    if out_file == '':
        ips_str = '-'.join(kept_ips)
        out_file = os.path.splitext(in_file)[0] + f'-src_{ips_str}.pcap'  # Split a path in root and extension.
    if os.path.exists(out_file):
        return out_file
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    print(out_file)
    # only keep srcIPs' traffic
    if direction == 'src':
        srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
    else:  # default
        srcIP_str = " or ".join([f'ip.addr=={srcIP}' for srcIP in kept_ips])
    cmd = f"tshark -r {in_file} -w {out_file} {srcIP_str}"

    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return out_file


def keep_csv_ip(label_file, out_file, ips=[], direction='src_dst', header=True, keep_original=True, verbose=10):
    # from shutil import copyfile
    # copyfile(label_file, out_file)

    # print(label_file_lst, mrg_label_path)
    # if os.path.exists(mrg_label_path):
    #     os.remove(mrg_label_path)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    with open(out_file, 'w') as out_f:
        with open(label_file, 'r') as in_f:
            line = in_f.readline()
            while line:
                if line.strip().startswith('Flow') and header:
                    if header:
                        header = False
                        print(line)
                        out_f.write(line.strip('\n') + '\n')
                    else:
                        pass
                    line = in_f.readline()
                    continue
                if line.strip() == '':
                    line = in_f.readline()
                    continue

                exist = False
                for ip in ips:
                    if ip in line:
                        exist = True
                        break
                if exist:
                    out_f.write(line.strip('\n') + '\n')
                line = in_f.readline()

    return out_file


def merge_pcaps(in_files, out_file):
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    cmd = f"mergecap -w {out_file} " + ' '.join(in_files)

    print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return out_file


def merge_csvs(in_files=[], out_file=''):
    print(in_files, out_file)
    if os.path.exists(out_file):
        os.remove(out_file)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    # # combine all label files in the list
    # # combined_csv = pd.concat([pd.read_csv(f, header=None, usecols=[3,6]) for f in label_file_lst])
    # result_lst = []
    # for i, f in enumerate(label_file_lst):
    #     if i == 0:
    #         result_lst.append(pd.read_csv(f))
    #     else:
    #         result_lst.append(pd.read_csv(f, skiprows=0))
    # combined_csv = pd.concat(result_lst)
    # # export to csv
    # print(f'mrg_label_path: {mrg_label_path}')
    # combined_csv.to_csv(mrg_label_path, index=False, encoding='utf-8-sig')

    with open(out_file, 'w') as out_f:
        header = True
        for i, label_file in enumerate(in_files):
            with open(label_file, 'r') as in_f:
                line = in_f.readline()
                while line:
                    if line.strip().startswith('Flow ID') and header:
                        if header:
                            header = False
                            print(line)
                            out_f.write(line.strip('\n') + '\n')
                        else:
                            pass
                        line = in_f.readline()
                        continue
                    if line.strip() == '':
                        line = in_f.readline()
                        continue
                    out_f.write(line.strip('\n') + '\n')
                    line = in_f.readline()

    return out_file


def flows2subflows_SCAM(full_flows, interval=10, num_pkt_thresh=2, data_name='', abnormal=False):
    from scapy.layers.inet import UDP, TCP, IP
    remainder_cnt = 0
    new_cnt = 0  # a flow is not split by an intervals
    flows = []  # store the subflows
    step_flows = []
    tmp_arr2 = []
    tmp_arr1 = []
    print(f'interval: {interval}')
    print('normal file does not need split with different steps, only anomaly file needs.')
    for i, (fid, pkts) in enumerate(full_flows):
        times = [float(pkt.time) for pkt in pkts]
        if i % 1000 == 0:
            print(f'session_i: {i}, len(pkts): {len(pkts)}')

        flow_type = None
        new_flow = 0
        dur = max(times) - min(times)
        if dur >= 2 * interval:
            tmp_arr2.append(max(times) - min(times))  # 10% flows exceeds the interals

        if dur >= 1 * interval:
            tmp_arr1.append(max(times) - min(times))
        step = 0  # 'step' for 'normal data' always equals 0. If dataset needs to be agumented, then slide window with step
        while step < len(pkts):
            # print(f'i: {i}, step:{step}, len(pkts[{step}:]): {len(pkts[step:])}')
            dur_tmp = max(times[step:]) - min(times[step:])
            if dur_tmp <= interval:
                if step == 0:
                    subflow = [(float(pkt.time), pkt) for pkt in pkts[step:]]
                    step_flows.append((fid, subflow))
                    flows.append((fid, subflow))
                break  # break while loop
            flow_i = []
            subflow = []
            for j, pkt in enumerate(pkts[step:]):
                if TCP not in pkt and UDP not in pkt:
                    break
                if j == 0:
                    flow_start_time = float(pkt.time)
                    subflow = [(float(pkt.time), pkt)]
                    split_flow = False  # if a flow is not split with interval, label it as False, otherwise, True
                    continue
                # handle TCP packets
                if IP in pkt and TCP in pkt:
                    flow_type = 'TCP'
                    fid = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport, 6)
                    if float(pkt.time) - flow_start_time > interval:
                        flow_i.append((fid, subflow))
                        flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
                        subflow = [(float(pkt.time), pkt)]
                        split_flow = True
                    else:
                        subflow.append((float(pkt.time), pkt))

                # handle UDP packets
                elif IP in pkt and UDP in pkt:
                    # parse 5-tuple flow ID
                    fid = (pkt[IP].src, pkt[IP].dst, pkt[UDP].sport, pkt[UDP].dport, 17)
                    flow_type = 'UDP'
                    if float(pkt.time) - flow_start_time > interval:
                        flow_i.append((fid, subflow))
                        flow_start_time += int((float(pkt.time) - flow_start_time) // interval) * interval
                        subflow = [(float(pkt.time), pkt)]
                        split_flow = True
                    else:
                        subflow.append((float(pkt.time), pkt))

            if (split_flow == False) and (flow_type in ['TCP', 'UDP']):
                new_cnt += 1
                flow_i.append((fid, subflow))
            else:
                # drop the last subflow after splitting a flow
                remainder_cnt += 1
                # flow_i.append((fid, subflow)) # don't include the remainder
                # print(i, new_flow, subflow)

            # drop the last one which interval is less than interval
            if step == 0:
                flows.extend(flow_i)

            step_flows.extend(flow_i)
            if data_name.upper() in ['DS60_UChi_IoT', 'SCAM1', 'scam1', 'GHOM1',
                                     'SFRIG1'] and abnormal:  # only augment abnormal flows
                step += 5  # 10 is the step for sampling, 'agument' anomaly files in DS60
            else:
                break

    print(
        f'tmp_arr2: {len(tmp_arr2)},tmp_arr1: {len(tmp_arr1)}, all_flows: {len(full_flows)}, subflows: {len(flows)}, step_flows: {len(step_flows)}, {data_name}, remain_subflow: {len(subflow)}')

    # sort all flows by packet arrival time, each flow must have at least two packets
    flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in flows if
             len(times_pkts) >= max(2, num_pkt_thresh)]
    flows = [(fid, pkts) for fid, times, pkts in flows]
    print(f'the final subflows: len(flows): {len(flows)}, each of them has more than 2 pkts.')

    # sort all flows by packet arrival time, each flow must have at least two packets
    step_flows = [(fid, *list(zip(*sorted(times_pkts)))) for fid, times_pkts in step_flows if
                  len(times_pkts) >= max(2, num_pkt_thresh)]
    step_flows = [(fid, pkts) for fid, times, pkts in step_flows]
    print(f'the final step_flows: len(step_flows): {len(step_flows)}, each of them has more than 2 pkts.')
    if abnormal:
        return step_flows

    return flows


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


def _get_path(original_dir, in_dir, data_name, overwrite=False, direction='src'):
    """

    Parameters
    ----------
    in_dir
    data_name
    overwrite
    direction: str
        src_dst: use src + dst data
        src: only user src data

    Returns
    -------

    """
    if 'UNB/CICIDS_2017/pc_' in data_name and 'Mon' not in data_name:
        ##############################################################################################################
        # step 1: get path
        if data_name == 'UNB/CICIDS_2017/pc_192.168.10.5':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.5.pcap')
            pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.5.csv')
            if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
                in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
                keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.5'], direction=direction)
                # label_file
                in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
                    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
                out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
                merge_csvs(in_files, out_file)
                keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.5'], direction=direction, keep_original=True,
                            verbose=10)
            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.8':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.8.pcap')
            pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.8.csv')
            if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
                in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
                keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.8'], direction=direction)
                # label_file
                in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
                    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
                out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
                merge_csvs(in_files, out_file)
                keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.8'], direction=direction, keep_original=True,
                            verbose=10)

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.9':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.9.pcap')
            pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.9.csv')
            if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
                in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
                keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.9'], direction=direction)
                # label_file
                in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
                    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
                out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
                merge_csvs(in_files, out_file)
                keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.9'], direction=direction, keep_original=True,
                            verbose=10)
            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None


        elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.14':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.14.pcap')
            pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.14.csv')
            if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
                in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
                keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.14'], direction=direction)
                # label_file
                in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
                    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
                out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
                merge_csvs(in_files, out_file)
                keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.14'], direction=direction, keep_original=True,
                            verbose=10)
            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'UNB/CICIDS_2017/pc_192.168.10.15':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.15.pcap')
            pth_labels_mixed = pth.join(in_dir, direction, data_name, 'pc_192.168.10.15.csv')
            if not os.path.exists(pth_pcap_mixed) or not os.path.exists(pth_labels_mixed):
                in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Friday/Friday-WorkingHours.pcap')
                keep_ip(in_file, out_file=pth_pcap_mixed, kept_ips=['192.168.10.15'], direction=direction)
                # label_file
                in_files = [pth.join(original_dir, 'UNB/CICIDS_2017', 'labels/Friday', v) for v in [
                    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']]
                out_file = pth.join(in_dir, direction, data_name, 'Friday_labels.csv')
                merge_csvs(in_files, out_file)
                keep_csv_ip(out_file, pth_labels_mixed, ips=['192.168.10.15'], direction=direction, keep_original=True,
                            verbose=10)

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
        if data_name == 'UCHI/IOT_2019/smtv_10.42.0.1':
            # # normal and abormal are independent
            #  editcap -c 500000 merged.pcap merged.pcap
            pth_normal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.1.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.119-abnormal.pcap')
            # pth_labels_normal, pth_labels_abnormal = None, None

            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                merged_pcap = pth.join(in_dir, direction, data_name, 'merged.pcap')
                if not os.path.exists(merged_pcap):
                    pcap_files = [pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps', v)
                                  for v in os.listdir(
                            pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps')) if
                                  not v.startswith('.')]
                    merge_pcaps(in_files=pcap_files, out_file=merged_pcap)

                # 10.42.0.1 date is from the whole megerd pcap
                keep_ip(merged_pcap, out_file=pth_normal, kept_ips=['10.42.0.1'], direction=direction)

                merged_pcap = pth.join(in_dir, direction, data_name, 'pc_10.42.0.119_00000_20190927224625.pcap')
                keep_ip(merged_pcap, out_file=pth_abnormal, kept_ips=['10.42.0.119'], direction=direction)
                # copyfile(idle_pcap, pth_normal)

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2019/smtv_10.42.0.119':
            # # normal and abormal are independent
            #  editcap -c 500000 merged.pcap merged.pcap
            pth_normal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.119.pcap', )
            pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_10.42.0.1.pcap')
            # pth_labels_normal, pth_labels_abnormal = None, None

            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                merged_pcap = pth.join(in_dir, direction, data_name, 'merged_00000_20190927182134.pcap')
                if not os.path.exists(merged_pcap):
                    pcap_files = [pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps', v)
                                  for v in os.listdir(
                            pth.join(original_dir, 'UCHI/IOT_2019', 'smtv/roku-data-20190927-182117/pcaps')) if
                                  not v.startswith('.')]
                    merge_pcaps(in_files=pcap_files, out_file=merged_pcap)
                keep_ip(merged_pcap, out_file=pth_normal, kept_ips=['10.42.0.119'], direction=direction)
                keep_ip(merged_pcap, out_file=pth_abnormal, kept_ips=['10.42.0.1'], direction=direction)
                # copyfile(idle_pcap, pth_normal)

            pth_labels_normal, pth_labels_abnormal = None, None


        elif data_name == 'OCS1/IOT_2018/pc_192.168.0.13':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'pc_192.168.0.13-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_192.168.0.13-anomaly.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'OCS/IOT_2018', 'pcaps',
                                   'benign-dec.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.0.13'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'OCS/IOT_2018', 'pcaps',
                                   'mirai-udpflooding-2-dec.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.0.13'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'DS40_CTU_IoT/DS41-srcIP_10.0.2.15':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name,
                                  '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_anomaly.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_normal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU/IOT_2017/pc_192.168.1.196':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name,
                                  '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
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

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU/IOT_2017/pc_10.0.2.15_192.168.1.195':
            """
            normal_traffic:
                https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/ (around 1100 flows)
                https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-22/ 
            """
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, '2017-04-30_CTU-win-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-04-30_win-normal.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['10.0.2.15'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'CTU/IOT_2017',
                                   'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU/IOT_2017/pc_10.0.2.15_192.168.1.196':
            """
                        normal_traffic:
                            https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/
                        """
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, '2017-04-30_CTU-win-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-04-30_win-normal.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['10.0.2.15'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'CTU/IOT_2017',
                                   '2019-01-09-22-46-52-pc_192.168.1.196_CTU_IoT_CoinMiner_normal.pcap.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.196'], direction=direction)

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU/IOT_2017/pc_192.168.1.191_192.168.1.195':
            """
            normal_traffic:
                https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/ (around 1100 flows)
                https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-22/ 
            """
            ## editcap -c 500000 2017-05-02_kali.pcap 2017-05-02_kali.pcap

            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, '2017-05-02_CTU-kali-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '2018-12-21-15-50-14-pc_192.168.1.195-CTU_IoT_Mirai_abnormal.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'CTU/IOT_2017', '2017-05-02_kali_00000_20170502082205.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.1.191'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'CTU/IOT_2017',
                                   'CTU-IoT-Malware-Capture-34-1_2018-12-21-15-50-14-192.168.1.195.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.1.195'], direction=direction)

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU/IOT_2017/pc_192.168.1.191_192.168.1.196':
            """
            normal_traffic:
                https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-20/
            """
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, '2017-05-02_CTU-kali-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
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

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_202.171.168.50':
            # editcap -c 30000000 samplepoint-F_201912071400.pcap samplepoint-F_201912071400.pcap
            # file_name = 'samplepoint-F_201912071400_00000_20191207000000.pcap'
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name,
                                  '201912071400-pc_202.171.168.50_normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '201912071400-pc_202.4.27.109_anomaly.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                if direction == 'src_dst':
                    in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
                                       'samplepoint-F_201912071400-src_dst_202.171.168.50-5000000.pcap')
                    # editcap -c 5000000 samplepoint-F_201912071400-src_dst_202.171.168.50.pcap samplepoint-F_201912071400-src_dst_202.171.168.50-.pcap
                else:
                    in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
                                       'samplepoint-F_201912071400-src_dst_202.171.168.50.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['202.171.168.50'], direction=direction)

                # in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
                #                    'samplepoint-F_201912071400_00000_20191207000000.pcap')
                in_file = pth.join(original_dir, 'MAWI/WIDE_2019',
                                   'samplepoint-F_201912071400-src_dst_202.4.27.109.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['202.4.27.109'], direction=direction)

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2020/pc_203.78.7.165':
            # normal and abormal are independent
            # editcap -c 30000000 samplepoint-F_202007011400.pcap samplepoint-F_202007011400.pcap
            # tshark -r samplepoint-F_202007011400.pcap -w 202007011400-pc_203.78.7.165.pcap ip.addr==203.78.7.165
            pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.7.165.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '202007011400-pc_185.8.54.240.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400_00000_20200701010000.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.7.165'], direction=direction)

                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400_00000_20200701010000.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['185.8.54.240'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2020/pc_203.78.4.32':
            pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.4.32.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '202007011400-pc_202.75.33.114.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400.pcap-src_dst_203.78.4.32.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.4.32'], direction=direction)

                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400.pcap-src_dst_202.75.33.114.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['202.75.33.114'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2020/pc_203.78.4.32-2':

            pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.4.32.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '202007011400-pc_203.78.8.151.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400.pcap-src_dst_203.78.4.32.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.4.32'], direction=direction)

                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400.pcap-src_dst_203.78.8.151.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['203.78.8.151'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2020/pc_203.78.7.165-2':
            pth_normal = pth.join(in_dir, direction, data_name, '202007011400-pc_203.78.7.165.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    '202007011400-pc_203.78.8.151.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400-src_dst_203.78.7.165.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['203.78.7.165'], direction=direction)

                in_file = pth.join(original_dir, 'MAWI/WIDE_2020',
                                   'samplepoint-F_202007011400.pcap-src_dst_203.78.8.151.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['203.78.8.151'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_203.78.4.32':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.4.32.pcap')
            pth_abnormal = pth.join(in_dir, data_name,
                                    '202007011400-srcIP_203.78.7.165.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_222.117.214.171':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(in_dir, data_name,
                                    '202007011400-srcIP_222.117.214.171.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_101.27.14.204':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.7.165.pcap')
            pth_abnormal = pth.join(in_dir, data_name,
                                    '202007011400-srcIP_101.27.14.204.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MAWI/WIDE_2019/pc_18.178.219.109':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, data_name, '202007011400-srcIP_203.78.4.32.pcap')
            pth_abnormal = pth.join(in_dir, data_name,
                                    '202007011400-srcIP_18.178.219.109.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2019/ghome_192.168.143.20':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'ghome_192.168.143.20-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'ghome_192.168.143.20-anomaly.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/ghome_192.168.143.20',
                                   'fridge_cam_sound_ghome_2daysactiv-ghome_normal.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.20'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/ghome_192.168.143.20',
                                   'fridge_cam_sound_ghome_2daysactiv-ghome_abnormal.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.20'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2019/scam_192.168.143.42':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'scam_192.168.143.42-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    'scam_192.168.143.42-anomaly.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/scam_192.168.143.42',
                                   'fridge_cam_sound_ghome_2daysactiv-scam_normal.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.42'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/scam_192.168.143.42',
                                   'fridge_cam_sound_ghome_2daysactiv-scam_abnormal.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.42'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2019/sfrig_192.168.143.43':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'sfrig_192.168.143.43-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    'sfrig_192.168.143.43-anomaly.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/sfrig_192.168.143.43',
                                   'fridge_cam_sound_ghome_2daysactiv-sfrig_normal.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.43'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/sfrig_192.168.143.43',
                                   'fridge_cam_sound_ghome_2daysactiv-sfrig_abnormal.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.43'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2019/bstch_192.168.143.48':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'bstch_192.168.143.48-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name,
                                    'bstch_192.168.143.48-anomaly.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/bstch_192.168.143.48',
                                   'fridge_cam_sound_ghome_2daysactiv-bstch_normal.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.143.48'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'UCHI/IOT_2019/bstch_192.168.143.48',
                                   'fridge_cam_sound_ghome_2daysactiv-bstch_abnormal.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.143.48'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        # elif data_name == 'UCHI/IOT_2019/pc_192.168.143.43':
        #     # normal and abormal are independent
        #     # 'idle'
        #     # 'iotlab_open_shut_fridge_192.168.143.43/open_shut'
        #     pth_normal = pth.join(in_dir, data_name, 'bose_soundtouch-2daysactiv-src_192.168.143.48-normal.pcap')
        #     pth_abnormal = pth.join(in_dir, data_name,
        #                             'bose_soundtouch-2daysactiv-src_192.168.143.48-anomaly.pcap')
        #     pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2020/aecho_192.168.143.74':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'idle-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'shop-anomaly.pcap')
            # pth_abnormal = pth.join(in_dir, direction, data_name)   # directory
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
                                          out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
                                          kept_ips=['192.168.143.74'], direction=direction)
                                  for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
                                  not v.startswith('.')]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_normal)
                # abnormal
                # abnormal_file = pth.join(original_dir, data_name, 'echo_song.pcap')
                # Can not use the whole abnormal pcap directly because when we split it to subpcap,
                # one flow will split to multi-flows.
                # pth_abnormal = keep_ip(abnormal_file, out_file=pth_abnormal, kept_ips=['192.168.143.74'],
                #                        direction=direction)
                activity = 'shop'
                whole_abnormal = pth.join(original_dir, data_name, f'echo_{activity}.pcap')
                num = split_by_activity(whole_abnormal, out_dir=os.path.dirname(whole_abnormal), activity=activity)
                idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, data_name, v),
                                          out_file=pth.join(os.path.dirname(idle_pcap), v),
                                          kept_ips=['192.168.143.74'], direction=direction)
                                  for v in [f'{activity}/capture{i}.seq/deeplens_{activity}_{i}.' \
                                            f'pcap' for i in range(num)]
                                  ]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_abnormal)

            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2020/sfrig_192.168.143.43':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'idle.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'open_shut.pcap')
            # pth_abnormal = pth.join(in_dir, direction, data_name, 'browse.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
                                          out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
                                          kept_ips=['192.168.143.43'], direction=direction)
                                  for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
                                  not v.startswith('.')]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_normal)

                # abnormal
                idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, data_name, v),
                                          out_file=pth.join(os.path.dirname(idle_pcap), v),
                                          kept_ips=['192.168.143.43'], direction=direction)
                                  for v in [f'open_shut/capture{i}.seq/deeplens_open_shut_fridge_batch_{i}.' \
                                            f'pcap_filtered.pcap' for i in range(9)]
                                  ]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_abnormal)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2020/wshr_192.168.143.100':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'idle.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'open_wshr.pcap')
            # pth_abnormal = pth.join(in_dir, direction, data_name, 'browse.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
                                          out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
                                          kept_ips=['192.168.143.100'], direction=direction)
                                  for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
                                  not v.startswith('.')]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_normal)

                # abnormal
                idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, data_name, v),
                                          out_file=pth.join(os.path.dirname(idle_pcap), v),
                                          kept_ips=['192.168.143.100'], direction=direction)
                                  for v in [f'open_wshr/capture{i}.seq/deeplens_open_washer_{i}.' \
                                            f'pcap' for i in range(31)]
                                  ]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_abnormal)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2020/dwshr_192.168.143.76':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'idle.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'open_dwshr.pcap')
            # pth_abnormal = pth.join(in_dir, direction, data_name, 'browse.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                idle_pcap = pth.join(in_dir, direction, data_name, 'idle-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, 'UCHI/IOT_2020', 'idle', v),
                                          out_file=os.path.dirname(idle_pcap) + f'/{v}-filtered.pcap',
                                          kept_ips=['192.168.143.76'], direction=direction)
                                  for v in os.listdir(pth.join(original_dir, 'UCHI/IOT_2020', 'idle')) if
                                  not v.startswith('.') and 'pcap' in v]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_normal)

                # abnormal
                idle_pcap = pth.join(in_dir, direction, data_name, 'abnormal-merged.pcap')
                if not os.path.exists(idle_pcap):
                    idle_files = [keep_ip(pth.join(original_dir, data_name, v),
                                          out_file=pth.join(os.path.dirname(idle_pcap), v),
                                          kept_ips=['192.168.143.76'], direction=direction)
                                  for v in [f'open_dwshr/capture{i}.seq/deeplens_open_dishwasher_{i}.' \
                                            f'pcap_filtered.pcap' for i in range(31)]
                                  ]
                    merge_pcaps(in_files=idle_files, out_file=idle_pcap)
                copyfile(idle_pcap, pth_abnormal)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'UCHI/IOT_2020/ghome_192.168.143.20':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, data_name, 'google_home-2daysactiv-src_192.168.143.20-normal.pcap')
            pth_abnormal = pth.join(in_dir, data_name,
                                    'google_home-2daysactiv-src_192.168.143.20-anomaly.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'WRCCDC/2020-03-20':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, data_name, 'wrccdc.2020-03-20.174351000000002-172.16.16.30-normal.pcap')
            # pth_abnormal = pth.join(in_dir, data_name,
            #                         'wrccdc.2020-03-20.174351000000002-172.16.16.16.pcap')
            pth_abnormal = pth.join(in_dir, data_name,
                                    'wrccdc.2020-03-20.174351000000002-10.183.250.172-abnormal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None
        elif data_name == 'DEFCON/ctf26':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, data_name, 'DEFCON26ctf_packet_captures-src_10.0.0.2-normal.pcap')
            pth_abnormal = pth.join(in_dir, data_name,
                                    'DEFCON26ctf_packet_captures-src_10.13.37.23-abnormal.pcap')
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'ISTS/ISTS_2015':
            # normal and abormal are independent
            # pth_normal = pth.join(in_dir, data_name, 'snort.log.1425741051-src_10.128.0.13-normal.pcap')
            # pth_normal = pth.join(in_dir, data_name, 'snort.log.1425823409-src_10.2.1.80.pcap')
            # pth_normal = pth.join(in_dir, data_name, 'snort.log.1425824560-src_129.21.3.17.pcap')
            # pth_normal = pth.join(in_dir, data_name,
            #                       'snort.log-merged-srcIP_10.128.0.13-10.0.1.51-10.0.1.4-10.2.12.40.pcap')
            #
            # pth_abnormal = pth.join(in_dir, data_name,
            #                         'snort.log-merged-srcIP_10.2.12.50.pcap')

            pth_normal = pth.join(in_dir, direction, data_name, 'snort.log-merged-3pcaps-normal.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'snort.log.1425824164.pcap')
            if not pth.exists(pth_normal) or not pth.exists(pth_abnormal):
                in_files = [
                    'snort.log.1425741002.pcap',
                    'snort.log.1425741051.pcap',
                    'snort.log.1425823409.pcap',
                    # 'snort.log.1425842738.pcap',
                    # 'snort.log.1425824560.pcap',

                ]
                # in_file = 'data/data_reprst/pcaps/ISTS/2015/snort.log.1425824164.pcap' # for abnormal dataset
                in_files = [os.path.join(original_dir, data_name, v) for v in in_files]
                out_file = os.path.join(in_dir, direction, data_name, 'snort.log-merged-3pcaps.pcap')
                merge_pcaps(in_files, out_file)
                copyfile(out_file, pth_normal)
                in_file = pth.join(original_dir, data_name, 'snort.log.1425824164.pcap')
                copyfile(in_file, pth_abnormal)
            # pth_abnormal = pth.join(in_dir, data_name, 'snort.log-merged-srcIP_10.2.4.30.pcap')
            # if not pth.exists(pth_abnormal):
            #     out_file = keep_ip(pth_normal, out_file=pth_abnormal, kept_ips=['10.2.4.30'])
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'MACCDC/MACCDC_2012/pc_192.168.202.79':
            # normal and abormal are independent
            # pth_normal = pth.join(in_dir, data_name, 'maccdc2012_00000-srcIP_192.168.229.153.pcap')
            # the result does beat OCSVM.
            pth_normal = pth.join(in_dir, direction, data_name, 'maccdc2012_00000-pc_192.168.202.79.pcap')
            pth_abnormal = pth.join(in_dir, direction, data_name, 'maccdc2012_00000-pc_192.168.202.76.pcap')
            if not os.path.exists(pth_normal) or not os.path.exists(pth_abnormal):
                # normal
                in_file = pth.join(original_dir, 'MACCDC/MACCDC_2012', 'maccdc2012_00000.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.202.79'], direction=direction)
                # abnormal
                in_file = pth.join(original_dir, 'MACCDC/MACCDC_2012', 'maccdc2012_00000.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.202.76'], direction=direction)
            pth_labels_normal, pth_labels_abnormal = None, None

        elif data_name == 'CTU_IOT23/CTU-IoT-Malware-Capture-7-1':
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(in_dir, data_name, '2018-07-20-17-31-20-192.168.100.108.pcap')
            pth_labels_mixed = pth.join(in_dir, data_name,
                                        'CTU-IoT-Malware-Capture-7-1-conn.log.labeled.txt.csv-src_192.168.100.108.csv')

            pth_normal, pth_abnormal, pth_labels_normal, pth_labels_abnormal = pth_pcap_mixed, None, pth_labels_mixed, None

        elif data_name == 'UNB/CICIDS_2017/Mon-pc_192.168.10.5':
            # normal and abormal are independent
            pth_normal = pth.join(in_dir, direction, data_name, 'pc_192.168.10.5.pcap')
            if not os.path.exists(pth_normal):
                in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Monday/Monday-WorkingHours.pcap')
                keep_ip(in_file, out_file=pth_normal, kept_ips=['192.168.10.5'], direction=direction)

            # normal and abormal are mixed together
            pth_abnormal = pth.join(in_dir, direction, data_name, 'pc_192.168.10.8.pcap')
            if not os.path.exists(pth_abnormal):
                in_file = pth.join(original_dir, 'UNB/CICIDS_2017', 'pcaps/Monday/Monday-WorkingHours.pcap')
                keep_ip(in_file, out_file=pth_abnormal, kept_ips=['192.168.10.8'], direction=direction)

            pth_labels_normal, pth_labels_abnormal = None, None

        else:
            print('debug')
            data_name = 'DEMO_IDS/DS-srcIP_192.168.10.5'
            # normal and abormal are mixed together
            pth_pcap_mixed = pth.join(in_dir, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap')
            pth_labels_mixed = pth.join(in_dir, data_name, 'AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.csv')

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

    print(f'normal_file: {normal_file}, exists: {pth.exists(normal_file)}')
    print(f'abnormal_file: {abnormal_file}, exists: {pth.exists(abnormal_file)}')

    return normal_file, abnormal_file


class New_PCAP(PCAP):
    def __init__(self, pcap_file='xxx.pcap', *, flow_pkts_thres=2, verbose=10, random_state=42, sampling_rate=0.1):
        super(New_PCAP, self).__init__()
        # self.q_samps = q_samp
        self.verbose = verbose
        self.random_state = random_state
        self.flow_pkts_thres = flow_pkts_thres
        self.sampling_rate = sampling_rate

    @timing
    def _flow2features(self, feat_type='IAT', *, fft=False, header=False, dim=None):
        """Extract features from each flow according to feat_type, fft and header.

        Parameters
        ----------
        feat_type: str (default is 'IAT')
            which features do we want to extract from flows

        fft: boolean (default is False)
            if we need fft-features

        header: boolean (default is False)
            if we need header+features
        dim: dim of the "SIZE" feature

        Returns
        -------
            self
        """
        self.feat_type = feat_type

        if dim is None:
            num_pkts = [len(pkts) for fid, pkts in self.flows]
            dim = int(np.floor(np.quantile(num_pkts, self.q_interval)))  # use the same q_interval to get the dimension

        if feat_type in ['IAT', 'FFT-IAT']:
            self.dim = dim - 1
            self.features, self.fids = _get_IAT(self.flows)
        elif feat_type in ['SIZE', 'FFT-SIZE']:
            self.dim = dim
            self.features, self.fids = _get_SIZE(self.flows)
        elif feat_type in ['IAT_SIZE', 'FFT-IAT_SIZE']:
            self.dim = 2 * dim - 1
            self.features, self.fids = _get_IAT_SIZE(self.flows)
        elif feat_type in ['STATS']:
            self.dim = 10
            self.features, self.fids = _get_STATS(self.flows)
        elif feat_type in ['SAMP_NUM', 'FFT-SAMP_NUM']:
            self.dim = dim - 1
            # flow_durations = [_get_flow_duration(pkts) for fid, pkts in self.flows]
            # # To obtain different samp_features, you should change q_interval ((0, 1))
            # sampling_rate = _get_split_interval(flow_durations, q_interval=self.q_interval)
            self.features, self.fids = _get_SAMP_NUM(self.flows, self.sampling_rate)
        elif feat_type in ['SAMP_SIZE', 'FFT-SAMP_SIZE']:
            self.dim = dim - 1  # here the dim of "SAMP_SIZE" is dim -1, which equals to the dimension of 'SAMP_NUM'
            self.features, self.fids = _get_SAMP_SIZE(self.flows, self.sampling_rate)
        else:
            msg = f'feat_type ({feat_type}) is not correct! '
            raise ValueError(msg)

        print(f'self.dim: {self.dim}, feat_type: {feat_type}')
        if fft:
            self.features = _get_FFT_data(self.features, fft_bin=self.dim)
        else:
            # fix each flow to the same feature dimension (cut off the flow or append 0 to it)
            self.features = [v[:self.dim] if len(v) > self.dim else v + [0] * (self.dim - len(v)) for v in
                             self.features]

        if header:
            _headers = _get_header_features(self.flows)
            h_dim = 8 + dim  # 8 TCP flags
            if fft:
                fft_headers = _get_FFT_data(_headers, fft_bin=h_dim)
                self.features = [h + f for h, f in zip(fft_headers, self.features)]
            else:
                # fix header dimension firstly
                headers = [h[:h_dim] if len(h) > h_dim else h + [0] * (h_dim - len(h)) for h in _headers]
                self.features = [h + f for h, f in zip(headers, self.features)]

        # change list to numpy array
        self.features = np.asarray(self.features, dtype=float)
        if self.verbose > 5:
            print(np.all(self.features >= 0))


def _subflows2featutes(flows, labels, dim=10, feat_type='IAT_SIZE', sampling_rate=0.1,
                       header=False, verbose=10):
    # extract features from each flow given feat_type

    pp = New_PCAP(sampling_rate=sampling_rate)
    pp.flows = flows
    pp.labels = labels
    pp.flow2features(feat_type.upper(), fft=False, header=header, dim=dim)
    # out_file = f'{out_dir}/features-q_interval:{q_interval}.dat'
    # print('features+labels: ', out_file)
    # features = pp.features
    # labels = pp.labels
    # dump_data((features, labels), out_file)

    return pp.features, pp.labels


def generate_data_speed_up(data_name='', out_file='', random_state=42, direction='src_dst',
                           feat_type='IAT_SIZE', header=False, overwrite=False):
    print(f'feat_type: {feat_type}, header={header}')
    print(generate_data_speed_up.__dict__)
    # data_type = 'one_dataset'
    original_dir = f'../../Datasets'
    in_dir = f'speedup/data'
    if data_name in ['mimic_GMM', 'mimic_GMM1', 'CinC']:
        out_file = _generate_mimic_data(data_type=data_name, random_state=random_state, out_file=out_file)
    elif data_name in ['UNB1', 'UNB2', 'UNB3', 'UNB4', 'UNB5',
                       'UNB2_UNB1', 'UNB2_UNB3']:  # mix UNB1 and UNB2
        # pcaps and flows directory
        # in_dir = f'./data/data_reprst/pcaps'
        if data_name == 'UNB1':
            subdatasets = (
                'UNB/CICIDS_2017/pc_192.168.10.5',
            )  # each_data has normal and abnormal
        elif data_name == 'UNB2':
            subdatasets = (
                'UNB/CICIDS_2017/pc_192.168.10.8',
            )  # each_data has normal and abnormal
        elif data_name == 'UNB3':
            subdatasets = (
                'UNB/CICIDS_2017/pc_192.168.10.9',
            )  # each_data has normal and abnormal
        elif data_name == 'UNB4':
            subdatasets = (
                'UNB/CICIDS_2017/pc_192.168.10.14',
            )  # each_data has normal and abnormal

        elif data_name == 'UNB5':
            subdatasets = (
                'UNB/CICIDS_2017/pc_192.168.10.15',
            )  # each_data has normal and abnormal
        # elif data_name == 'UNB1_UNB5':
        #     subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
        #                    'UNB/CICIDS_2017/pc_192.168.10.15')  # each_data has normal and abnormal
        # elif data_name == 'UNB2_UNB3':
        #     subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8', 'UNB/CICIDS_2017/pc_192.168.10.9')
        # elif data_name == 'UNB3_UNB4':
        #     subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.9', 'UNB/CICIDS_2017/pc_192.168.10.14')
        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file
    elif data_name in ['UNB12', 'UNB13', 'UNB24', ]:  # mix UNB1 and others
        if data_name == 'UNB12':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.8')  # each_data has normal and abnormal
        elif data_name == 'UNB13':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.9')  # each_data has normal and abnormal
        elif data_name == 'UNB24':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal

        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        normal_files, abnormal_files = [normal_files[0]], [normal_files[1]]
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file
        return out_file
    elif data_name in ['UNB12_comb', 'UNB13_comb', 'UNB14_comb',
                       'UNB23_comb', 'UNB24_comb', 'UNB25_comb',
                       'UNB34_comb', 'UNB35_comb', 'UNB45_comb', ]:  # mix UNB1 and others

        if data_name == 'UNB12_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.8')  # each_data has normal and abnormal
        elif data_name == 'UNB13_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.9')  # each_data has normal and abnormal
        elif data_name == 'UNB14_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB23_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.9')  # each_data has normal and abnormal
        elif data_name == 'UNB24_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB34_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.9',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB35_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.9',
                           'UNB/CICIDS_2017/pc_192.168.10.15')  # each_data has normal and abnormal
        elif data_name == 'UNB45_comb':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.14',
                           'UNB/CICIDS_2017/pc_192.168.10.15')  # each_data has normal and abnormal

        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file
        return out_file
    elif data_name in ['UNB12_1', 'UNB13_1', 'UNB14_1',
                       'UNB23_2', 'UNB24_2', 'UNB25_1',
                       'UNB34_3', 'UNB35_3', 'UNB45_1',

                       'UNB123_1',  # combine UNB1, UNB2, UNB3 attacks, only use UNB1 normal
                       'UNB134_1',
                       'UNB145_1',
                       'UNB234_2',  # combine UNB2, UNB3, UNB4 attacks, only use UNB2 normal
                       'UNB245_2',
                       'UNB345_3',
                       ]:  # mix UNB1 and others

        if data_name == 'UNB12_1':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.8')  # each_data has normal and abnormal
        elif data_name == 'UNB13_1':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.9')  # each_data has normal and abnormal
        elif data_name == 'UNB14_1':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB23_2':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.9')  # each_data has normal and abnormal
        elif data_name == 'UNB24_2':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB34_3':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.9',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB35_3':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.9',
                           'UNB/CICIDS_2017/pc_192.168.10.15')  # each_data has normal and abnormal
        elif data_name == 'UNB45_4':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.14',
                           'UNB/CICIDS_2017/pc_192.168.10.15')  # each_data has normal and abnormal

        elif data_name == 'UNB123_1':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.9')  # each_data has normal and abnormal
        elif data_name == 'UNB134_1':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.9',
                           'UNB/CICIDS_2017/pc_192.168.10.14')  # each_data has normal and abnormal
        elif data_name == 'UNB145_1':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.5',
                           'UNB/CICIDS_2017/pc_192.168.10.14',
                           'UNB/CICIDS_2017/pc_192.168.10.15')  # each_data has normal and abnormal
        elif data_name == 'UNB234_2':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.9',
                           'UNB/CICIDS_2017/pc_192.168.10.14',)  # each_data has normal and abnormal
        elif data_name == 'UNB245_2':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8',
                           'UNB/CICIDS_2017/pc_192.168.10.14',
                           'UNB/CICIDS_2017/pc_192.168.10.15',)  # each_data has normal and abnormal
        elif data_name == 'UNB345_3':
            subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.9',
                           'UNB/CICIDS_2017/pc_192.168.10.14',
                           'UNB/CICIDS_2017/pc_192.168.10.15',)  # each_data has normal and abnormal

        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        normal_files = [normal_files[0]]
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file
        return out_file
    elif data_name in ['CTU1', 'CTU21', 'CTU22', 'CTU31', 'CTU32', 'MAWI1_2019', 'ISTS1', 'MACCDC1', 'OCS1',
                       'MAWI1_2020', 'MAWI32_2020',
                       'MAWI32-2_2020', 'MAWI165-2_2020', 'SMTV1_2019', 'SMTV2_2019',
                       'UNB2_CTU1', 'UNB2_MAWI1', 'UNB1_ISTS1', 'UNB_5_8_Mon', ]:  # mix UNB1 and others
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
        elif data_name == 'OCS1':
            subdatasets = ('OCS1/IOT_2018/pc_192.168.0.13',)
        elif data_name == 'MAWI1_2019':
            # subdatasets = ('MAWI/WIDE_2020/pc_203.78.7.165',)
            subdatasets = ('MAWI/WIDE_2019/pc_202.171.168.50',)  # not work well
        elif data_name == 'MAWI1_2020':
            subdatasets = ('MAWI/WIDE_2020/pc_203.78.7.165',)
        elif data_name == 'MAWI32_2020':
            subdatasets = ('MAWI/WIDE_2020/pc_203.78.4.32',)
        elif data_name == 'MAWI32-2_2020':
            subdatasets = ('MAWI/WIDE_2020/pc_203.78.4.32-2',)
        elif data_name == 'MAWI165-2_2020':
            subdatasets = ('MAWI/WIDE_2020/pc_203.78.7.165-2',)

        elif data_name == 'SMTV1_2019':
            subdatasets = ('UCHI/IOT_2019/smtv_10.42.0.1',)
        elif data_name == 'SMTV2_2019':
            subdatasets = ('UCHI/IOT_2019/smtv_10.42.0.119',)

        elif data_name == 'ISTS1':
            subdatasets = ('ISTS/ISTS_2015',)
        elif data_name == 'MACCDC1':
            subdatasets = ('MACCDC/MACCDC_2012/pc_192.168.202.79',)
        elif data_name == 'UNB_5_8_Mon':
            subdatasets = ('UNB/CICIDS_2017/Mon-pc_192.168.10.5',
                           )  # use Monday data, and pc_5 as normal and pc_8 as abnormal
        # elif data_name == 'UNB1_ISTS2':
        #     subdatasets = ('UNB/CICIDS_2017/pc_192.168.10.8', 'MAWI/WIDE_2019/pc_203.78.7.165')
        # elif data_name == 'CTU1_MAWI1':
        #     subdatasets = ('CTU/IOT_2017/pc_192.168.1.196', 'MAWI/WIDE_2019/pc_203.78.7.165')
        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file
    elif data_name in [  # 2019 IoT dataset
        'SCAM1', 'CTU1_SCAM1', 'MAWI1_SCAM1',
        'GHOM1', 'SFRIG1',
    ]:
        if data_name == 'SCAM1':
            subdatasets = ('UCHI/IOT_2019/scam_192.168.143.42',)
        elif data_name == 'GHOM1':
            subdatasets = ('UCHI/IOT_2019/ghome_192.168.143.20',)  # each_data has normal and abnormal
        elif data_name == 'SFRIG1':
            subdatasets = ('UCHI/IOT_2019/sfrig_192.168.143.43',)

        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        normal_files, abnormal_files = [normal_files[0]], [normal_files[1]]
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        out_file = pf.Xy_file
        return out_file
    elif data_name in [
        'SFRIG1_2020',
        'AECHO1_2020',
        'WSHR_2020', 'DWSHR_2020', 'DRYER_2020',
    ]:
        if data_name == 'SFRIG1_2020':
            subdatasets = ('UCHI/IOT_2020/sfrig_192.168.143.43',)
        elif data_name == 'AECHO1_2020':
            subdatasets = ('UCHI/IOT_2020/aecho_192.168.143.74',)
        elif data_name == 'WSHR_2020':
            subdatasets = ('UCHI/IOT_2020/wshr_192.168.143.100',)
        elif data_name == 'DWSHR_2020':
            subdatasets = ('UCHI/IOT_2020/dwshr_192.168.143.76',)
        elif data_name == 'DRYER_2020':
            subdatasets = ('UCHI/IOT_2020/dryer_192.168.143.99',)

        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        # 1) get_files
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        # 2) full_flows
        (normal_flows, normal_labels), load_time = time_func(load_data, normal_files[0])
        (abnormal_flows, abnormal_labels), load_time = time_func(load_data, abnormal_files[0])
        # 3) get the interval from normal flows, and use it to split flows
        durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
        q_interval = 0.9
        interval = _get_split_interval(durations, q_interval)
        print(f'interval {interval} when q_interval: {q_interval}')
        # 4) get subflows
        normal_flows = flows2subflows_SCAM(normal_flows, interval=interval, num_pkt_thresh=2, data_name=data_name,
                                           abnormal=False)
        normal_labels = [normal_labels[0]] * len(normal_flows)
        abnormal_flows = flows2subflows_SCAM(abnormal_flows, interval=interval, num_pkt_thresh=2, data_name=data_name,
                                             abnormal=True)
        abnormal_labels = [abnormal_labels[0]] * len(abnormal_flows)
        # 5). subflows2features
        num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
        dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
        print(f'dim={dim}')
        if feat_type.upper().startswith('SAMP_'):
            X = {}
            y = {}
            flow_durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
            for q_samp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                # get sampling_rate on normal_flows first
                # print(f'np.quantile(flows_durations): {np.quantile(flow_durations, q=[0.1, 0.2, 0.3, 0.9, 0.95])}')
                sampling_rate = _get_split_interval(flow_durations, q_interval=q_samp)
                if sampling_rate <= 0.0: continue
                print(f'sampling_rate: {sampling_rate}, q = {q_samp}')
                X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
                                                        feat_type=feat_type, sampling_rate=sampling_rate,
                                                        header=header)

                X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
                                                            feat_type=feat_type, sampling_rate=sampling_rate,
                                                            header=header
                                                            )
                print(
                    f'q_samp: {q_samp}, subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
                pf.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}

                X[q_samp] = np.concatenate([X_normal, X_abnormal], axis=0)
                y[q_samp] = np.concatenate([y_normal, y_abnormal], axis=0)
        else:

            X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim, feat_type=feat_type,
                                                    header=header)
            X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
                                                        feat_type=feat_type, header=header)
            print(f'subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')

            pf.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
            X = np.concatenate([X_normal, X_abnormal], axis=0)
            y = np.concatenate([y_normal, y_abnormal], axis=0)
        pf.Xy_file = os.path.join(pf.out_dir, 'Xy-normal-abnormal.dat')
        dump_data((X, y), out_file=pf.Xy_file)
        print(f'Xy_file: {pf.Xy_file}')
        out_file = pf.Xy_file

    elif data_name in [
        'DWSHR_WSHR_2020'
    ]:
        if data_name == 'DWSHR_WSHR_2020':
            subdatasets = ('UCHI/IOT_2020/dwshr_192.168.143.76', 'UCHI/IOT_2020/wshr_192.168.143.100')
        pf = PCAP2FEATURES(out_dir=os.path.dirname(out_file), feat_type=feat_type, header=header,
                           random_state=random_state, overwrite=overwrite)
        # 1) get_files
        normal_files, abnormal_files = pf.get_path(subdatasets, original_dir, in_dir, direction,
                                                   )  # pcap to xxx_flows_labels.dat.dat
        # 2) full_flows
        (normal_flows, normal_labels), load_time = time_func(load_data, normal_files[0])
        (abnormal_flows, abnormal_labels), load_time = time_func(load_data, abnormal_files[0])
        (abnormal_flows2, abnormal_labels2), load_time = time_func(load_data, abnormal_files[1])
        abnormal_flows.extend(abnormal_flows2)
        abnormal_labels.extend(abnormal_labels2)
        # 3) get the interval from normal flows, and use it to split flows
        durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
        q_interval = 0.9
        interval = _get_split_interval(durations, q_interval)
        print(f'interval {interval} when q_interval: {q_interval}')
        # 4) get subflows
        normal_flows = flows2subflows_SCAM(normal_flows, interval=interval, num_pkt_thresh=2, data_name=data_name,
                                           abnormal=False)
        normal_labels = [normal_labels[0]] * len(normal_flows)
        abnormal_flows = flows2subflows_SCAM(abnormal_flows, interval=interval, num_pkt_thresh=2, data_name=data_name,
                                             abnormal=True)
        abnormal_labels = [abnormal_labels[0]] * len(abnormal_flows)
        # 5). subflows2features
        num_pkts = [len(pkts) for fid, pkts in normal_flows]  # only on normal flows
        dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
        print(f'dim={dim}')
        if feat_type.upper().startswith('SAMP_'):
            X = {}
            y = {}
            flow_durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
            for q_samp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                # get sampling_rate on normal_flows first
                # print(f'np.quantile(flows_durations): {np.quantile(flow_durations, q=[0.1, 0.2, 0.3, 0.9, 0.95])}')
                sampling_rate = _get_split_interval(flow_durations, q_interval=q_samp)
                if sampling_rate <= 0.0: continue
                print(f'sampling_rate: {sampling_rate}, q = {q_samp}')
                X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
                                                        feat_type=feat_type, sampling_rate=sampling_rate,
                                                        header=header)

                X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
                                                            feat_type=feat_type, sampling_rate=sampling_rate,
                                                            header=header
                                                            )
                print(
                    f'q_samp: {q_samp}, subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
                pf.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}

                X[q_samp] = np.concatenate([X_normal, X_abnormal], axis=0)
                y[q_samp] = np.concatenate([y_normal, y_abnormal], axis=0)
        else:

            X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim, feat_type=feat_type,
                                                    header=header)
            X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
                                                        feat_type=feat_type, header=header)
            print(f'subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')

            pf.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
            X = np.concatenate([X_normal, X_abnormal], axis=0)
            y = np.concatenate([y_normal, y_abnormal], axis=0)
        pf.Xy_file = os.path.join(pf.out_dir, 'Xy-normal-abnormal.dat')
        dump_data((X, y), out_file=pf.Xy_file)
        print(f'Xy_file: {pf.Xy_file}')
        out_file = pf.Xy_file
    else:
        msg = data_name
        raise NotImplementedError(msg)

    return out_file


class PCAP2FEATURES():

    def __init__(self, out_dir='', feat_type='IAT_SIZE', header=False, random_state=100, overwrite=False):
        self.out_dir = out_dir
        self.feat_type = feat_type.upper()
        self.header = header
        self.verbose = 10
        self.random_state = random_state
        self.overwrite = overwrite

        if not os.path.exists(os.path.abspath(self.out_dir)): os.makedirs(self.out_dir)

    def get_path(self, datasets, original_dir, in_dir, direction):
        normal_files = []
        abnormal_files = []
        for _idx, _name in enumerate(datasets):
            normal_file, abnormal_file = _get_path(original_dir, in_dir, data_name=_name, direction=direction,
                                                   overwrite=self.overwrite)
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
        interval = _get_split_interval(durations, q_interval=q_interval)
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
        # dim is for SIZE features
        dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension

        if self.feat_type.startswith('SAMP'):
            X = {}
            y = {}

            flow_durations = [_get_flow_duration(pkts) for fid, pkts in normal_flows]
            for q_samp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                # get sampling_rate on normal_flows first
                # print(f'np.quantile(flows_durations): {np.quantile(flow_durations, q=[0.1, 0.2, 0.3, 0.9, 0.95])}')
                sampling_rate = _get_split_interval(flow_durations, q_interval=q_samp)
                if sampling_rate <= 0.0: continue
                X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
                                                        feat_type=self.feat_type, sampling_rate=sampling_rate,
                                                        header=self.header, verbose=self.verbose)

                X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
                                                            feat_type=self.feat_type, sampling_rate=sampling_rate,
                                                            header=self.header,
                                                            verbose=self.verbose)
                print(
                    f'q_samp: {q_samp}, subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
                self.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}

                X[q_samp] = np.concatenate([X_normal, X_abnormal], axis=0)
                y[q_samp] = np.concatenate([y_normal, y_abnormal], axis=0)

        else:
            X_normal, y_normal = _subflows2featutes(normal_flows, labels=normal_labels, dim=dim,
                                                    feat_type=self.feat_type, header=self.header, verbose=self.verbose)
            X_abnormal, y_abnormal = _subflows2featutes(abnormal_flows, labels=abnormal_labels, dim=dim,
                                                        feat_type=self.feat_type, header=self.header,
                                                        verbose=self.verbose)
            print(f'subfeatures: normal_labels: {Counter(normal_labels)}, abnormal_labels: {Counter(abnormal_labels)}')
            self.data = {'normal': (X_normal, y_normal), 'abnormal': (X_abnormal, y_abnormal)}
            X = np.concatenate([X_normal, X_abnormal], axis=0)
            y = np.concatenate([y_normal, y_abnormal], axis=0)
        self.Xy_file = os.path.join(self.out_dir, 'Xy-normal-abnormal.dat')
        dump_data((X, y), out_file=self.Xy_file)
        print(f'Xy_file: {self.Xy_file}')

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

#
# def split_train_arrival_test(X, y, params):
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
#
#     def random_select(X, y, n=100, random_state=100):
#         X, y = shuffle(X, y, random_state=random_state)
#         X0 = X[:n, :]
#         y0 = y[:n]
#
#         rest_X = X[n:, :]
#         rest_y = y[n:]
#         # X_nm_1, y_nm_1 = sklearn.utils.resample(X, y, n_samples=n, replace=False,
#         #                                         random_state=random_state)
#         # if n <=0:
#         #     _, dim = X.shape
#         #     X0, rest_X, y0, rest_y = np.empty((0, dim)), X, np.empty((0,)), y
#         # else:
#         #     X0, rest_X, y0, rest_y = train_test_split(X, y, train_size=n, random_state=random_state, shuffle=True)
#         return X0, y0, rest_X, rest_y
#
#     random_state = params.random_state
#     verbose = params.verbose
#     # Step 1. Shuffle data first
#     X, y = shuffle(X, y, random_state=random_state)
#     if verbose >= DEBUG: data_info(X, name='X')
#
#     n_init_train = params.n_init_train  # 5000
#     n_init_test_abnm_0 = 1  # 50
#     n_arrival = params.n_init_train  # params.n_init_train # 5000
#     n_test_abnm_0 = 100
#
#     idx_nm_0 = y == 'normal_0'
#     X_nm_0, y_nm_0 = X[idx_nm_0], y[idx_nm_0]
#     idx_abnm_0 = y == 'abnormal_0'
#     X_abnm_0, y_abnm_0 = X[idx_abnm_0], y[idx_abnm_0]
#
#     if params.data_type == 'one_dataset':
#         X_nm_1, y_nm_1 = X_nm_0, y_nm_0
#         X_abnm_1, y_abnm_1 = X_abnm_0, y_abnm_0
#     elif params.data_type == 'two_datasets':
#         idx_nm_1 = y == 'normal_1'
#         X_nm_1, y_nm_1 = X[idx_nm_1], y[idx_nm_1]
#         idx_abnm_1 = y == 'abnormal_1'
#         X_abnm_1, y_abnm_1 = X[idx_abnm_1], y[idx_abnm_1]
#
#         if len(y_abnm_1) == 0:
#             # split X_abnm_0 into X_abnm_0 and X_abnm_1
#             X_abnm_0, y_abnm_0, X_abnm_1, y_abnm_1 = random_select(X_abnm_0, y_abnm_0,
#                                                                    n=int(len(y_abnm_0) // 2), random_state=random_state)
#
#     else:
#         raise NotImplementedError()
#
#     N1 = int(round(params.percent_first_init * n_init_train)) + n_init_test_abnm_0 + n_test_abnm_0 + \
#          int(round((1 - params.percent_first_init) * n_arrival))
#     N2 = int(round((1 - params.percent_first_init) * n_init_train)) + n_init_test_abnm_0 + n_test_abnm_0 + int(
#         round(params.percent_first_init * n_arrival))
#
#     AN1 = n_init_test_abnm_0 + n_test_abnm_0
#     is_resample = True
#     if is_resample:
#         print(
#             f'before reampling, y_nm_0: {Counter(y_nm_0)}, y_abnm_0: {Counter(y_abnm_0)}, y_nm_1: {Counter(y_nm_1)}, y_abnm_1: {Counter(y_abnm_1)}')
#         if len(y_nm_0) < N1:
#             X_nm_0, y_nm_0 = sklearn.utils.resample(X_nm_0, y_nm_0, n_samples=N1, replace=True,
#                                                     random_state=42)
#         if len(y_nm_1) < N2:
#             X_nm_1, y_nm_1 = sklearn.utils.resample(X_nm_1, y_nm_1, n_samples=N2, replace=True,
#                                                     random_state=42)
#         if len(y_abnm_0) < AN1:
#             X_abnm_0, y_abnm_0 = sklearn.utils.resample(X_abnm_0, y_abnm_0, n_samples=AN1, replace=True,
#                                                         random_state=42)
#         if len(y_abnm_1) < AN1:
#             X_abnm_1, y_abnm_1 = sklearn.utils.resample(X_abnm_1, y_abnm_1, n_samples=AN1, replace=True,
#                                                         random_state=42)
#
#         print(
#             f'after reampling, y_nm_0: {Counter(y_nm_0)}, y_abnm_0: {Counter(y_abnm_0)}, y_nm_1: {Counter(y_nm_1)}, y_abnm_1: {Counter(y_abnm_1)}')
#     X_normal = np.concatenate([X_nm_0, X_nm_1], axis=0)
#     X_abnormal = np.concatenate([X_abnm_0, X_abnm_1], axis=0)
#     if verbose >= DEBUG: data_info(X_normal, name='X_normal')
#     if verbose >= DEBUG:   data_info(X_abnormal, name='X_abnormal')
#
#     ########################################################################################################
#     # Step 2.1. Get init_set
#     # 1) get init_train: normal
#     X_init_train_nm_0, y_init_train_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0,
#                                                                          n=int(round(
#                                                                              params.percent_first_init * n_init_train,
#                                                                              0)), random_state=random_state)
#     X_init_train_nm_1, y_init_train_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1,
#                                                                          n=int(round((1 - params.percent_first_init) *
#                                                                                      n_init_train, 0)),
#                                                                          random_state=random_state)
#     X_init_train = np.concatenate([X_init_train_nm_0, X_init_train_nm_1], axis=0)
#     y_init_train = np.concatenate([y_init_train_nm_0, y_init_train_nm_1], axis=0)
#
#     # 2) get init_test: normal + abnormal
#     X_init_test_nm_0, y_init_test_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0, n=n_init_test_abnm_0,
#                                                                        random_state=random_state)
#     X_init_test_nm_1, y_init_test_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1, n=n_init_test_abnm_0,
#                                                                        random_state=random_state)
#     X_init_test_abnm_0, y_init_test_abnm_0, X_abnm_0, y_abnm_0 = random_select(X_abnm_0, y_abnm_0,
#                                                                                n=n_init_test_abnm_0,
#                                                                                random_state=random_state)
#     X_init_test_abnm_1, y_init_test_abnm_1, X_abnm_1, y_abnm_1 = random_select(X_abnm_1, y_abnm_1,
#                                                                                n=n_init_test_abnm_0,
#                                                                                random_state=random_state)
#     X_init_test = np.concatenate([X_init_test_nm_0, X_init_test_nm_1,
#                                   X_init_test_abnm_0, X_init_test_abnm_1,
#                                   ], axis=0)
#     y_init_test = np.concatenate([y_init_test_nm_0, y_init_test_nm_1,
#                                   y_init_test_abnm_0, y_init_test_abnm_1,
#                                   ], axis=0)
#
#     ########################################################################################################
#     # Step 2.2. Get arrival_set: normal
#     X_arrival_nm_0, y_arrival_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0,
#                                                                    n=int(
#                                                                        round(
#                                                                            (1 - params.percent_first_init) * n_arrival,
#                                                                            0)), random_state=random_state)
#     X_arrival_nm_1, y_arrival_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1,
#                                                                    n=int(round(params.percent_first_init *
#                                                                                n_arrival, 0)),
#                                                                    random_state=random_state)
#     X_arrival = np.concatenate([X_arrival_nm_0, X_arrival_nm_1], axis=0)
#     y_arrival = np.concatenate([y_arrival_nm_0, y_arrival_nm_1], axis=0)
#
#     ########################################################################################################
#     # Step 2.3. Get test_set
#     # get test_set: normal + abnormal
#     X_test_nm_0, y_test_nm_0, X_nm_0, y_nm_0 = random_select(X_nm_0, y_nm_0, n=n_test_abnm_0,
#                                                              random_state=random_state)
#     X_test_nm_1, y_test_nm_1, X_nm_1, y_nm_1 = random_select(X_nm_1, y_nm_1, n=n_test_abnm_0,
#                                                              random_state=random_state)
#     X_test_abnm_0, y_test_abnm_0, X_abnm_0, y_abnm_0 = random_select(X_abnm_0, y_abnm_0, n=n_test_abnm_0,
#                                                                      random_state=random_state)
#     X_test_abnm_1, y_test_abnm_1, X_abnm_1, y_abnm_1 = random_select(X_abnm_1, y_abnm_1, n=n_test_abnm_0,
#                                                                      random_state=random_state)
#     X_test = np.concatenate([X_test_nm_0, X_test_nm_1, X_test_abnm_0, X_test_abnm_1], axis=0)
#     y_test = np.concatenate([y_test_nm_0, y_test_nm_1, y_test_abnm_0, y_test_abnm_1], axis=0)
#
#     X_init_train, y_init_train = shuffle(X_init_train, y_init_train, random_state=random_state)
#     X_init_test, y_init_test = shuffle(X_init_test, y_init_test, random_state=random_state)
#     X_arrival, y_arrival = shuffle(X_arrival, y_arrival, random_state=random_state)
#     X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
#
#     mprint(f'X_init_train: {X_init_train.shape}, in which, y_init_train is {Counter(y_init_train)}', verbose, INFO)
#     mprint(f'X_init_test: {X_init_test.shape}, in which, y_init_test is {Counter(y_init_test)}', verbose, INFO)
#     mprint(f'X_arrival: {X_arrival.shape}, in which, y_arrival is {Counter(y_arrival)}', verbose, INFO)
#     mprint(f'X_test: {X_test.shape}, in which, y_test is {Counter(y_test)}', verbose, INFO)
#
#     if verbose >= INFO:
#         data_info(X_init_train, name='X_init_train')
#         data_info(X_init_test, name='X_init_test')
#         data_info(X_arrival, name='X_arrival')
#         data_info(X_test, name='X_test')
#
#     return X_init_train, y_init_train, X_init_test, y_init_test, X_arrival, y_arrival, X_test, y_test
#

def plot_data(X, y, title='Data'):
    plt.figure()
    y_unique = np.unique(y)
    colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        plt.scatter(this_X[:, 0], this_X[:, 1], s=50,
                    c=color[np.newaxis, :],
                    alpha=0.5, edgecolor='k',
                    label=f"Class {this_y} {this_X.shape}")
    plt.legend(loc="best")
    plt.title(title)
    plt.show()


def make_circles(N=5000, r1=1, r2=5, w1=0.8, w2=1 / 3, arms=64):
    """ clusterincluster.m
    function data = clusterincluster(N, r1, r2, w1, w2, arms)
    %% Data is N x 3, where the last column is the label


        if nargin < 1
            N = 1000;
        end
        if nargin < 2
            r1 = 1;
        end
        if nargin < 3
            r2 = 5*r1;
        end
        if nargin < 4
            w1 = 0.8;
        end
        if nargin < 5
            w2 = 1/3;
        end
        if nargin < 6
            arms = 64;
        end

        data = [];

        N1 = floor(N/2);
        N2 = N-N1;

        phi1 = rand(N1,1) * 2 * pi;
        %dist1 = r1 + randint(N1,1,3)/3 * r1 * w1;
        dist1 = r1 + randi([0, 2], [N1 1])/3 * r1 * w1;

        d1 = [dist1 .* cos(phi1) dist1 .* sin(phi1) zeros(N1,1)];
        perarm = round(N2/arms);
        N2 = perarm * arms;
        radperarm = (2*pi)/arms;
        phi2 = ((1:N2) - mod(1:N2, perarm))/perarm * (radperarm);
        phi2 = phi2';
        dist2 = r2 * (1 - w2/2) + r2 * w2 * mod(1:N2, perarm)'/perarm;
        d2 = [dist2 .* cos(phi2) dist2 .* sin(phi2) ones(N2,1)];

        data = [d1;d2];
        %scatter(data(:,1), data(:,2), 20, data(:,3)); axis square;
    end

    """
    N1 = int(np.floor(N / 2))
    N2 = N - N1

    phi1 = np.random.rand(N1, 1) * 2 * np.pi  # return a matrix with shape N1x1, values in [0,1]
    # # % dist1 = r1 + np.random.randint(N1, 1, 3) / 3 * r1 * w1;
    dist1 = r1 + np.random.randint(0, high=2 + 1, size=[N1, 1]) / 3 * r1 * w1

    # d1 = [dist1. * np.cos(phi1) dist1. * np.sin(phi1) zeros(N1, 1)];
    # d1 = [col1, col2, label]
    d1 = np.concatenate([dist1 * np.cos(phi1), dist1 * np.sin(phi1), np.zeros((N1, 1))], axis=1)

    perarm = round(N2 / arms)
    N2 = perarm * arms
    radperarm = (2 * np.pi) / arms
    # phi2 = ((1:N2) - mod(1:N2, perarm)) / perarm * (radperarm)
    vs = np.reshape(range(1, N2 + 1), (N2, 1))
    phi12 = (vs - np.mod(vs, perarm)) / perarm * (radperarm)
    # phi2 = phi2';
    phil2 = phi12
    # dist2 = r2 * (1 - w2 / 2) + r2 * w2 * mod(vs, perarm)'/perarm;
    dist2 = r2 * (1 - w2 / 2) + r2 * w2 * np.mod(vs, perarm) / perarm
    # d2 = [dist2. * cos(phi2) dist2. * sin(phi2) ones(N2, 1)]
    d2 = np.concatenate([dist2 * np.cos(phil2), dist2 * np.sin(phil2), np.ones((N2, 1))], axis=1)
    data = np.concatenate([d1, d2], axis=0)
    # % scatter(data(:, 1), data(:, 2), 20, data(:, 3)); axis square;
    X, y = data[:, :2], data[:, -1]
    y = np.asarray([int(v) for v in y])
    return X, y


def _generate_mimic_data(data_type='', random_state=42, out_file=''):
    if data_type == 'one_dataset':
        X, y = make_blobs(n_samples=[12000, 200, 12000, 200],
                          centers=[(-1, -2), (0, 0), (5, 5), (7.5, 7.5)], cluster_std=[(1, 1), (1, 1), (1, 1), (1, 1)],
                          # cluster_std=[(2, 10), (1,1), (2,3)
                          n_features=2,
                          random_state=random_state)  # generate data from multi-variables normal distribution

        y = np.asarray(y, dtype=str)
        y[y == '0'] = 'normal'
        y[y == '1'] = 'abnormal'
        y[y == '2'] = 'normal'
        y[y == '3'] = 'abnormal'

    elif data_type == 'mimic_GMM1':
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

    elif data_type == 'CinC':

        X, y = make_circles()

        y = np.asarray(y, dtype=str)
        y[y == '0'] = 'normal_0'
        y[y == '1'] = 'abnormal_0'
    else:
        msg = out_file
        raise NotImplementedError(msg)

    # plt.scatter(X[:, 0], X[:, 1])
    plot_data(X, y)
    dump_data((X, y), out_file)

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
        obs_dir = '../../../IoT_feature_sets_comparison_20190822/examples/'
        in_dir = f'{obs_dir}data/data_reprst/pcaps'
        pf = PCAP2FEATURES(out_dir=os.path.dirname(Xy_file), random_state=random_state)
        normal_files, abnormal_files = pf.get_path(subdatasets, in_dir, out_dir='../online/out/')
        pf.flows2features(normal_files, abnormal_files, q_interval=0.9)
        data, Xy_file = pf.data, pf.Xy_file


if __name__ == '__main__':
    # main(random_state=RANDOM_STATE, n_jobs=1, single=False)

    _generate_mimic_data(data_type='two_datasets', random_state=42, out_file='./data/demo.dat')
