"""
    get ioT_lab_data info
"""

import os
import os.path as pth
import subprocess
from glob import glob
import numpy as np

from odet.pparser.parser import PCAP, _pcap2flows, _get_flow_duration, _get_split_interval, _flows2subflows
from kjl.utils.data import dump_data, data_info
from kjl.utils.tool import load_data
from kjl.utils.utils import func_running_time

def filter_srcIP(pcap_file, kept_ips, output_file='', verbose=1, direction='forward'):
    """

    Parameters
    ----------
    pcap_file
    kept_ips
    output_file
    verbose

    Returns
    -------

    """
    if direction == 'forward':
        IP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
    elif direction =='backward':
        IP_str = " or ".join([f'ip.dst=={srcIP}' for srcIP in kept_ips])
    else:
        IP_str = " or ".join([f'ip.addr=={srcIP}' for srcIP in kept_ips])
    cmd = f"tshark -r {pcap_file} -w {output_file} {IP_str}"

    if verbose > 0: print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return output_file


def get_flows(in_dir, srcIP, direction='forward'):
    flows_arr = []
    for sub_dir in os.listdir(in_dir):
        for pcap_file in glob(pth.join(in_dir, sub_dir, '*.pcap')):
            print(pcap_file)
            flows = _pcap2flows(filter_srcIP(pcap_file, [srcIP], output_file='~tmp.pcap', verbose=1, direction =direction), verbose=10)
            print(f'num_flows: {len(flows)}, num_packets: {[len(pkts) for (fid, pkts) in flows]}')
            flows_arr.extend(flows)

    return flows_arr


class FEATURES(PCAP):

    def __init__(self, feat_type = 'IAT_SIZE', fft=False, header=False, q_interval = 0.9, verbose = 10):
        self.feat_type = feat_type
        self.fft = fft
        self.header = header
        self.q_interval = q_interval
        self.verbose = verbose

    def flow2feats(self, flows, dim = None):
        self.flows = flows
        # self.dim = dim
        self.flow2features(feat_type=self.feat_type, fft=self.fft, header = self.header, dim=dim)


def main():
    in_dir = 'data/iotlab_devices'
    out_dir = 'data/feats/iotlab_devices'
    #
    # sub_dir_srcIPs = [
    #
    #     # normal traffic for each device
    #     ('iotlab_idle', '192.168.143.76'),  # dishwasher
    #     ('iotlab_idle', '192.168.143.99'),  # dryer
    #     ('iotlab_idle', '192.168.143.43'),  # fridge: faucet, lower, shut
    #     ('iotlab_idle', '192.168.143.100'),  # washer
    #
    #     # abnormal traffic for each device
    #     ('iotlab_open_dishwasher', '192.168.143.76'),
    #     ('iotlab_open_dryer', '192.168.143.99'),
    #     ('iotlab_open_faucet_fridge', '192.168.143.43'),
    #     ('iotlab_open_fridge', '192.168.143.43'),
    #     ('iotlab_open_lower_fridge', '192.168.143.43'),
    #     ('iotlab_open_shut_fridge', '192.168.143.43'),
    #     ('iotlab_open_washer', '192.168.143.100'),
    #
    # ]

    datasets = {
        #             'dishwasher': {'normal': ('iotlab_idle', '192.168.143.76'),
        #                            'abnormal': ('iotlab_open_dishwasher', '192.168.143.76')},
        #             'dryer': {'normal':  ('iotlab_idle', '192.168.143.99'),
        #                            'abnormal':  ('iotlab_open_dryer', '192.168.143.99'),},
        #
        # 'fridge': {'normal': ('iotlab_open_shut_fridge_192.168.143.43/idle', '192.168.143.43'),
        #            'abnormal': ('iotlab_open_shut_fridge_192.168.143.43/open_shut', '192.168.143.43')},

        # browse
        'fridge': {'normal': ('iotlab_open_shut_fridge_192.168.143.43/idle', '192.168.143.43'),
                   'abnormal': ('iotlab_browse_app_fridge/', '192.168.143.43')},

        # 'faucet_fridge': {'normal': ('iotlab_idle', '192.168.143.43'),
        #            'abnormal':   ('iotlab_open_faucet_fridge', '192.168.143.43'),},
        #
        # 'lower_fridge': {'normal': ('iotlab_idle', '192.168.143.43'),
        #            'abnormal':   ('iotlab_open_lower_fridge', '192.168.143.43')},
        #
        # 'shut_fridge': {'normal': ('iotlab_idle', '192.168.143.43'),
        #            'abnormal':    ('iotlab_open_shut_fridge', '192.168.143.43')},
        #
        # 'washer': {'normal': ('iotlab_idle', '192.168.143.100'),
        #                'abnormal':  ('iotlab_open_washer', '192.168.143.100')},
    }

    results = {}
    for device in datasets.keys():
        expand_dir = pth.join(out_dir, device)
        normal_subflows_file = os.path.join(expand_dir, 'normal_subflows.dat')
        abnormal_subflows_file = os.path.join(expand_dir, 'abnormal_subflows.dat')

        overwrite =  True
        direction = 'backward' # 'forward': only upload stream stats; 'backward', only download stream stats; 'both'
        print(f'direction: {direction}')
        if overwrite:
            if os.path.exists(normal_subflows_file): os.remove(normal_subflows_file)

        if not os.path.exists(normal_subflows_file) or not os.path.exists(abnormal_subflows_file):
            # normal:
            expand_dir = pth.join(in_dir, datasets[device]['normal'][0])
            srcIP = datasets[device]['normal'][1]
            normal_flows = get_flows(expand_dir, srcIP, direction=direction)

            # abnormal:
            expand_dir = pth.join(in_dir, datasets[device]['abnormal'][0])
            srcIP = datasets[device]['abnormal'][1]
            abnormal_flows = get_flows(expand_dir, srcIP, direction=direction)

            expand_dir =os.path.join(out_dir, device)
            if not os.path.exists(expand_dir):
                os.makedirs(expand_dir)
            out_file = os.path.join(expand_dir,'raw_normal_abnormal_flows.dat')
            dump_data((normal_flows, abnormal_flows), out_file)

            subflow_flg = True  # get subflows
            if subflow_flg:
                q_interval = 0.9
                durations = [_get_flow_duration(flow[1]) for flow in normal_flows]
                interval = _get_split_interval(durations, q_interval=q_interval)
                data_info(np.asarray(durations).reshape(-1, 1), name='durations (normal)')
                print(f'interval: {interval}, q_interval: {q_interval}')
                normal_subflows, _ = _flows2subflows(normal_flows, interval, labels= ['normal']* len(normal_flows))
                dump_data(normal_subflows, normal_subflows_file)

                # abnormal_durations = [_get_flow_duration(flow[1]) for flow in abnormal_flows]
                # data_info(np.asarray(abnormal_durations).reshape(-1, 1), name='durations (abnormal)')
                abnormal_subflows, _ = _flows2subflows(abnormal_flows, interval,  labels = ['abnormal']* len(abnormal_flows))   # here the interval equals normal interval
                dump_data(abnormal_subflows, abnormal_subflows_file)

            results[device] = {'normal': (normal_flows, normal_subflows),
                               'abnormal': (abnormal_flows, abnormal_subflows),
                               'q_interval': q_interval, 'interval': interval}

            # print results
            for key, value in results.items():
                print(f'\n***{key}')
                flows = value['normal']
                print(f'{key}, num_flows (normal): {len(flows[0])}')
                print(f'num_packets: {[len(pkts) for (fid, pkts) in flows[0]]}')
                print(f'q_interval: {q_interval}, interval: {interval}')
                print(f'{key}, num_sub_flows (normal): {len(flows[1])}')
                print(f'num_packets (sub_flows): {[len(pkts) for (fid, pkts) in flows[1]]}')

                flows = value['abnormal']
                print(f'{key}, num_flows (abnormal): {len(flows[0])}')
                print(f'num_packets: {[len(pkts) for (fid, pkts) in flows[0]]}')
                print(f'{key}, num_sub_flows (abnormal): {len(flows[1])}')
                print(f'num_packets (sub_flows): {[len(pkts) for (fid, pkts) in flows[1]]}')

            dump_data(results, out_file=f'{out_dir}/result.dat')

        ##########################################################################################
        # subflows to features
        print('load data...')
        normal_subflows, load_normal_time = func_running_time(load_data, normal_subflows_file)
        print(f'finish loading normal data and it takes {load_normal_time} seconds')
        abnormal_subflows, load_abnormal_time = func_running_time(load_data, abnormal_subflows_file)
        print(f'finish loading abnormal data and it takes {load_abnormal_time} seconds')

        feat_type = 'IAT_SIZE'
        ft = FEATURES(feat_type, fft=False, header=False)

        tmp_feat_lst = []
        # for i, (type, subflows) in enumerate([('normal', normal_subflows), ('abnormal', abnormal_subflows)]):
        #     if i == 0:
        #         dim = None
        #     if i == 1:
        #         if feat_type == 'IAT_SIZE': dim = (ft.dim + 1) // 2
        #     data_info(np.asarray([len(pkts) for fid, pkts in subflows])[:, np.newaxis], name=f'packets of {type} flows')
        #     ft.flow2feats(subflows, dim)
        #     features = ft.features
        #     print(f'{type}: feature.shape: {features.shape}')
        #     out_file = os.path.join(expand_dir, f'{type}_{feat_type}.dat')
        #     print(out_file)
        #     dump_data(features, out_file)

        combined_flows_file = normal_subflows_file
        combined_subflows = [(normal_subflows, abnormal_subflows)]
        combined_normal_subflows = normal_subflows
        num_pkts = [len(pkts) for fid, pkts in combined_normal_subflows]
        dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
        print(f'{feat_type}, dim: {dim}, q_interval: {q_interval}')
        if feat_type == 'IAT_SIZE': iat_size_dim = 2 * dim - 1
        combined_features_file = os.path.join(os.path.dirname(combined_flows_file),
                                              f'combined_feats-q_{q_interval}-{feat_type}-dim_{iat_size_dim}.dat')

        if overwrite:
            if os.path.exists(combined_features_file): os.remove(combined_features_file)
        if not os.path.exists(combined_features_file):
            combined_features = []
            for (normal_flows, abnormal_flows) in combined_subflows:
                # data_info(np.asarray([len(pkts) for fid, pkts in normal_flows])[:, np.newaxis],
                #           name=f'packets of {type} flows')
                ft.flow2feats(normal_flows, dim)
                normal_feats = ft.features
                ft.flow2feats(abnormal_flows, dim)
                abnormal_feats = ft.features
                print(f'normal.shape: {normal_feats.shape}, abnormal.shape: {abnormal_feats.shape}')

                combined_features.append((normal_feats, abnormal_feats))

            print(combined_features_file)
            dump_data(combined_subflows, combined_features_file)

        else:
            print('load data')
            combined_feats, load_time = func_running_time(load_data, combined_features_file)
            print(f'load {combined_features_file} takes {load_time} s.')

#


if __name__ == '__main__':
    main()
