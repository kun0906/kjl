"""
    get ioT_lab_data info
"""

import os
import os.path as pth
import subprocess
from glob import glob
import numpy as np
from kjl.pparser.parser import _pcap2flows, _get_flow_duration, _get_split_interval, _flows2subflows
from kjl.utils.data import dump_data, data_info


def filter_srcIP(pcap_file, kept_ips, output_file='', verbose=1):
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
    srcIP_str = " or ".join([f'ip.src=={srcIP}' for srcIP in kept_ips])
    cmd = f"tshark -r {pcap_file} -w {output_file} {srcIP_str}"

    if verbose > 0: print(f'{cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'{e}, {result}')
        return -1

    return output_file


def get_flows(in_dir, srcIP):
    flows_arr = []
    for sub_dir in os.listdir(in_dir):
        for pcap_file in glob(pth.join(in_dir, sub_dir, '*.pcap')):
            print(pcap_file)
            flows = _pcap2flows(filter_srcIP(pcap_file, [srcIP], output_file='~tmp.pcap', verbose=0), verbose=10)
            print(f'num_flows: {len(flows)}, num_packets: {[len(pkts) for (fid, pkts) in flows]}')
            flows_arr.extend(flows)

    return flows_arr


def main():
    in_dir = 'data/iotlab_devices'
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

        'fridge': {'normal': ('iotlab_open_shut_fridge_192.168.143.43/idle', '192.168.143.43'),
                   'abnormal': ('iotlab_open_shut_fridge_192.168.143.43/open_shut', '192.168.143.43')},

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
        # normal:
        expand_dir = pth.join(in_dir, datasets[device]['normal'][0])
        srcIP = datasets[device]['normal'][1]
        normal_flows = get_flows(expand_dir, srcIP)

        # abnormal:
        expand_dir = pth.join(in_dir, datasets[device]['abnormal'][0])
        srcIP = datasets[device]['abnormal'][1]
        abnormal_flows = get_flows(expand_dir, srcIP)

        subflow_flg = True  # get subflows
        if subflow_flg:
            q_interval = 0.5
            durations = [_get_flow_duration(flow[1]) for flow in normal_flows]
            interval = _get_split_interval(durations, q_interval=q_interval)
            data_info(np.asarray(durations).reshape(-1, 1), name='durations (normal)')
            print(f'interval: {interval}, q_interval: {q_interval}')
            normal_subflows = _flows2subflows(normal_flows, interval)

            abnormal_durations = [_get_flow_duration(flow[1]) for flow in abnormal_flows]
            data_info(np.asarray(abnormal_durations).reshape(-1, 1), name='durations (abnormal)')
            abnormal_subflows = _flows2subflows(abnormal_flows, interval)
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

    dump_data(results, out_file='result.dat')


if __name__ == '__main__':
    main()
