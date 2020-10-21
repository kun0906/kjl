"""
    Process UChicago IoT data
"""
import os
import os.path as pth
import subprocess
from collections import Counter
from glob import glob

from odet.pparser.parser import PCAP

from kjl.dataset.base import Base
from kjl.utils.data import dump_data


class UChicago(Base):

    def __init__(self, verbose=15):
        self.verbose = verbose

    def get_path(self, in_dir):
        pcap_files = []
        for sub_dir in os.listdir(in_dir):
            for pcap_file in glob(pth.join(in_dir, sub_dir, '*.pcap')):
                pcap_files.append(pcap_file)
        # pcap_files = [os.path.join(dir_pth, v) for v in os.listdir(dir_pth)]
        pcap_files = sorted(pcap_files, reverse=False)
        if self.verbose > 10: print(pcap_files)
        return pcap_files

    def get_flows(self, pcap_files='', labels=[]):
        self.flows = []
        self.labels = []
        for pcap, label in zip(pcap_files, labels):
            print(f'pcap: {pcap}, label: {label}')
            _flows, _labels = self.pcap2flows(pcap, label=label)
            self.flows.extend(_flows)
            self.labels.extend(_labels)


def get_flows(in_dir, subdatasets, out_dir='examples/data/feats', verbose = 15):
    normal_files = []  # store normal flows
    abnormal_files = []  # store abnormal flows

    for i, subdataset in enumerate(subdatasets):
        normal_dir, abnormal_dir = subdataset
        #########################################################################################################
        # get normal flows
        uc = UChicago()
        # normal_dir = 'DS60_UChi_IoT/iotlab_open_shut_fridge_192.168.143.43/idle'
        pcap_files = uc.get_path(os.path.join(in_dir, normal_dir))
        labels = [f'normal' for v in range(len(pcap_files))]
        uc.get_flows(pcap_files, labels)
        out_file = os.path.join(out_dir, normal_dir, 'flows.dat')
        dump_data((uc.flows, uc.labels), out_file)
        normal_files.append(out_file)

        #########################################################################################################
        # get abnormal flows
        uc = UChicago()
        # abnormal_dir = 'DS60_UChi_IoT/iotlab_open_shut_fridge_192.168.143.43/open_shut'
        pcap_files = uc.get_path(os.path.join(in_dir, abnormal_dir))
        labels = [f'abnormal' for v in range(len(pcap_files))]
        uc.get_flows(pcap_files, labels)
        out_file = os.path.join(out_dir, abnormal_dir, 'flows.dat')
        dump_data((uc.flows, uc.labels), out_file)
        abnormal_files.append(out_file)

    if verbose > 10:
        print(f'normal_files: {normal_files}')
        print(f'abnormal_files: {abnormal_files}')
    return normal_files, abnormal_files


def filter_ips(in_dir='', out_dir='', ips=[], direction='both', keep_original=True):

    #########################################################################################################
    uc = UChicago()
    pcap_files = uc.get_path(in_dir)
    for pcap in pcap_files:
        # https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case
        # out_file = pth.join(out_dir, pcap.split(sep=in_dir)[-1]) # second_dir cannot start with '/'
        out_file = out_dir + pcap.split(sep=in_dir)[-1] +  '_filtered.pcap'
        if '_filtered' not in pcap:
            uc.filter_ip(pcap, out_file, ips=ips, direction=direction, keep_original=keep_original)


    print('finished!')


if __name__ == '__main__':
    # ''
    in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')

    in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')

    in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')

    in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')

    in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')