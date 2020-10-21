"""
    Base class for preprocessing pcap
"""
import os
import os.path as pth
import subprocess
from collections import Counter
from glob import glob

from odet.pparser.parser import PCAP

from kjl.utils.data import dump_data


class Base:

    def __init__(self, verbose=15):

        self.verbose = verbose

    def filter_ip(self, pcap_file, out_file, ips=[], direction='both', keep_original = True, verbose=10):
        if not pth.exists(pth.dirname(out_file)):
            os.makedirs(pth.dirname(out_file))

        if direction == 'src':
            ip_str = " or ".join([f'ip.src=={ip}' for ip in ips])
        elif direction == 'dst':
            ip_str = " or ".join([f'ip.dst=={ip}' for ip in ips])
        else:
            ip_str = " or ".join([f'ip.addr=={ip}' for ip in ips])
        cmd = f"tshark -r {pcap_file} -w {out_file} {ip_str}"

        if verbose > 10: print(f'{cmd}')
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
            if not keep_original:
                os.remove(pcap_file)
        except Exception as e:
            print(f'{e}, {result}')
            return -1

        return out_file

    def get_path(self, in_dir):
        pass

    def pcap2flows(self, pcap_file, label_file=None, label=None):
        pp = PCAP(pcap_file=pcap_file)
        pp.pcap2flows()
        # pp.label_flows(label_file, label=label)
        flows = pp.flows
        # labels = pp.labels
        # if self.verbose > 10: print(f'labels: {Counter(labels)}')
        labels = [label]*len(flows)

        return flows, labels

    def get_flows(self, pcap_files=''):
        flows = []
        for pcap in pcap_files:
            _flows, _labels = self.pcap2flows(pcap)
            flows.extend(_flows)

        return flows
