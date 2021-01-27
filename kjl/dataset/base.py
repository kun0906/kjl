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
        if pcap_file == out_file:
            print(f'in_file == out_file: {out_file}')
            return out_file
        if os.path.exists(out_file):
            print(f'out_file exists: {out_file}')
            return out_file
        if direction == 'src':
            ip_str = " or ".join([f'ip.src=={ip}' for ip in ips])
        elif direction == 'dst':
            ip_str = " or ".join([f'ip.dst=={ip}' for ip in ips])
        else:
            ip_str = " or ".join([f'ip.addr=={ip}' for ip in ips])
        cmd = f"tshark -r {pcap_file} -w {out_file} {ip_str}"

        if verbose >= 10: print(f'cmd: {cmd}')
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



def process_CIC_IDS_2017(label_file, time_range=['start', 'end'], output_file='_reduced.txt'):
    """ timezone: ADT in CICIDS_2017 label.csv

    Parameters
    ----------
    label_file
    time_range
    output_file

    Returns
    -------

    """
    with open(output_file, 'w') as out_f:
        start = 0
        i = 0
        start_flg = True
        end = 0
        max_sec = -1
        min_sec = -1
        with open(label_file, 'r') as in_f:
            line = in_f.readline()
            flg = False
            while line:
                if line.startswith("Flow"):
                    line = in_f.readline()
                    continue
                arr = line.split(',')
                # time
                # print(arr[6])
                time_str = datetime.strptime(arr[6], "%d/%m/%Y %H:%M")
                time_str = convert_datetime_timezone(str(time_str), tz1='Canada/Atlantic', tz2='UTC')
                ts = time_string_to_seconds(str(time_str), '%Y-%m-%d %H:%M:%S')
                if start_flg:
                    print(i, ts, start)
                    start = ts
                    min_sec = start
                    start_flg = False
                else:
                    if ts > end:
                        end = ts
                    if ts < min_sec:
                        min_sec = ts
                    if ts > max_sec:
                        max_sec = ts
                if ts > time_range[0] and ts < time_range[1]:
                    out_f.write(line.strip('\n') + '\n')
                # if ts > time_range[1]:
                #     break

                line = in_f.readline()
                i += 1
        print(start, end, time_range, i, min_sec, max_sec)

    return output_file
