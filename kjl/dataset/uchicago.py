"""
    Process UChicago IoT data
"""
import os.path as pth
import subprocess
from datetime import datetime
from glob import glob

from odet.pparser.parser import _get_frame_time, PcapReader

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


def get_flows(in_dir, subdatasets, out_dir='examples/data/feats', verbose=15, overwrite=False):
    normal_files = []  # store normal flows
    abnormal_files = []  # store abnormal flows

    for i, subdataset in enumerate(subdatasets):
        if subdataset is None: continue
        normal_dir, abnormal_dir = subdataset
        #########################################################################################################
        # get normal flows
        out_file = os.path.join(out_dir, normal_dir, 'flows.dat')
        if overwrite:
            if os.path.exists(out_file): os.remove(out_file)
        if not os.path.exists(out_file):
            uc = UChicago()
            # normal_dir = 'DS60_UChi_IoT/iotlab_open_shut_fridge_192.168.143.43/idle'
            pcap_files = uc.get_path(os.path.join(in_dir, normal_dir))
            labels = [f'normal' for v in range(len(pcap_files))]
            uc.get_flows(pcap_files, labels)

            dump_data((uc.flows, uc.labels), out_file)
        normal_files.append(out_file)

        #########################################################################################################
        # get abnormal flows
        if not (abnormal_dir is None):
            out_file = os.path.join(out_dir, abnormal_dir, 'flows.dat')
            if overwrite:
                if os.path.exists(out_file): os.remove(out_file)
            if not os.path.exists(out_file):
                uc = UChicago()
                # abnormal_dir = 'DS60_UChi_IoT/iotlab_open_shut_fridge_192.168.143.43/open_shut'
                pcap_files = uc.get_path(os.path.join(in_dir, abnormal_dir))
                labels = [f'abnormal' for v in range(len(pcap_files))]
                uc.get_flows(pcap_files, labels)
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
    print(f'pcap_files: {pcap_files}')
    for pcap_file in pcap_files:
        # https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case
        # out_file = pth.join(out_dir, pcap.split(sep=in_dir)[-1]) # second_dir cannot start with '/'

        # old_pcap = pcap_file
        # pcap_file = pcap_file.replace('.pcap_filtered', '')
        # print(old_pcap, os.path.exists(old_pcap))
        # # os.rename(old_pcap, pcap_file)    # it will overwrite the old file.
        # if '_filtered' not in pcap_file:
        out_file = out_dir + pcap_file.split(sep=in_dir)[-1] + '_filtered.pcap'
        uc.filter_ip(pcap_file, out_file, ips=ips, direction=direction, keep_original=keep_original)

    print('finished!')


def get_pcap_time(pcap_file):
    for i, pkt in enumerate(PcapReader(pcap_file)):
        if i == 0:
            start = _get_frame_time(pkt)

    end = _get_frame_time(pkt)
    return start, end


def float2datetime(fl):
    v = datetime.datetime.fromtimestamp(fl)
    return str(v)


def _extract_abnormal_pkts(pcap_file, out_dir=None, verbose=20, keep_original=True):
    ' editcap -A "2017-07-04 09:02:00" -B "2017-07-04 09:05:00" AGMT-WorkingHours-WorkingHours.pcap AGMT-WorkingHours-WorkingHours-10000pkts.pcap'
    """
         f'online/data/deeplens_open_shut_fridge_batch_8.pcap_filtered.pcap':
         In each 15 seconds, open the fridge once 
         
    """
    if out_dir is None:
        out_dir = os.path.dirname(pcap_file) + '/filtered'

    if not pth.exists(out_dir):
        os.makedirs(out_dir)

    start, end = get_pcap_time(pcap_file)
    print(f'start: {start}, end: {end}')
    for i in range(int((end - start) // 15) + 5):
        if i == 0:
            start = start + 15  # 15 is offset of time
            continue

        if start > end:
            print(f'num of split_pcaps: {i}')
            break
        out_file = os.path.join(out_dir, f'start={start}.pcap')
        # keep the open activity in previous 5s and after 5s
        cmd = f"editcap -A \"{float2datetime(start - 2)}\" -B \"{float2datetime(start + 9)}\" {pcap_file} {out_file}"
        if verbose > 10: print(f'{cmd}')
        result = ''
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
            # if not keep_original:
            #     os.remove(pcap_file)
        except Exception as e:
            print(f'{e}, {result}')
            return -1

        start = start + 15

    return 0


def extract_abnormal_pkts(in_dir='', out_dir=''):
    uc = UChicago()
    pcap_files = uc.get_path(in_dir)
    for pcap_file in pcap_files:
        _extract_abnormal_pkts(pcap_file, out_dir=out_dir)


############################################

# split iotlab dataset
import os
#
# # Relative path, no need to change
# script_path = '../script/'
#
# # Absolute path, change it according to data locations
# data_path = '../../data/'
# feature_path = '../../features/'
import pandas as pd
import pickle


def save_header(device, data_path):
    freq_st = []
    acc_st = []

    sequences = ['capture' + str(i) + '.seq'
                 for i in range(len(glob.glob(data_path + device + '/capture*')))]

    for i in range(len(sequences)):
        folder = data_path + device + '/capture' + str(i) + '.seq/'
        f = open(glob.glob(folder + '*.head')[0], 'rb')
        header = pickle.load(f)
        freq_st.append(header['frequency_start'])
        acc_st.append(header['access_history_start'])

    timestamps = [sequences, freq_st, acc_st]

    df = pd.DataFrame(timestamps).transpose()
    df.columns = ['Sequences', 'Frequency Start', 'Access History Start']

    df.to_csv(data_path + device + '/deeplens_' + device.split('iotlab_')[-1] + '_timestamps.csv')
    return df


def time2datetime(data):
    datetime_li = []

    for time in data['frame.time'].tolist():
        date_time_str = time.split('000 CDT')[0]
        date_time_obj = datetime.strptime(date_time_str, '%b %d, %Y  %H:%M:%S.%f')
        datetime_li.append(date_time_obj)

    return (datetime_li)


def cut_pcaps(df_ts, device, data_path):
    status = device.split('iotlab_')[-1]
    for i in range(len(df_ts['Frequency Start'].tolist())):
        start_time = df_ts['Frequency Start'][i].strftime("%Y-%m-%d %H:%M:%S")
        if i == len(df_ts['Frequency Start'].tolist()) - 1:
            end_time = (df_ts['Frequency Start'][i]
                        + datetime.timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            end_time = (df_ts['Frequency Start'][i + 1]
                        + datetime.timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        cmd = ('editcap -A "' + start_time + '" -B "'
               + end_time + '" ' + data_path + device + '/deeplens_'
               + status + '.pcap ' + data_path + device + '/capture' + str(i) + '.seq/deeplens_'
               + status + '_' + str(i) + '.pcap')
        os.system(cmd)


def split_by_activity(pcap_file, out_dir='aecho_192.168.143.74/', activity='shop', interval=30):
    # # activities = ['iotlab_open_shut_fridge','iotlab_open_faucet_fridge','iotlab_open_lower_fridge']
    # activities = ['iotlab_browse_app_fridge']
    # activity = activities[0]
    # data_path = ''
    # df_act_ts = save_header(activity, data_path)
    # cut_pcaps(df_act_ts, activity, data_path)

    # status = device.split('iotlab_')[-1]
    # start_time = ''
    start_time, end_time_pcap = get_pcap_time(pcap_file)
    print(f'{pcap_file}, start: {start_time}, end: {end_time_pcap}')
    i = 0   # 2020-12-01 14:49:52.751311
    while start_time < end_time_pcap:
        end_time = start_time + float(interval)
        out_file = out_dir + f'/{activity}/capture' + str(i) + '.seq/deeplens_' + activity + '_' + str(i) + '.pcap'
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        cmd = ('editcap -A "' + datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S") + '" -B "'
               + datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S") + '" ' + f"{pcap_file} {out_file}")
        print(i, cmd)
        os.system(cmd)

        i += 1
        start_time = end_time

    return i


if __name__ == '__main__':
    # # ''
    # in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    # out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    # filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')
    #
    # in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    # out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    # filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')
    #
    # in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    # out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    # filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')
    #
    # in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    # out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    # filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')
    #
    # in_dir = f'../../Datasets/UCHI/IOT_2020/iotlab_browse_app_fridge-20200906T225950Z-001'
    # out_dir = f'../../Datasets/UCHI/IOT_2020/sfrig_192.168.143.43/browse'
    # filter_ips(in_dir, out_dir, ips=['192.168.143.43'], direction='both')

    pcap_file = f'online/data/deeplens_open_shut_fridge_batch_8.pcap_filtered.pcap'
    _extract_abnormal_pkts(pcap_file, out_dir=None)
