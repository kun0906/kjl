"""

"""
import pickle
from datetime import datetime

def load_video_head(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)
    print(data)
    print(datetime.fromtimestamp(float(12960)))
    print(datetime.fromtimestamp(float(17279)))


def flow_check(in_file):
    print(in_file)


if __name__ == '__main__':
    load_video_head(in_file='speedup/data/fezggxizdz.head')
    flow_check(in_file='online/data/deeplens_open_shut_fridge_batch_8.pcap_filtered.pcap')
