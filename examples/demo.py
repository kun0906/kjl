import datetime

import numpy as np
from odet.pparser.parser import PCAP, _get_flow_duration, _get_frame_time
from sklearn.base import BaseEstimator

from kjl.utils.data import dump_data, load_data, data_info


def op_demo():
    a = np.array([1, 2, 3, 4])
    # b = np.array([[1, 2, 3],
    #    [3, 4, 5],
    #    [3, 6, 9]])
    print(a.T, a.T.shape, a, a.shape)
    b = np.array([[1, 2, 3, 5],
                  [3, 4, 5, 5],
                  [3, 6, 9, 5]])
    a1 = a[:, np.newaxis]
    print(f'a1:', a1, )
    print(f'b:', b)
    c = a * b
    print(f'c:', c)
    d = a * b.T
    print(f'd:', d)

import logging
from logging import getLogger

lg = getLogger(name='mylog')
lg.setLevel(level=logging.INFO)

lg.info('afadfasf')
lg.debug('afadfasf')
lg.warning('afadfasf')
lg.error('afadfasf')

class MYCLASSIFIER(BaseEstimator):
    print(type(BaseEstimator))
    a = 10

    def __init__(self, a=1, b=2, c = ''):
        print(type(self))
        self.a = a
        self.b = b
        self.c1 =c

    def fit(self):
        pass


def estimator_demo():
    cf = MYCLASSIFIER()
    cf.fit()


def pcap2time():

    pcap_file = '../../IoT_feature_sets_comparison_20190822/examples/data/data_reprst/pcaps/DEMO_IDS' \
                '/DS-srcIP_192.168.10.5/AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap'
    pp=PCAP(pcap_file)
    pp.pcap2flows()
    out_file = pp.pcap_file + '.dat'
    data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in pp.flows]).reshape(-1, 1), name=f'durations before')
    dump_data(pp.flows, out_file)

    flows = load_data(out_file)
    for fid, pkts in flows:
        print([str(datetime.datetime.fromtimestamp(_get_frame_time(pkt))) for pkt in pkts])
    data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in flows]).reshape(-1, 1), name=f'durations')


if __name__ == '__main__':
    # op_demo()
    # estimator_demo()
    pcap2time()