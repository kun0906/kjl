"""

"""
from kjl.utils.tool import execute_time

RANDOM_STATE = 42

import os
import numpy as np
from odet.pparser.parser import PCAP, _get_split_interval, _get_flow_duration, _flows2subflows
from odet.utils.tool import dump_data, load_data


class FEATURES(PCAP):

    def __init__(self, pcap_file='data/demo.pcap', label_file='demo.csv', label=None,
                 feat_type='IAT_SIZE',
                 interval=0, q_interval=0.9,
                 fft=False, header=False, out_dir='.', flow_pkts_thres=2,
                 verbose=10, random_state=100):
        if not os.path.exists(pcap_file): print(f'{pcap_file} does not exist.')

        (super, FEATURES).__init__(pcap_file=pcap_file)

        self.pcap_file = pcap_file
        self.label_file = label_file
        self.label = label
        self.feat_type = feat_type
        self.q_interval = q_interval
        self.interval = interval
        self.fft = fft
        self.header = header
        self.out_dir = out_dir
        self.flow_pkts_thres = flow_pkts_thres
        self.verbose = verbose
        self.random_state = random_state

    def pcap2features(self):
        # extract flows from pcap
        self.pcap2flows()
        # label each flow with a file or label
        self.label_flows(label_file=self.label_file, label=self.label)
        out_file = f'{self.out_dir}/flows.dat'
        print('raw_flows+labels: ', out_file)
        dump_data((self.flows, self.labels), out_file)

        self.flows2subflows(interval=self.interval, q_interval=self.q_interval)
        out_file = f'{self.out_dir}/subflows-q_interval:{self.q_interval}.dat'
        print('subflows+labels: ', out_file)
        dump_data((self.flows, self.labels), out_file)

        # extract features from each flow given feat_type
        self.flow2features(self.feat_type, fft=self.fft, header=self.header)
        out_file = f'{self.out_dir}/features-q_interval:{self.q_interval}.dat'
        print('features+labels: ', out_file)
        dump_data((self.features, self.labels), out_file)

        print(self.features.shape, self.pcap2flows.tot_time, self.flows2subflows.tot_time, self.flow2features.tot_time)

        return self.features

#
# @execute_time
# def get_normal_abnormal_featrues(normal_file=(), abnormal_file=(), feat_type='IAT_SIZE',
#                                  q_interval=0.9, fft=False, header=False, out_dir='.',
#                                  verbose=10, random_state=100):
#     """get features from a single normal and abnormal pcap
#
#     Parameters
#     ----------
#     normal_file: (pcap_file, label_file)
#     abnormal_file
#     q_interval
#
#     Returns
#     -------
#
#     """
#     pcap_file, label_file = normal_file
#     norm_ft = FEATURES(pcap_file=pcap_file, interval=0, q_interval=q_interval, feat_type=feat_type,
#                        label_file=label_file, label='normal', fft=fft, header=header, out_dir=out_dir,
#                        verbose=verbose,
#                        random_state=random_state)
#     norm_ft.pcap2features()
#
#     if 'DS10_UNB_IDS' in pcap_file or 'AGMT-WorkingHours/' in pcap_file:
#         X, y = norm_ft.features, norm_ft.labels
#         # normal_test_idx = np.in1d(range(normal_data.shape[0]), normal_test_idx)  # return boolean idxes
#         bools = [y == 'normal']
#         res = {'norm': (X[bools], y[bools]), 'abnorm': (X[not bools], y[not bools])}
#     else:
#         pcap_file, label_file = abnormal_file
#         abnorm_ft = FEATURES(pcap_file=pcap_file, interval=norm_ft.interval, q_interval=q_interval, feat_type=feat_type,
#                              label_file=label_file, label='abnormal', fft=fft, header=header, out_dir=out_dir,
#                              verbose=verbose,
#                              random_state=random_state)
#         abnorm_ft.pcap2features()
#
#         res = {'norm': (norm_ft.features, norm_ft.labels), 'abnorm': (abnorm_ft.features, abnorm_ft.labels)}
#
#     out_file = f'{out_dir}/Xy-normal-abnormal.dat'
#     print(out_file)
#     dump_data(res, out_file)
#
#     return res, out_file
#
#
#
# def seperate_normal_abnormal(pcap_file, label_file, out_dir, random_state=100):
#
#     ft = FEATURES(pcap_file=pcap_file, interval=0,
#                        label_file=label_file, label=None,
#                        out_dir=out_dir, verbose=10,
#                        random_state=random_state)
#
#     # extract flows from pcap
#     ft.pcap2flows()
#     # label each flow with a file or label
#     ft.label_flows(label_file=label_file, label=None)
#
#     normal_flows=[]
#     normal_labels=[]
#     abnormal_flows = []
#     abnormal_labels = []
#     for (f, l) in zip(ft.flows, ft.labels):
#         if 'normal' in l:
#             normal_flows.append(f)
#             normal_labels.append(l)
#         else:
#             abnormal_flows.append(f)
#             abnormal_labels.append(l)
#
#     # bools = [ft.labels == 'normal']
#     # normal_flows = ft.flows[bools]
#     # normal_labels = ft.labels[bools]
#     # normal_data = (normal_flows, normal_labels)
#     #
#     # abnormal_flows = ft.flows[not bools]
#     # abnormal_labels = ft.labels[not bools]
#     # abnormal_data = (abnormal_flows, abnormal_labels)
#
#     normal_file  = os.path.join(out_dir, 'normal.dat')
#     dump_data((normal_flows, normal_labels), out_file=normal_file)
#     abnormal_file = os.path.join(out_dir, 'abnormal.dat')
#     dump_data((abnormal_flows, abnormal_labels), out_file= abnormal_file)
#
#     normal_data = (normal_flows, normal_labels)
#     abnormal_data = (abnormal_flows, abnormal_labels)
#     return normal_data, abnormal_data, normal_file, abnormal_file
#
#
# def get_flows(pcap_file, label='normal', out_dir='', random_state=100):
#     ft = FEATURES(pcap_file=pcap_file, interval=0,
#                   label_file=None, label=label,
#                   out_dir=out_dir, verbose=10,
#                   random_state=random_state)
#
#     # extract flows from pcap
#     ft.pcap2flows()
#
#     data = (ft.flows, [label]* len(ft.flows))
#
#     data_file = os.path.join(out_dir, f'{label}.dat')
#     dump_data(data, out_file=data_file)
#
#     return data, data_file
#
#
#
#
# @execute_time
# def get_mutli_nomral_abnormal_features(normal_files=[], abnormal_files=[], feat_type='IAT_SIZE',
#                                        q_interval=0.9, fft=False, header=False, out_dir='.',
#                                        verbose=10, random_state=100):
#     """ures from normal and abnormal pcaps
#
#     Parameters
#     ----------
#     normal_files
#     abnormal_files
#     q_interval
#
#     Returns
#     -------
#
#     """
#
#
#     durations=[]
#     normal_flows = []
#     normal_labels = []
#     abnormal_flows= []
#     abnormal_labels = []
#     for i, ((normal_pcap_file, normal_label_file), (abnormal_pcap_file, abnormal_label_file)) in enumerate(zip(normal_files, abnormal_files)):
#         if 'DS10_UNB' in normal_pcap_file or 'AGMT-WorkingHours' in normal_pcap_file:
#             normal_data, abnormal_data, normal_file, abnormal_file = seperate_normal_abnormal(normal_pcap_file, normal_label_file,
#                                                                                               out_dir=out_dir, random_state=random_state)
#         else:
#             normal_data, normal_file = get_flows(pcap_file=normal_pcap_file, label='normal', out_dir=out_dir, random_state=random_state)
#             abnormal_data, abnormal_file = get_flows(pcap_file=abnormal_pcap_file, label='abnormal', out_dir=out_dir,
#                                                  random_state=random_state)
#
#         durations.extend([_get_flow_duration(pkts) for fid, pkts in normal_data[0]])
#         normal_flows.extend(normal_data[0])
#         normal_labels.extend([f'normal_{i}'] * len(normal_data[1]))
#         abnormal_flows.extend(abnormal_data[0])
#         abnormal_labels.extend([f'abnormal_{i}']* len(abnormal_data[1]))
#
#     # get interval from all normal flows
#     interval = _get_split_interval(durations)
#     print(f'interval {interval} when q_interval: {q_interval}')
#
#     normal_feats = []
#
#     num_pkts = [len(pkts) for fid,  pkts in normal_flows]   # only on normal flows
#     dim = int(np.floor(np.quantile(num_pkts, q_interval)))  # use the same q_interval to get the dimension
#
#     flows = []
#     labels = []
#     flows.extend(normal_flows)
#     labels.extend(normal_labels)
#     flows.extend(abnormal_flows)
#     labels.extend(abnormal_labels)
#     subflows, sublabels = _flows2subflows(flows, interval=interval,  labels=labels, flow_pkts_thres=2, verbose=1)
#     out_file = f'{out_dir}/subflows-q_interval:{q_interval}.dat'
#     print('subflows+labels: ', out_file)
#     dump_data((subflows, sublabels), out_file)
#
#     # extract features from each flow given feat_type
#     pp = PCAP()
#     pp.flows = subflows
#     pp.labels = sublabels
#     pp.flow2features(feat_type, fft=False, header=False, dim=dim)
#     out_file = f'{out_dir}/features-q_interval:{q_interval}.dat'
#     print('features+labels: ', out_file)
#     features = pp.features
#     labels = pp.labels
#     dump_data((features, labels), out_file)
#
#     ##########################################################################################
#     # res = {}
#
#     # def _get_features(normal_files, interval, label='', feat_type='IAT_SIZE',
#     #                   q_interval=0.9, fft=False, header=False, out_dir='.',
#     #                   verbose=10, random_state=100):
#     #     features = []
#     #     labels = []
#     #     for i, (pcap_file, label_file) in enumerate(normal_files):
#     #         ft = FEATURES(pcap_file=pcap_file, interval=interval, q_interval=q_interval, feat_type=feat_type,
#     #                       label_file=label_file, label=f'{label}_{i}', out_dir=out_dir,
#     #                       fft=fft, header=header, verbose=verbose,
#     #                       random_state=random_state)
#     #         ft.pcap2features()
#     #         features.extend(ft.features)
#     #         labels.extend(ft.labels)
#     #     return features, labels
#     #
#     # # get normal data
#     # features, labels = _get_features(normal_files, interval, label='normal')
#     # res['norm'] = (features, labels)
#     #
#     # # get abnormal data
#     # features, labels = _get_features(abnormal_files, interval, label='abnormal')
#     # res['abnorm'] = (features, labels)
#
#     # # # get normal data
#     # norm_features = []
#     # norm_labels = []
#     # abnorm_features = []
#     # abnorm_labels = []
#     # for i, (pcap_file, label_file) in enumerate(normal_files):
#     #     if 'DS10_UNB' in pcap_file  and label_file is not None:
#     #         ft = FEATURES(pcap_file=pcap_file, interval=interval, q_interval=q_interval, feat_type=feat_type,
#     #                       label_file=label_file, label=None, out_dir=os.path.join(out_dir, f'mixed_{i}'),
#     #                       fft=fft, header=header, verbose=verbose,
#     #                       random_state=random_state)
#     #         ft.pcap2features()
#     #
#     #         _features = np.asarray(ft.features)
#     #         _labels = np.asarray(ft.labels)
#     #         bools = [_labels == f'normal']      # here the label uses label_file, not label parameters
#     #
#     #         norm_features.extend(list(_features[bools]))
#     #         norm_labels.extend([f'normal_{i}'] * len(list(_labels[bools])))
#     #         abnorm_features.extend(list(_features[not bools]))
#     #         abnorm_labels.extend([f'abnormal_{i}']*len(list(_labels[not bools])))
#     #
#     #     else:
#     #         label = 'normal'
#     #         ft = FEATURES(pcap_file=pcap_file, interval=interval, q_interval=q_interval, feat_type=feat_type,
#     #                       label_file=label_file, label=f'{label}_{i}', out_dir=os.path.join(out_dir, f'{label}_{i}'),
#     #                       fft=fft, header=header, verbose=verbose,
#     #                       random_state=random_state)
#     #         ft.pcap2features()
#     #
#     #         norm_features.extend(ft.features)
#     #         norm_labels.extend(ft.labels)
#     #
#     # # # get abnormal data
#     # for i, (pcap_file, label_file) in enumerate(abnormal_files):
#     #     if pcap_file is None:
#     #         continue
#     #     label = 'abnormal'
#     #     ft = FEATURES(pcap_file=pcap_file, interval=interval, q_interval=q_interval, feat_type=feat_type,
#     #                   label_file=label_file, label=f'{label}_{i}', out_dir=os.path.join(out_dir, f'{label}_{i}'),
#     #                   fft=fft, header=header, verbose=verbose,
#     #                   random_state=random_state)
#     #     ft.pcap2features()
#     #     abnorm_features.extend(ft.features)
#     #     abnorm_labels.extend(ft.labels)
#
#     idx = [True if 'normal' in l else False for l in labels ]
#     normal_features = features[idx]
#     normal_labels= np.asarray(labels)[idx]
#     abnormal_features = features[not idx]
#     abnormal_labels = np.asarray(labels)[not idx]
#
#     res = {'norm': (np.asarray(normal_features), np.asarray(normal_labels)), 'abnorm': (np.asarray(abnormal_features),
#                                                                                     np.asarray(abnormal_labels))}
#     print('norm:', res['norm'][0].shape, ', abnorm', res['abnorm'][0].shape)
#     out_file = f'{out_dir}/Xy-normal-abnormal.dat'
#     print(out_file)
#     dump_data(res, out_file)
#
#     return res, out_file
