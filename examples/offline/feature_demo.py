""" Main function for the feature extraction

Main steps:
	1. Parse data and extract features
	2. Create and builds models
	3. Evaluate models on variate datasets

Command:
	current directory is project_root_dir (i.e., kjl/.)
	PYTHONPATH=. PYTHONUNBUFFERED=1 python3.7 examples/offline/feature_demo.py
"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import os
import sys
from time import time

from loguru import logger as lg
from odet.ndm.model import MODEL
from odet.ndm.ocsvm import OCSVM
from odet.pparser.parser import PCAP
from sklearn.model_selection import train_test_split

from examples.offline._constants import *
from kjl.utils.tool import timer, check_path, load, dump

lg.remove()
lg.add(sys.stdout, format="{message}", level='INFO')

IN_DIR = 'examples/offline/datasets'
@timer
def pcap2feature():

	pcap_file = f'{IN_DIR}/demo.pcap'
	pp = PCAP(pcap_file, flow_pkts_thres=2, verbose=1, random_state=RANDOM_STATE)

	# extract flows from pcap
	lg.info(f'\n---1. PCAP to flows')
	pp.pcap2flows()

	# # label each flow with a label (optional)
	# label_file = 'data/demo.csv'
	# pp.label_flows(label_file=label_file)
	pp.labels = [(fid, 0) if i % 5 == 0 else (fid, 1) for i, (fid, _) in enumerate(pp.flows)]  # only for demo purpose

	# flows to subflows
	lg.info(f'\n---2. flows to subflows')
	pp.flows2subflows(q_interval=0.9)

	# extract features from each flow given feat_type
	lg.info(f'\n---3. Extract features')
	feat_type = 'IAT+SIZE'
	pp.flow2features(feat_type, fft=False, header=False)

	# dump data to disk
	lg.info(f'\n---4. Save the results')
	X, y = pp.features, pp.labels
	y = [v for fid, v in y]
	check_path(OUT_DIR)
	out_file = f'{OUT_DIR}/DEMO_{feat_type}.dat'
	dump((X, y), out_file)

	# lg.info(f"feature_shape: {pp.features.shape}, {pp.pcap2flows.tot_time},"
	#         f"{pp.flows2subflows.tot_time}, {pp.flow2features.tot_time}")

	return out_file


@timer
def model(data_file):
	# load data
	lg.info(f'\n---1.1 Load data')
	X, y = load(data_file)

	# split train and test test
	lg.info(f'\n---1.2 Split train and test data')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

	# create detection model
	lg.info(f'\n---2.1 Build one model')
	model = OCSVM(kernel='rbf', nu=0.5, random_state=RANDOM_STATE)
	model.name = 'OCSVM'
	ndm = MODEL(model, score_metric='auc', verbose=10, random_state=RANDOM_STATE)
	# learned the model from the train set
	ndm.train(X_train, y_train)

	# evaluate the learned model
	lg.info(f'\n---2.2 Evaluate the model')
	ndm.test(X_test, y_test)

	# dump data to disk
	lg.info(f'\n---3. Save the results')
	out_dir = os.path.dirname(data_file)
	dump((model, ndm.history), out_file=f'{out_dir}/{ndm.model_name}-results.dat')

	lg.info(f"train_time: {ndm.train.tot_time}, test_time: {ndm.test.tot_time}, auc: {ndm.score:.4f}")


def main():
	lg.info('\n*** Extract feature from PCAP')
	data_file = pcap2feature()

	lg.info('\n*** EValuate models')
	model(data_file)


if __name__ == '__main__':
	main()
