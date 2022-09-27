""" Includes all configurations, such as constants and global random_state.
    1. set a random seed for os, random, and so on.
    2. constants
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import datetime
import os
import random
import sys

import numpy as np

#############################################################################################
# 1. random state control in order to achieve reproductive results
# ref: https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
START_TIME = datetime.datetime.now()
print(START_TIME)
print(sys.path)
# Seed value
# Apparently you may use different seed values at each stage
RANDOM_STATE = 42
# 1). Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# 2). Set the `python` built-in pseudo-random generator at a fixed value
random.seed(RANDOM_STATE)

# 3). Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(RANDOM_STATE)
#
# # 4). set torch
# import torch
#
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#############################################################################################
"""2. Constant
"""
OVERWRITE = False
ORIG_DIR = '../Datasets'    # original datasets' directory
OUT_DIR = 'examples/offline/out'
FEATURES = ['IAT+SIZE', 'STATS']
HEADERS = [False, True]
FLOW_DIRECTION = 'src_dst'       # aggregate all packets in both directions (i.e., src + dst)
TUNINGS = [False, True]
DATASETS = [
    # 'DUMMY',
	 # Final datasets for the paper
    'UNB3_345',  # Combine UNB3, UNB4 and UNB5 attack data as attack data and only use UNB3's normal as normal data
    'CTU1',      # Two different abnormal data
    'MAWI1_2020', # Two different normal data
    'MACCDC1',    # Two different normal data
    # 'SFRIG1_2020', #  Two different normal data
    'SFRIG1_2021',  # Two different normal data: # extract from data-clean.zip (collected at 2021 for human activity recognition: contains pcap and videos)
    'AECHO1_2020', # Two different normal data
    'DWSHR_AECHO_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
    # 'DWSHR_WSHR_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
    # 'DWSHR_WSHR_AECHO_2020'
]

data_orig2name = {
                  # Final datasets for the paper
                 'UNB3_345': 'UNB', # Combine UNB3, UNB4 and UNB5 attack data as attack data and only use UNB3's normal as normal data
                 'CTU1': 'CTU',  # Two different abnormal data
                 'MAWI1_2020': 'MAWI',  # Two different normal data
                 'MACCDC1':'MACCDC',  # Two different normal data
                 'SFRIG1_2021':'SFRIG',  # Two different normal data: # extract from data-clean.zip (collected at 2021 for human activity recognition: contains pcap and videos)
                 'AECHO1_2020':'AECHO',  # Two different normal data
                 # 'DWSHR_WSHR_2020':'DWSHR',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
                     'DWSHR_AECHO_2020':'DWSHR'
}

data_name2orig = {v: k for k, v in data_orig2name.items()}

# might not needed
model_orig2name = {}

model_name2orig = {v: k for k, v in model_orig2name.items()}

MODELS = [
    ################################################################################################################
    # # 1. OCSVM
    "OCSVM(rbf)",
    "KJL-OCSVM(linear)",
    "Nystrom-OCSVM(linear)",

    "KDE",

    ################################################################################################################
    "GMM(full)", "GMM(diag)",

    # # ################################################################################################################s
    # # # 2. KJL/Nystrom
    "KJL-GMM(full)", "KJL-GMM(diag)",
    "Nystrom-GMM(full)", "Nystrom-GMM(diag)",

    ################################################################################################################
    # quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    # "QS-KJL-GMM(full)", "QS-KJL-GMM(diag)",
    # "MS-KJL-GMM(full)", "MS-KJL-GMM(diag)",

    # "QS-Nystrom-GMM(full)", "QS-Nystrom-GMM(diag)",
    # "MS-Nystrom-GMM(full)", "MS-Nystrom-GMM(diag)",

    ################################################################################################################
    # 3. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    "KJL-QS-GMM(full)",   "KJL-QS-GMM(diag)",
    # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)",

    "Nystrom-QS-GMM(full)",   "Nystrom-QS-GMM(diag)",
    # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"
    #
    # ################################################################################################################
    # # 4. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection and initialize GMM (set 'GMM_is_init_all'=True)
    # "KJL-QS-init_GMM(full)",   "KJL-QS-init_GMM(diag)",
    # "Nystrom-QS-init_GMM(full)",   "Nystrom-QS-init_GMM(diag)",
]


#############################################################################################
"""3. log 

"""
from loguru import logger as lg

lg.remove()
lg.add(sys.stdout, level='DEBUG')
