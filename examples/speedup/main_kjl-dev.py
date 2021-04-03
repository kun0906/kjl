""" Main function
    1. Instructions for executing the main file
        1) Change the current directory to "examples/"
            cd examples/
        2) Check the python3 veriosn
            python3 -V
        3) Execute the main file
            PYTHONPATH=../:./ python3.7 speedup/main_kjl.py > speedup/out/main_kjl.txt 2>&1 &

    Note:
        Check the memory leak issue with memory_profiler (mprof)
    E.g.:
        python3 -m memory_profiler example.py
        # Visualization
        mprof run --multiprocess <script>
        mprof plot

    # Instructions for using "mprof run"
    1) Add "mprof" path into system path
        PATH=$PATH:~/.local/bin
    2) Execute the main file
        PYTHONPATH=../:./ python3.7 mprof run -multiprocess speedup/main_kjl.py > speedup/out/main_kjl.txt 2>&1 &
        # " PYTHONPATH=../:./ python3.7 ~/.local/lib/python3.7/site-packages/mprof.py  run —multiprocess
        # speedup/main_kjl.py > speedup/out/main_kjl.txt 2>&1 & "
    3) Visualize the memory data
        mprof plot mprofile_20210201000108.dat
"""
# Authors: kun.bj@outlook.com
# License: XXX

import os
# must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '1'  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1'  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '1'  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '1' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '1' # export NUMEXPR_NUM_THREADS=6
import copy
import itertools
import os.path as pth
import time
import traceback
import joblib
import numpy as np
from joblib import delayed, Parallel
from memory_profiler import profile
import sklearn

from kjl.log import get_log
from kjl.utils.tool import execute_time, dump_data, load_data
from speedup._merge import _dat2csv, merge_res
from speedup._speedup_kjl import _get_single_result
from speedup.generate_data import generate_data_speed_up

# create a customized log instance that can print the information.
lg = get_log(level='info')
# print(sklearn.get_config())
# sklearn.set_config(working_memory=1)
# print(sklearn.get_config())
np.random.seed(42)
np.set_printoptions(precision=2, suppress=True)

DATASETS = [
    ################################################################################################################
    # UNB datasets
    # 'UNB3',
    # 'UNB_5_8_Mon',  # auc: 0.5
    # 'UNB12_comb',  # combine UNB1 and UNB2 normal and attacks
    # 'UNB13_comb',  # combine UNB1 and UNB2 normal and attacks
    # 'UNB14_comb',
    # 'UNB23_comb',  # combine UNB1 and UNB2 normal and attacks
    # 'UNB24_comb',
    # 'UNB34_comb',
    # 'UNB35_comb',
    # 'UNB12_1',  # combine UNB1 and UNB2 attacks, only use UNB1 normal
    # 'UNB13_1',
    # 'UNB14_1',
    # 'UNB23_2', # combine UNB1 and UNB2 attacks, only use UNB2 normal
    # 'UNB24_2',
    # 'UNB34_3',
    # 'UNB35_3',
    # 'UNB34_3',
    # 'UNB45_4',
    # 'UNB123_1',  # combine UNB1, UNB2, UNB3 attacks, only use UNB1 normal
    # 'UNB134_1',
    # 'UNB145_1',
    # 'UNB245_2',
    # 'UNB234_2',  # combine UNB2, UNB3, UNB4 attacks, only use UNB2 normal
    # 'UNB35_3',  # combine  UNB3, UNB5 attacks, only use UNB3 normal
    # 'UNB24',

    ################################################################################################################
    # CTU datasets
    # 'CTU21', # normal + abnormal (botnet) # normal 10.0.0.15 (too few normal flows)
    # 'CTU22',  # normal + abnormal (coinminer)
    # 'CTU31',  # normal + abnormal (botnet)   # 192.168.1.191
    # 'CTU32',  # normal + abnormal (coinminer)

    ################################################################################################################
    # MAWI datasets
    # 'MAWI32_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32',
    # 'MAWI32-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32-2',
    # 'MAWI165-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.7.165-2',  # ~25000 (flows src_dst)
    # 'ISTS1',

    ################################################################################################################
    # IOT datasets
    # 'DWSHR_2020',   # 79 flows
    # 'WSHR_2020', # 4 flows

    ################################################################################################################
    # SMTV datasets. All smtv dataset are on NOEN server: /opt/smart-tv/roku-data-20190927-182117
    # SMTV_2019      # cp -rp roku-data-20190927-182117 ~/Datasets/UCHI/IOT_2019/
    # 'SMTV1_2019',
    # 'SMTV2_2019',

    ################################################################################################################
    # Final datasets for the paper
    'UNB345_3',  # Combine UNB3, UNB3 and UNB5 attack data as attack data and only use UNB3's normal as normal data
    'CTU1',      # Two different abnormal data
    'MAWI1_2020', # Two different normal data
    'MACCDC1',    # Two different normal data
    'SFRIG1_2020', #  Two different normal data
    'AECHO1_2020', # Two different normal data
    'DWSHR_WSHR_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
]

MODELS = [
    ################################################################################################################
    # 1. OCSVM
    "OCSVM(rbf)",
    "KJL-OCSVM(linear)",
    "Nystrom-OCSVM(linear)",

    ################################################################################################################
    # # "GMM(full)", "GMM(diag)",

    # ################################################################################################################s
    # # 2. KJL/Nystrom
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
    # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"

    "Nystrom-QS-GMM(full)",   "Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"
    #
    # ################################################################################################################
    # # 4. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection and initialize GMM (set 'GMM_is_init_all'=True)
    "KJL-QS-init_GMM(full)",   "KJL-QS-init_GMM(diag)",
    "Nystrom-QS-init_GMM(full)",   "Nystrom-QS-init_GMM(diag)",
]

# if it ues grid search or not. True for best params and False for default params
GSES = [('is_gs', True), ('is_gs', False)]

# which data do we want to extract features? src_dat or src
DIRECTIONS = [('direction', 'src_dst'), ]

# Features.
FEATS = [('feat', 'iat_size'), ('feat', 'stats')]

# if it uses header in feature or not.
HEADERS = [('is_header', True), ('is_header', False)]

# if quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection or after projection
BEFORE_PROJS = [('before_proj', False), ]


def _get_data(data_pth=None, data_name=None, direction='src_dst', feat='iat_size', header=False,
              overwrite=False):
    """Load data from data_pth if data_path exists, otherwise, generate data from pcap fiels

    Parameters
    ----------
    data_pth:
    data_name
    direction
    feat
    header
    overwrite

    Returns
    -------
        X: features
        y: labels
    """
    if overwrite:
        if pth.exists(data_pth): os.remove(data_pth)

    if not pth.exists(data_pth):
        data_pth = generate_data_speed_up(data_name, feat_type=feat, header=header, direction=direction,
                                          out_file=data_pth,
                                          overwrite=overwrite)

    return load_data(data_pth)


def _get_model_cfg(model_cfg, n_repeats=5, q=0.3, kjl_n=100, kjl_d=5, n_comp=1,
                   random_state=42, verbose=10, overwrite=False, n_jobs=10):
    """ Get all params needed for an experiment except for dataset params

    Parameters
    ----------
    model_cfg
    n_repeats
    q
    kjl_n
    kjl_d
    n_comp
    random_state
    verbose
    overwrite
    n_jobs

    Returns
    -------
        updated model_cfg: dict
    """

    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    GMM_n_components = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    before_proj = model_cfg['before_proj']
    is_gs = model_cfg['is_gs']
    model_name = model_cfg['model_name']
    train_size = model_cfg['train_size']
    if 'k_qs' not in model_cfg.keys():
        model_cfg['k_qs'] = '2/3'

    # if '5/7' in model_cfg['k_qs']:
    #     quickshift_k = int(train_size ** (5 / 7))
    # elif '2/3' in model_cfg['k_qs']:
    #     quickshift_k = int(train_size ** (2 / 3))
    # elif '3/4' in model_cfg['k_qs']:
    #     quickshift_k = int(train_size ** (3 / 4))
    # else:
    #     quickshift_k = 500 if 'qs' in model_name else None

    quickshift_k = int(train_size ** (2 / 3))

    if 'GMM' in model_name:
        if 'full' in model_name:
            covariance_type = 'full'
        elif 'diag' in model_name:
            covariance_type = 'diag'
        else:
            msg = model_name
            raise NotImplementedError(msg)
    else:
        covariance_type = None


    TEMPLATE = {"model_name": model_name,  # the case name of current experiment
                "train_size": model_cfg['train_size'],
                'detector': {'detector_name': 'GMM', 'GMM_covariance_type': None, 'GMM_is_init_all': False},
                'is_gs': False,
                'before_proj': before_proj,
                'after_proj': not before_proj,
                'std': {'is_std': False, 'is_std_mean': False},  # default use std
                'kjl': {'is_kjl': False, 'kjl_d': kjl_d},
                'nystrom': {'is_nystrom': False},
                'quickshift': {'is_quickshift': False},
                'meanshift': {'is_meanshift': False},
                'random_state': random_state,
                'n_repeats': n_repeats,
                # 'q_abnormal_thres': 0.9,
                'verbose': verbose,
                'overwrite': overwrite,
                'n_jobs': n_jobs,
                }

    def create_case(template=None, **kwargs):
        """

        Parameters
        ----------
        template
        kwargs

        Returns
        -------

        """
        case = {}

        for k, v in template.items():
            if type(v) is dict:
                for _k, _v in v.items():
                    case[_k] = _v
            else:
                case[k] = v

        for k, v in kwargs.items():
            if type(v) is dict:
                for _k, _v in v.items():
                    case[_k] = _v
            else:
                case[k] = v

        return case

    if model_name == "OCSVM(rbf)":
        # case 11: OCSVM(rbf)
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'OCSVM',
                                          'OCSVM_kernel': 'rbf',
                                          'OCSVM_qs': qs if is_gs else [q],
                                          'OCSVM_nus': [0.5]
                                          },
                                is_gs=is_gs,  # default kjl=False
                                )
    elif model_name == "KJL-OCSVM(linear)":
        # case 12: KJL-OCSVM(linear)
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'OCSVM',
                                          'OCSVM_kernel': 'linear',
                                          # 'OCSVM_qs': qs if is_gs else [q],
                                          'OCSVM_nus': [0.5]
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [kjl_n],
                                     'kjl_ds': [kjl_d]
                                     }
                                )

    elif model_name == "Nystrom-OCSVM(linear)":
        # case 13: Nystrom-OCSVM(linear)
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'OCSVM',
                                          'OCSVM_kernel': 'linear',
                                          # 'OCSVM_qs': qs if is_gs else [q],
                                          'OCSVM_nus': [0.5]
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [kjl_n],
                                         'nystrom_ds': [kjl_d]
                                         }
                                )
    elif model_name == 'GMM(full)' or model_name == "GMM(diag)":
        # case 21: GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': GMM_n_components if is_gs else [n_comp]
                                          },
                                is_gs=is_gs
                                )

    elif model_name == "KJL-GMM(full)" or model_name == "KJL-GMM(diag)":
        # case 3: KJL-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': GMM_n_components if is_gs else [n_comp]
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [kjl_n],
                                     'kjl_ds': [kjl_d]
                                     }
                                )
    elif model_name == "Nystrom-GMM(full)" or model_name == "Nystrom-GMM(diag)":
        # case 4: Nystrom-GMM   # nystrom will take more time than kjl
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': GMM_n_components if is_gs else [n_comp]
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [kjl_n],
                                         'nystrom_ds': [kjl_d]
                                         }
                                )

    elif model_name == "QS-KJL-GMM(full)" or model_name == "QS-KJL-GMM(diag)" or \
            model_name == "KJL-QS-GMM(full)" or model_name == "KJL-QS-GMM(diag)":
        # case 5: QS-KJL-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []  # use QS to obtain n_components
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [kjl_n],
                                     'kjl_ds': [kjl_d]
                                     },
                                quickshift={'is_quickshift': True,
                                            'quickshift_ks': [quickshift_k],  # fix quickshift_k
                                            # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                                            'quickshift_betas': [0.9]  # fix QS
                                            }
                                )
    elif model_name == "MS-KJL-GMM(full)" or model_name == "MS-KJL-GMM(diag)" or \
            model_name == "KJL-MS-GMM(full)" or model_name == "KJL-MS-GMM(diag)":
        # case 6: MS-KJL-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []  # use MS to obtain n_components
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [kjl_n],
                                     'kjl_ds': [kjl_d]
                                     },
                                meanshift={'is_meanshift': True,
                                           'meanshift_qs': []  # meanshift and kjl use the same q
                                           }
                                )
    elif model_name == "Nystrom-QS-GMM(full)" or model_name == "Nystrom-QS-GMM(diag)" or \
            model_name == "QS-Nystrom-GMM(full)" or model_name == "QS-Nystrom-GMM(diag)":
        # case 7: Nystrom-QS-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []  # use QS to obtain n_components
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [kjl_n],
                                         'nystrom_ds': [kjl_d]
                                         },
                                quickshift={'is_quickshift': True,
                                            'quickshift_ks': [quickshift_k],
                                            # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                                            'quickshift_betas': [0.9]
                                            }
                                )
    elif model_name == "Nystrom-MS-GMM(full)" or model_name == "Nystrom-MS-GMM(diag)" or \
            model_name == "MS-Nystrom-GMM(full)" or model_name == "MS-Nystrom-GMM(diag)":
        # case 8: Nystrom-MS-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []  # use MS to obtain n_components
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [kjl_n],
                                         'nystrom_ds': [kjl_d]
                                         },
                                meanshift={'is_meanshift': True,
                                           'meanshift_qs': []
                                           }
                                )

    elif model_name == "KJL-QS-init_GMM(full)" or model_name == "KJL-QS-init_GMM(diag)":
        # case 9: QS-KJL-init_GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': [],  # use QS to obtain n_components
                                          'GMM_is_init_all': True  # use QS to initialize GMM
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [kjl_n],
                                     'kjl_ds': [kjl_d]
                                     },
                                quickshift={'is_quickshift': True,
                                            'quickshift_ks': [quickshift_k],  # fix quickshift_k
                                            # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                                            'quickshift_betas': [0.9]  # fix QS
                                            }
                                )

    elif model_name == "Nystrom-QS-init_GMM(full)" or model_name == "Nystrom-QS-init_GMM(diag)":
        # case 10: Nystrom-QS-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': [],  # use QS to obtain n_components,
                                          'GMM_is_init_all': True  # use QS to initialize GMM
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [kjl_n],
                                         'nystrom_ds': [kjl_d]
                                         },
                                quickshift={'is_quickshift': True,
                                            'quickshift_ks': [quickshift_k],
                                            # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                                            'quickshift_betas': [0.9]
                                            }
                                )
    else:
        msg = model_name
        raise NotImplementedError(msg)

    return model_cfg


class SingleEXP:
    def __init__(self, in_dir=None, n_repeats=5, q=0.3, kjl_n=100, kjl_d=5, n_comp=1,
                 random_state=42, verbose=10, overwrite=False, n_jobs=10):
        """Generate the result by one algorithm on one dataset.

        Parameters
        ----------
        in_dir
        n_repeats
        q
        kjl_n
        kjl_d
        n_comp
        random_state
        verbose
        overwrite
        n_jobs
        """
        self.in_dir = in_dir
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs

        self.q = q
        self.kjl_n = kjl_n
        self.kjl_d = kjl_d
        self.n_comp = n_comp

        self.verbose = verbose
        self.random_state = random_state
        self.overwrite = overwrite

    def get_data(self, data_cfg):
        """Get one data from data_cfg

        Parameters
        ----------
        data_cfg

        Returns
        -------

        """
        data_file = pth.join(self.in_dir, data_cfg['direction'],
                             data_cfg['feat'] + '-header_' + str(data_cfg['is_header']),
                             data_cfg['data_name'], 'Xy-normal-abnormal.dat')
        X, y = _get_data(data_pth=data_file,
                         data_name=data_cfg['data_name'], direction=data_cfg['direction'],
                         feat=data_cfg['feat'], header=data_cfg['is_header'],
                         overwrite=self.overwrite)
        self.data_cfg = data_cfg
        self.data_cfg['data'] = (X, y)
        self.data_cfg['data_file'] = data_file

        return self.data_cfg

    def get_model_cfg(self, model_cfg):
        """get all model cfg based on model_cfg

        Returns
        -------

        """
        model_cfg = _get_model_cfg(model_cfg, n_repeats=self.n_repeats,
                                   q=self.q, kjl_n=self.kjl_n, kjl_d=self.kjl_d,
                                   n_comp=self.n_comp, n_jobs=self.n_jobs,
                                   random_state=self.random_state, verbose=self.verbose,
                                   overwrite=self.overwrite)
        self.model_cfg = model_cfg

        return self.model_cfg

    def execute(self, model_cfg=None, data_cfg=None):
        """build and evaluate a model on a data

        Parameters
        ----------
        model_cfg
        data_cfg

        Returns
        -------

        """
        res = _get_single_result(model_cfg, data_cfg)
        return res

    def save_model(self):
        pass

    def save_data(self, res, out_dir=''):
        """ save res to disk

        Parameters
        ----------
        res
        out_dir

        Returns
        -------

        """
        if not pth.exists(out_dir):
            os.makedirs(out_dir)

        out_file = pth.join(out_dir, 'res.dat')
        dump_data(res, out_file=out_file)

        # (data_file, model_name), (_best_res, _middle_res) = res
        try:
            _dat2csv(res, out_file=out_file + '.csv', feat_set=self.data_cfg['feat'])
        except Exception as e:
            print(f"Error({e})")

        return out_file

    def show_model(self, model, out_dir=None):
        pass

    def show_data(self, res, out_dir=None):
        pass


def single_main(data_cfg, model_cfg, out_dir=''):
    """ Get the result by one model on one dataset

    Parameters
    ----------
    data_cfg
    model_cfg
    out_dir

    Returns
    -------

    """
    start = time.time()

    lg.info(f"data_cfg: {data_cfg}, model_cfg: {model_cfg}")
    # Store the result obtained by the model on the dataset
    res = []

    try:
        ################################################################################################################
        # 1. Initialize the parameters
        exp = SingleEXP(in_dir=f'speedup/data', n_repeats=5,
                        q=0.25, kjl_n=100, kjl_d=model_cfg['kjl_d'], n_comp=1,
                        random_state=42, verbose=10, overwrite=False,
                        n_jobs=0)

        ################################################################################################################
        # 2. Get data (X, y) from file
        data_cfg = exp.get_data(data_cfg)

        ################################################################################################################
        # 3. Get the model's parameters
        model_cfg = exp.get_model_cfg(model_cfg)
        model_cfg['out_dir'] = out_dir

        ################################################################################################################
        # 4. Run the model and get the result
        res = exp.execute(model_cfg, data_cfg)

        ################################################################################################################
        # 5. save the result to file
        out_dir = pth.join(out_dir,
                           exp.data_cfg['direction'],
                           exp.data_cfg['feat'] + "-header_" + str(exp.data_cfg['is_header']),
                           exp.data_cfg['data_name'],
                           "before_proj_" + str(exp.model_cfg['before_proj']) + \
                           "-gs_" + str(exp.model_cfg['is_gs']),
                           exp.model_cfg['model_name'] + "-std_" + str(exp.model_cfg['is_std'])
                           + "_center_" + str(exp.model_cfg['is_std_mean']) + "-d_" + str(exp.model_cfg['kjl_d']) \
                           + "-" + str(exp.model_cfg['GMM_covariance_type']))
        out_file = exp.save_data(res, out_dir=out_dir)
        lg.info(out_file)

        ################################################################################################################
        # future functions
        # exp.save_model(, out_dir=out_dir)
        # exp.show_data(res, out_dir=out_dir)

    except Exception as e:
        traceback.print_exc()
        lg.error(f"Error: {data_cfg}, {model_cfg}")

    ################################################################################################################
    end = time.time()
    time_taken = end - start

    return res, time_taken


@execute_time
@profile
def main1(directions=[('direction', 'src_dst'), ],
          feats=[('feat', 'iat_size'), ('feat', 'stats')],
          headers=[('is_header', True), ('is_header', False)],
          gses=[('is_gs', True), ('is_gs', False)],
          before_projs=[('before_proj', False), ],
          ds=[('kjl_d', 5), ], out_dir='speedup/out',
          train_sizes=[('train_size', 5000)],
          is_parallel=True):
    """ Get all results on all datasets and save the results to file.

    Parameters
    ----------
    directions
    feats
    headers
    gses
    before_projs
    ds
    out_dir
    train_sizes
    is_parallel

    Returns
    -------

    """
    start = time.time()

    # Store all the results
    res = []

    ################################################################################################################
    # 1. Get all datasets
    datasets = [('data_name', v) for v in DATASETS]
    datasets_cfg = list(itertools.product(datasets, directions, feats, headers))

    ################################################################################################################
    # 2. Get all models
    models = [('model_name', v) for v in MODELS]
    models_cfg = list(itertools.product(models, gses, before_projs, ds, train_sizes))

    ################################################################################################################
    # 3. Get the total number of the experiments and print each of them out
    n_tot = len(list(itertools.product(datasets_cfg, models_cfg)))
    lg.info(f'n_tot: {n_tot}')
    for i, (data_cfg, model_cfg) in enumerate(list(itertools.product(datasets_cfg, models_cfg))):
        lg.info(f'{i}/{n_tot}, {dict(data_cfg)}, {dict(model_cfg)}')
    n_cpus = os.cpu_count()
    lg.info(f'n_cpus: {n_cpus}')

    ################################################################################################################
    # 4. Run experiments in parallel or serial
    if is_parallel:
        # if backend='loky', the time taken is less than that of serial. but if backend='multiprocessing', we can
        # get very similar time cost comparing with serial.
        # The reason may be that loky module manages a pool of worker that can be re-used across time.
        # It provides a robust and dynamic implementation os the ProcessPoolExecutor and a function
        # get_reusable_executor() which hide the pool management under the hood.
        with Parallel(n_jobs=30, verbose=0, backend='loky') as parallel:
                res = parallel(delayed(single_main)(copy.deepcopy(dict(data_cfg_)), copy.deepcopy(dict(model_cfg_)),
                                                    copy.deepcopy(out_dir)) for data_cfg_,
                                                                                model_cfg_ in
                               list(itertools.product(datasets_cfg, models_cfg)))
    else:
        # Running each combination in serial to obtain correct train time and test time
        for i, (data_cfg_, model_cfg_) in enumerate(list(itertools.product(datasets_cfg, models_cfg))):
            res_, time_taken = single_main(copy.deepcopy(dict(data_cfg_)), copy.deepcopy(dict(model_cfg_)),
                                           copy.deepcopy(out_dir))
            res.append((res_, time_taken))
            lg.info(f'{i + 1}/{n_tot}, it takes {time_taken:.5f}s')

    ################################################################################################################
    # 5. Dump all results to disk
    dump_data(res, out_file=f'{out_dir}/res.dat')

    ################################################################################################################
    end = time.time()
    lg.info(f'\n***It takes {end - start:.5f}s to finish {n_tot} experiments!')


def main():
    out_dir = 'speedup/out/kjl_joblib_30'
    # out_dir = 'speedup/out/kjl-with_init_params'    # modify __speedup_kjl.py
    try:
        ###########################################################################################################
        # Get results with IAT_SIZE
        main1(feats=[('feat', 'iat_size')],
              headers=[('is_header', False)],
              # gses=[('is_gs', True)],
              before_projs=[('before_proj', False), ],
              ds=[('kjl_d', 5), ],
              train_sizes=[('train_size', 5000)],
              out_dir=out_dir,
              )
        # ##########################################################################################################
        # # Get results with STATS
        # main1(feats=[('feat', 'stats')],
        #      headers=[('is_header', True)],
        #      # gses=[('is_gs', False)],
        #      before_projs=[('before_proj', False), ],
        #      ds=[('kjl_d', 5), ],
        #      out_dir = out_dir,
        #      )
    except Exception as e:
        traceback.print_exc()
        lg.error(e)

    ###########################################################################################################
    # Merge all results
    merge_res(in_dir=out_dir, datasets=DATASETS,
              directions=[('direction', 'src_dst'), ],
              feats=[('feat', 'iat_size'), ('feat', 'stats'), ],
              headers=[('is_header', False)],
              models=MODELS,
              # gses=[('is_gs', True), ('is_gs', False)],
              before_projs=[('before_proj', False), ],
              ds=[('kjl_d', 5), ], )


if __name__ == '__main__':
    # with sklearn.utils.parallel_backend('loky', inner_max_num_threads=32):
    main()
