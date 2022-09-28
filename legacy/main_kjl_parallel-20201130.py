"""Main entrance
    run under "applications"
    python3 -V
    PYTHONPATH=../:./ python3.8 offline/main_kjl_parallel.py > offline/out/main_kjl_parallel.txt 2>&1 &
"""

import itertools
import os
import os.path as pth
import numpy as np
import joblib
from joblib import delayed, Parallel

from kjl.utils.data import load_data, dump_data, dat2csv
from kjl.utils.tool import execute_time
from offline._speedup_kjl import get_best_results
from offline.generate_data import generate_data_speed_up
from offline.ratio_variance import dat2xlxs, improvement, dat2latex

print('PYTHONPATH: ', os.environ['PYTHONPATH'])


def single_main(**kwargs):
    print(f'single_main.kwargs: {kwargs.items()}')
    (data_name, data_file), (X, y) = kwargs['data']
    case, params = kwargs['params']
    is_gs = params['is_gs']
    if 'GMM_covariance_type' in params.keys():
        GMM_covariance_type = params['GMM_covariance_type']
    else:
        GMM_covariance_type = None

    _best_results, _middle_results = get_best_results(X, y, params)

    result = ((f'{data_name}|{data_file}', case), (_best_results, _middle_results))
    # dump each result to disk to avoid runtime error in parallel context.
    dump_data(result, out_file=(f'{os.path.dirname(data_file)}/gs_{is_gs}-{GMM_covariance_type}/{case}.dat'))

    return result


def parallel_ray(cfg):
    import ray
    # import psutil
    #
    # # # Register Ray Backend to be called with parallel_backend("ray").
    # # register_ray()
    # num_cpus = psutil.cpu_count(logical=False)

    if cfg.is_debug:
        # Turning off parallelism for debug
        # force all Ray functions to occur on a single process with local_mode
        ray.init(local_mode=True)
    else:
        # for parallelism
        ray.init(num_cpus=1, logging_level="WARNING")  # temp_dir="log/", not working, don't know why?
    assert ray.is_initialized()

    # # config your cluster
    # # To run it on a Ray cluster add ray.init(address=”auto”) or ray.init(address=”<address:port>”) before
    # # calling with parallel_backend(“ray”).
    # ray.init(address="auto")
    # # https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/local/example-full.yaml

    # To turn a Python function f into a “remote function” (a function that can be executed remotely
    # and asynchronously)
    @ray.remote
    def _main(**kwargs):
        return single_main(**kwargs)

    # Start num_cpus tasks in parallel.
    results = []
    for i, (data, params) in \
            enumerate(itertools.product(cfg.datasets, cfg.experiments)):
        print(f'i={i}, data_key={data[0]}, params={params}')

        # 'remote' function is executed remotely and asynchronously
        # data = {(data_name, data_file): (X, y)}
        results.append(_main.remote(data=data, params=params))

    # Wait for the tasks to complete and retrieve the results.  # get() must be a list of objects.
    results = ray.get(results)
    results = {k: v for (k, v) in results}
    return results


def parallel_joblib(cfg):
    def _main(**kwargs):
        return single_main(**kwargs)

    parallel = Parallel(n_jobs=1, verbose=30)
    with parallel:
        results = parallel(delayed(_main)(data=data, params=params)
                           for (data, params) in itertools.product(cfg.datasets, cfg.experiments))

    results = {k: v for (k, v) in results}
    return results


class Config:

    def __init__(self, parallel_name='ray', is_debug=False, is_gs=False, covariance_type='diag', n_jobs = 16):
        self.parallel_name = parallel_name
        self.is_debug = is_debug
        self.random_state = 42
        self.n_repeats = 5 if not is_debug else 1
        self.verbose = 10
        self.overwrite = False
        self.is_gs = is_gs
        self.covariance_type = covariance_type

        # in parallel
        # get the number of cores. Note that, one cpu might have more than one core.
        if self.is_debug:
            self.n_jobs = 1
        else:
            # self.n_jobs = int(np.sqrt(n_jobs))
            # if 0 < self.n_jobs * self.n_jobs < n_jobs:
            #     pass
            # elif self.n_jobs*self.n_jobs == n_jobs:
            #     self.n_jobs = self.n_jobs-1 if self.n_jobs - 1 > 0 else 1
            # else:
            #     self.n_jobs = 1
            self.n_jobs = 1
        print(f'n_job: {self.n_jobs}')

        def _generate_datasets(overwrite=False):
            feat = 'iat_size'
            in_dir = f'offline/data/{feat}'
            dataname_path_mappings = {
                # 'mimic_GMM': f'{in_dir}/mimic_GMM/Xy-normal-abnormal.dat',
                # 'mimic_GMM1': f'{in_dir}/mimic_GMM1/Xy-normal-abnormal.dat',
                # 'UNB1': f'{in_dir}/UNB1/Xy-normal-abnormal.dat',
                'UNB2': f'{in_dir}/UNB2/Xy-normal-abnormal.dat',
                #     # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
                #     # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
                #     # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
                #     # # # # #
                #     # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
                #     # # # # # #
                #     'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
                'CTU1': f'{in_dir}/CTU1/Xy-normal-abnormal.dat',
                # #     # # #
                # #     # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
                # #     # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
                'MAWI1':  f'{in_dir}/MAWI1/Xy-normal-abnormal.dat',
                # #     # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
                # #     # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
                # #     # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
                # #     # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
                # #     # #
                # #     # # #
                # #     # 'WRCCDC/2020-03-20',
                # #     # 'DEFCON/ctf26',
                'ISTS1':  f'{in_dir}/ISTS1/Xy-normal-abnormal.dat',
                'MACCDC1': f'{in_dir}/MACCDC1/Xy-normal-abnormal.dat',
                #
                # #     # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
                'SCAM1': f'{in_dir}/SCAM1/Xy-normal-abnormal.dat',
                # # # #     # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
                # # # #     # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
                # # # #

            }

            # dataname_path_mappings = {
            #     # 'DEMO_IDS': 'DEMO_IDS/DS-srcIP_192.168.10.5',
            #     # 'mimic_GMM': f'{in_dir}/mimic_GMM/Xy-normal-abnormal.dat',
            #     #
            #     # 'UNB1_UNB2': f'{in_dir}/UNB1_UNB2/Xy-normal-abnormal.dat',
            #     # 'UNB1_UNB3': f'{in_dir}/UNB1_UNB3/Xy-normal-abnormal.dat',
            #     # # 'UNB1_UNB4': f'{in_dir}/UNB1_UNB4/Xy-normal-abnormal.dat',
            #     # # 'UNB1_UNB5': f'{in_dir}/UNB1_UNB5/Xy-normal-abnormal.dat',
            #     # # 'UNB2_UNB3': f'{in_dir}/UNB2_UNB3/Xy-normal-abnormal.dat',
            #     # 'UNB1_CTU1': f'{in_dir}/UNB1_CTU1/Xy-normal-abnormal.dat',
            #     # 'UNB1_MAWI1': f'{in_dir}/UNB1_MAWI1/Xy-normal-abnormal.dat',
            #     # # 'UNB2_CTU1': f'{in_dir}/UNB2_CTU1/Xy-normal-abnormal.dat',
            #     # # 'UNB2_MAWI1': f'{in_dir}/UNB2_MAWI1/Xy-normal-abnormal.dat',
            #     # 'UNB2_FRIG1': f'{in_dir}/UNB2_FRIG1/Xy-normal-abnormal.dat',
            #     # # # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
            #     # # 'UNB2_FRIG2': f'{in_dir}/UNB_FRIG2/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
            #     # #
            #     # # 'CTU1_UNB1': f'{in_dir}/CTU1_UNB1/Xy-normal-abnormal.dat',
            #     # # 'CTU1_MAWI1': f'{in_dir}/CTU1_MAWI1/Xy-normal-abnormal.dat',
            #     # # #
            #
            #     # # # # # Fridge:
            #     # # subdatasets1 = (abnormal1, normal1)  # normal(open_shut) + abnormal(idle1)
            #     # # subdatasets2 = (abnormal2, None)  # normal(browse)
            #     # 'FRIG_OPEN_BROWSE': f'{in_dir}/FRIG_OPEN_BROWSE/Xy-normal-abnormal.dat',
            #     #
            #     # # # # # Fridge:
            #     # # subdatasets1 = (abnormal2, normal1)  # normal(browse) + abnormal(idle1)
            #     # # subdatasets2 = (abnormal1, None)  # normal(open_shut)
            #     # 'FRIG_BROWSE_OPEN': f'{in_dir}/FRIG_BROWSE_OPEN/Xy-normal-abnormal.dat',
            #     #
            #     # # # # # Fridge:
            #     # # subdatasets1 = (abnormal2, normal2)  # normal(browse) + abnormal (idle2)
            #     # # subdatasets2 = (abnormal1, normal1)  # normal(open_shut) + abnormal(idle1)
            #     # # 'FRIG_IDLE12': f'{in_dir}/FRIG_IDLE12/Xy-normal-abnormal.dat',
            #     #
            #     # # # # # Fridge:
            #     # # subdatasets1 = (normal1, abnormal2)  # normal(idle1) + abnormal(browse)
            #     # # subdatasets2 = (abnormal1,None)  # normal(open_shut)
            #     # 'FRIG_IDLE1_OPEN': f'{in_dir}/FRIG_IDLE1_OPEN/Xy-normal-abnormal.dat',
            #     # open and idle have much similar flows
            #     #
            #     # # # # # # Fridge:
            #     # # # subdatasets1 = (abnormal1, abnormal2)  # normal(open_shut) + abnormal(browse)
            #     # # # subdatasets2 = (normal1,None)  # normal(idle1)
            #     # 'FRIG_OPEN_IDLE1': f'{in_dir}/FRIG_OPEN_IDLE1/Xy-normal-abnormal.dat',
            #     # #
            #     # # # # # Fridge:
            #     # # subdatasets1 = (normal1, abnormal1)  # normal(idle1) + abnormal (open_shut)
            #     # # subdatasets2 = (abnormal2,None)  # normal(browse)
            #     # 'FRIG_IDLE1_BROWSE': f'{in_dir}/FRIG_IDLE1_BROWSE/Xy-normal-abnormal.dat',
            #     # # # # # Fridge:
            #     # # subdatasets1 = (abnormal2, abnormal1)  # normal(browse) + abnormal (open_shut)
            #     # # subdatasets2 = (normal1,None)  # normal(idle1)
            #     # 'FRIG_BROWSE_IDLE1': f'{in_dir}/FRIG_BROWSE_IDLE1/Xy-normal-abnormal.dat',
            #
            #     # # # # Fridge:
            #     # # subdatasets1 = (normal2, abnormal2)  # normal(idle2) + abnormal(browse)
            #     # # subdatasets2 = (abnormal1,None)  # normal(open_shut)
            #     # 'FRIG_IDLE2_OPEN': f'{in_dir}/FRIG_IDLE2_OPEN/Xy-normal-abnormal.dat',
            #
            #     # # # # # Fridge:
            #     # # subdatasets1 = (abnormal1, abnormal2)  # normal(open_shut) + abnormal(browse)
            #     # # subdatasets2 = (normal2,None)  # normal(idle2)
            #     # 'FRIG_OPEN_IDLE2': f'{in_dir}/FRIG_OPEN_IDLE2/Xy-normal-abnormal.dat',
            #     #
            #
            #     # # # # # # Fridge:
            #     # # # subdatasets1 = (normal2, abnormal1)  # normal(idle2) + abnormal (open_shut)
            #     # # # subdatasets2 = (abnormal2,None)  # normal(browse)
            #     # 'FRIG_IDLE2_BROWSE': f'{in_dir}/FRIG_IDLE2_BROWSE/Xy-normal-abnormal.dat',
            #     # # # # # # Fridge:
            #     # # # subdatasets1 = (abnormal2, abnormal1)  # normal(browse) + abnormal (open_shut)
            #     # # # subdatasets2 = (normal2,None)  # normal(idle2)
            #     # 'FRIG_BROWSE_IDLE2': f'{in_dir}/FRIG_BROWSE_IDLE2/Xy-normal-abnormal.dat',
            #
            #     # # # # # # Fridge:
            #     # # # subdatasets1 = (normal2, abnormal1)  # normal(idle2) + abnormal (open_shut)
            #     # # # subdatasets2 = (abnormal2,None)  # normal(browse)
            #     # 'FRIG_IDLE2_BROWSE': f'{in_dir}/FRIG_IDLE2_BROWSE/Xy-normal-abnormal.dat',
            #     # # # # # # Fridge:
            #     # # # subdatasets1 = (abnormal2, abnormal1)  # normal(browse) + abnormal (open_shut)
            #     # # # subdatasets2 = (normal2,None)  # normal(idle2)
            #     # 'FRIG_BROWSE_IDLE2': f'{in_dir}/FRIG_BROWSE_IDLE2/Xy-normal-abnormal.dat',
            #
            #     # # #
            #     # 'MAWI1_UNB1': f'{in_dir}/MAWI1_UNB1/Xy-normal-abnormal.dat',
            #     # 'MAWI1_CTU1': f'{in_dir}/MAWI1_CTU1/Xy-normal-abnormal.dat',  # works
            #     # 'MAWI1_UNB2': f'{in_dir}/MAWI1_UNB2/Xy-normal-abnormal.dat',
            #     # 'CTU1_UNB2': f'{in_dir}/CTU1_UNB2/Xy-normal-abnormal.dat',
            #     #
            #     # 'UNB1_FRIG1': f'{in_dir}/UNB1_FRIG1/Xy-normal-abnormal.dat',
            #     # # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
            #     # 'CTU1_FRIG1': f'{in_dir}/CTU1_FRIG1/Xy-normal-abnormal.dat',
            #     # # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
            #     # 'MAWI1_FRIG1': f'{in_dir}/MAWI1_FRIG1/Xy-normal-abnormal.dat',
            #     # # MAWI1+Fridge: (normal: idle) abnormal: (open_shut)
            #     #
            #     # 'FRIG1_UNB1': f'{in_dir}/FRIG1_UNB1/Xy-normal-abnormal.dat',
            #     # # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
            #     # 'FRIG1_CTU1': f'{in_dir}/FRIG1_CTU1/Xy-normal-abnormal.dat',
            #     # # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
            #     # 'FRIG1_MAWI1': f'{in_dir}/FRIG1_MAWI1/Xy-normal-abnormal.dat',
            #     #
            #     # 'UNB1_FRIG2': f'{in_dir}/UNB1_FRIG2/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
            #     # 'CTU1_FRIG2': f'{in_dir}/CTU1_FRIG2/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (browse)
            #     # 'MAWI1_FRIG2': f'{in_dir}/MAWI1_FRIG2/Xy-normal-abnormal.dat',
            #     # # MAWI1+Fridge: (normal: idle) abnormal: (browse)
            #     #
            #     # 'FRIG2_UNB1': f'{in_dir}/FRIG2_UNB1/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
            #     # 'FRIG2_CTU1': f'{in_dir}/FRIG2_CTU1/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (browse)
            #     # 'FRIG2_MAWI1': f'{in_dir}/FRIG2_MAWI1/Xy-normal-abnormal.dat',
            #     # # MAWI1+Fridge: (normal: idle) abnormal: (browse)
            #     # # #
            #
            #     ## SCAM has less than 100 abnormal flows, so it cannot be used
            #     ## 'UNB1_SCAM1': f'{in_dir}/UNB1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
            #     ## 'CTU1_SCAM1': f'{in_dir}/CTU1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
            #     ## 'MAWI1_SCAM1': f'{in_dir}/MAWI1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
            #     ## 'FRIG1_SCAM1': f'{in_dir}/FRIG1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
            #     # # 'FRIG2_SCAM1': f'{in_dir}/FRIG2_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
            #     # #
            #     # 'MACCDC1_UNB1': f'{in_dir}/MACCDC1_UNB1/Xy-normal-abnormal.dat',
            #     # # 'MACCDC1_CTU1': f'{in_dir}/MACCDC1_CTU1/Xy-normal-abnormal.dat',
            #     # 'MACCDC1_MAWI1': f'{in_dir}/MACCDC1_MAWI1/Xy-normal-abnormal.dat',
            #     # #
            #     # # # less flows of wshr1
            #     # # 'UNB1_DRYER1': f'{in_dir}/UNB1_DRYER1/Xy-normal-abnormal.dat',
            #     # # 'DRYER1_UNB1': f'{in_dir}/DRYER1_UNB1/Xy-normal-abnormal.dat',
            #     # #
            #     # # # it works
            #     # 'UNB1_DWSHR1': f'{in_dir}/UNB1_DWSHR1/Xy-normal-abnormal.dat',
            #     # # 'DWSHR1_UNB1': f'{in_dir}/DWSHR1_UNB1/Xy-normal-abnormal.dat',
            #     # #
            #     # 'FRIG1_DWSHR1': f'{in_dir}/FRIG1_DWSHR1/Xy-normal-abnormal.dat',
            #     # # 'FRIG2_DWSHR1': f'{in_dir}/FRIG2_DWSHR1/Xy-normal-abnormal.dat',
            #     # # 'CTU1_DWSHR1': f'{in_dir}/CTU1_DWSHR1/Xy-normal-abnormal.dat',
            #     # # 'MAWI1_DWSHR1': f'{in_dir}/MAWI1_DWSHR1/Xy-normal-abnormal.dat',
            #     # # 'MACCDC1_DWSHR1': f'{in_dir}/MACCDC1_DWSHR1/Xy-normal-abnormal.dat',
            #     # #
            #     # # less flows of wshr1
            #     # 'UNB1_WSHR1': f'{in_dir}/UNB1_WSHR1/Xy-normal-abnormal.dat',
            #     # 'WSHR1_UNB1': f'{in_dir}/WSHR1_UNB1/Xy-normal-abnormal.dat',
            #
            # }

            print(f'len(dataname_path_mappings): {len(dataname_path_mappings)}')

            datasets = {}
            for data_name, data_path in dataname_path_mappings.items():
                if overwrite:
                    if pth.exists(data_path): os.remove(data_path)

                if not pth.exists(data_path):
                    data_path = generate_data_speed_up(data_name, out_file=data_path, overwrite=overwrite)
                X, y = load_data(data_path)
                datasets[(data_name, data_path)] = (X, y)

            datasets = list(datasets.items())

            return datasets

        self.datasets = _generate_datasets(overwrite=self.overwrite)

        def _generate_experiments(is_gs=True, covariance_type='diag', **kwargs):

            TEMPLATE = {'detector': {'detector_name': 'GMM'},
                        'is_gs': False,
                        # 'std': {'is_std': False, 'is_means_std': False},  # default use std
                        'kjl': {'is_kjl': False},
                        'nystrom': {'is_nystrom': False},
                        'quickshift': {'is_quickshift': False},
                        'meanshift': {'is_meanshift': False},
                        'random_state': kwargs['random_state'],
                        'n_repeats': kwargs['n_repeats'],
                        'q_abnormal_thres': 0.9,
                        'verbose': kwargs['verbose'],
                        'overwrite': kwargs['overwrite'],
                        'n_jobs': kwargs['n_jobs'],
                        }

            def create_case(template=TEMPLATE, **kwargs):
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

            experiments = {
                # # # # case 1: OCSVM-gs:True
                'case1': create_case(template=TEMPLATE,
                                     detector={'detector_name': 'OCSVM',
                                               'OCSVM_qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                                               if is_gs else [0.3],
                                               'OCSVM_nus': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                                               if is_gs else [0.5]},
                                     is_gs=is_gs),

                # case 2: GMM-gs:True
                # {'detector_name': 'GMM', 'covariance_type': 'full', 'gs': gs},
                'case2': create_case(template=TEMPLATE,
                                     detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                               'GMM_n_components': [1, 5, 10, 15, 20, 25, 30, 35, 40,
                                                                    45] if is_gs else [1]},
                                     is_gs=is_gs),

                # case 3: GMM-gs:True-kjl:True
                'case3': create_case(template=TEMPLATE,
                                     detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                               'GMM_n_components': [1, 5, 10, 15, 20, 25, 30, 35, 40,
                                                                    45] if is_gs else [1]},
                                     is_gs=is_gs,
                                     kjl={'is_kjl': True,
                                          'kjl_qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] if is_gs
                                          else [0.3], 'kjl_ns': [100], 'kjl_ds': [10]}),

                # case 4: GMM-gs:True-nystrom:True   # nystrom will take more time than kjl
                'case4': create_case(template=TEMPLATE,
                                     detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                               'GMM_n_components': [1, 5, 10, 15, 20, 25, 30, 35, 40,
                                                                    45] if is_gs else [1]},
                                     is_gs=is_gs,
                                     nystrom={'is_nystrom': True,
                                              'nystrom_qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] if is_gs
                                              else [0.3], 'nystrom_ns': [100], 'nystrom_ds': [10]}),

                # # case 5: GMM-gs:True-kjl:True-quickshift:True
                'case5': create_case(template=TEMPLATE,
                                     detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                               'GMM_n_components': [1, 5, 10, 15, 20, 25, 30, 35, 40,
                                                                    45] if is_gs else [1]},
                                     is_gs=is_gs,
                                     kjl={'is_kjl': True,
                                          'kjl_qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] if is_gs
                                          else [0.3], 'kjl_ns': [100], 'kjl_ds': [10]},
                                     quickshift={'is_quickshift': True, 'quickshift_ks': [100, 300] if is_gs else [100],
                                                 'quickshift_betas': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                      0.95] if is_gs else [0.3]}),

                # case 6: GMM-gs:True-kjl:True-meanshift:True
                'case6': create_case(template=TEMPLATE,
                                     detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                               'GMM_n_components': [1, 5, 10, 15, 20, 25, 30, 35, 40,
                                                                    45] if is_gs else [1]},
                                     is_gs=is_gs,
                                     kjl={'is_kjl': True,
                                          'kjl_qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] if is_gs
                                          else [0.3], 'kjl_ns': [100], 'kjl_ds': [10]},
                                     meanshift={'is_meanshift': True,
                                                'meanshift_qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                 0.95] if is_gs
                                                else [0.3]})
            }

            return experiments

        experiments = _generate_experiments(is_gs=self.is_gs, covariance_type=self.covariance_type, n_jobs=self.n_jobs,
                                            n_repeats=self.n_repeats, random_state=self.random_state,
                                            verbose=self.verbose, overwrite=self.overwrite)

        self.experiments = list(experiments.items())


def results2speedup(results, out_dir):
    out_file = f'{out_dir}/all_results.csv'
    print(f'\n\n***{out_file}***')
    out_dat = pth.splitext(out_file)[0] + '.dat'
    dump_data(results, out_dat)
    print(out_dat)
    out_csv = dat2csv(results, out_file)

    out_file = dat2xlxs(out_dat, out_file=out_dat + '.xlsx')
    out_xlsx = improvement(out_file, feat_set='iat_size', out_file=os.path.splitext(out_file)[0] + '-ratio.xlsx')
    print(out_xlsx)

    # for paper
    out_latex = dat2latex(out_xlsx, out_file=os.path.splitext(out_file)[0] + '-latex.xlsx')
    print(out_latex)

    return out_xlsx


@execute_time
def main(is_gs=False, covariance_type='diag', n_jobs = 16):
    # config:
    cfg = Config(parallel_name='joblib', n_jobs = n_jobs, is_debug=False, is_gs=is_gs, covariance_type=covariance_type)

    if cfg.parallel_name == 'ray':
        results = parallel_ray(cfg)
    elif cfg.parallel_name == 'joblib':
        results = parallel_joblib(cfg)
    else:
        msg = cfg.parallel_name
        raise NotImplementedError(msg)

    print(f'\n\nfinal results: {results}')

    # save results first
    out_dir = f'offline/out/iat_size-gs_{is_gs}-{covariance_type}'
    out_file = results2speedup(results, out_dir)
    print("\n\n---finish succeeded!")


if __name__ == '__main__':
    gses = [True, False]
    covariance_types = ['full', 'diag']
    combs = list(itertools.product(gses, covariance_types))
    print(combs)
    parallel = Parallel(n_jobs=1, verbose=30)
    with parallel:
        results = parallel(delayed(main)(is_gs=is_gs, covariance_type=covariance_type, n_jobs =joblib.cpu_count()//len(combs))
                           for (is_gs, covariance_type) in itertools.product(gses, covariance_types))
    # for is_gs, covariance_type in itertools.product([False], ['diag', 'full']):
    #     main(is_gs = is_gs, covariance_type=covariance_type)
