"""Main entrance
    run under "examples"
    python3 -V
    PYTHONPATH=../:./ python3.8 speedup/main_kjl.py > speedup/out/main_kjl.txt 2>&1 &

    # memory_profiler: for debugging memory leak
    python -m memory_profiler example.py

    mprof run --multiprocess <script>
    mprof plot

    PATH=$PATH:~/.local/bin

    PYTHONPATH=../:./ python3.8 mprof run -multiprocess speedup/main_kjl.py > speedup/out/main_kjl.txt 2>&1 &
    PYTHONPATH=../:./ python3.7 ~/.local/lib/python3.7/site-packages/mprof.py  run —multiprocess speedup/main_kjl.py > speedup/out/main_kjl.txt 2>&1 &


    mprof plot mprofile_20210201000108.dat
"""
import itertools
import os
import os.path as pth
import time
import  traceback
from joblib import delayed, Parallel
from kjl.log import get_log
from kjl.utils.tool import execute_time, dump_data, load_data
from speedup._speedup_kjl import single_main
from speedup.generate_data import generate_data_speed_up
from speedup._merge import _dat2csv, merge_res
from memory_profiler import profile

# get log
lg = get_log(level='info')

DATASETS = [
    # 'UNB3',
    # 'UNB_5_8_Mon',  # auc: 0.5
    # 'UNB12_comb',         # combine UNB1 and UNB2 normal and attacks
    # 'UNB13_comb',         # combine UNB1 and UNB2 normal and attacks
    # 'UNB14_comb',
    # 'UNB23_comb',  # combine UNB1 and UNB2 normal and attacks
    # 'UNB24_comb',
    # 'UNB34_comb',
    # 'UNB35_comb',
    # 'UNB24',
    'UNB345_3',
    # 'CTU1',
    # # # # # 'CTU21', # normal + abnormal (botnet) # normal 10.0.0.15
    # # # # # # 'CTU22',  # normal + abnormal (coinminer)
    # # # # # 'CTU31',  # normal + abnormal (botnet)   # 192.168.1.191
    # # # # # 'CTU32',  # normal + abnormal (coinminer)
    # 'MAWI1_2020',
    # # # # # # 'MAWI32_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32',
    # # # # # 'MAWI32-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32-2',
    # # # # # 'MAWI165-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.7.165-2',  # ~25000 (flows src_dst)
    'ISTS1',
    'MACCDC1',
    'SFRIG1_2020',
    'AECHO1_2020',
    'DWSHR_WSHR_2020',  # only use Dwshr normal, and combine Dwshr and wshr novelties
]


def _get_model_cfg(model_cfg, n_repeats=5, q=0.3, n_kjl=100, d_kjl=5, n_comp=1,
                   random_state=42, verbose=10, overwrite=False, n_jobs=10):
    """ Get all params needed for an experiment except for dataset params

    Parameters
    ----------
    model_cfg
    n_repeats
    q
    n_kjl
    d_kjl
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

    before_proj = model_cfg['before_proj']
    is_gs = model_cfg['is_gs']
    model_name = model_cfg['model_name']
    train_size = model_cfg['train_size']
    if '5/7' in model_cfg['k_qs']:
        quickshift_k = int(train_size ** (5/7))
    elif '2/3' in model_cfg['k_qs']:
        quickshift_k = int(train_size ** (2/3))
    elif '3/4' in model_cfg['k_qs']:
        quickshift_k = int(train_size ** (3 / 4))
    else:
        quickshift_k = 500 if 'qs' in model_name else None
    # if 'OCSVM' in model_name:
    #     if 'rbf' in model_name:
    #         kernel = 'rbf'
    #     elif 'linear' in model_name:
    #         kernel = 'linear'
    #     else:
    #         msg = model_name
    #         raise NotImplementedError(msg)

    if 'GMM' in model_name:
        if 'full' in model_name:
            covariance_type = 'full'
        elif 'diag' in model_name:
            covariance_type = 'diag'
        else:
            msg = model_name
            raise NotImplementedError(msg)

    # if 'KJL' in model_name:
    #     is_kjl = True
    # else:
    #     is_kjl = False

    TEMPLATE = {"model_name": model_name,  # the case name of current experiment
                "train_size": train_size,
                'detector': {'detector_name': 'GMM', 'GMM_covariance_type': None},
                'is_gs': False,
                'k_qs_str': f'{quickshift_k}-'+model_cfg['k_qs'].replace('/', '_'), # for quickshift k.
                'before_proj': before_proj,
                'after_proj': not before_proj,
                'std': {'is_std': False, 'is_std_mean': False},  # default use std
                'kjl': {'is_kjl': False, 'd_kjl': d_kjl},
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

    if model_name == "OCSVM(rbf)":
        # case 1: OCSVM(rbf)
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
                                     'kjl_qs': qs if is_gs else [q],  # kjl and OCSVM use the same q
                                     'kjl_ns': [n_kjl],
                                     'kjl_ds': [d_kjl]
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
                                         'nystrom_ns': [n_kjl],
                                         'nystrom_ds': [d_kjl]
                                         }
                                )
    elif model_name == 'GMM(full)' or model_name == "GMM(diag)":
        # case 2: GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': [1, 5, 10, 15, 20, 25, 30, 35, 40,
                                                               45] if is_gs else [n_comp]
                                          },
                                is_gs=is_gs
                                )

    elif model_name == "KJL-GMM(full)" or model_name == "KJL-GMM(diag)":
        # case 3: KJL-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                                          if is_gs else [n_comp]
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [n_kjl],
                                     'kjl_ds': [d_kjl]
                                     }
                                )
    elif model_name == "Nystrom-GMM(full)" or model_name == "Nystrom-GMM(diag)":
        # case 4: Nystrom-GMM   # nystrom will take more time than kjl
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': [1, 4, 6, 8, 10, 12, 14, 16, 18, 20] if is_gs
                                          else [n_comp]
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [n_kjl],
                                         'nystrom_ds': [d_kjl]
                                         }
                                )
    # quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    elif model_name == "QS-KJL-GMM(full)" or model_name == "QS-KJL-GMM(diag)" or \
            model_name == "KJL-QS-GMM(full)" or model_name == "KJL-QS-GMM(diag)":
        # # case 5: QS-KJL-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [n_kjl],
                                     'kjl_ds': [d_kjl]
                                     },
                                quickshift={'is_quickshift': True,
                                            'quickshift_ks':[quickshift_k],    #[500],
                                            # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                                            'quickshift_betas': [0.9]  # only tune beta for QS
                                            }
                                )
    elif model_name == "MS-KJL-GMM(full)" or model_name == "MS-KJL-GMM(diag)" or \
            model_name == "KJL-MS-GMM(full)" or model_name == "KJL-MS-GMM(diag)":
        # case 6: MS-KJL-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []
                                          },
                                is_gs=is_gs,
                                kjl={'is_kjl': True,
                                     'kjl_qs': qs if is_gs else [q],
                                     'kjl_ns': [n_kjl],
                                     'kjl_ds': [d_kjl]
                                     },
                                meanshift={'is_meanshift': True,
                                           'meanshift_qs': []  # meanshift and kjl use the same q
                                           }
                                )
    # quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    elif model_name == "Nystrom-QS-GMM(full)" or model_name == "Nystrom-QS-GMM(diag)" or \
            model_name == "QS-Nystrom-GMM(full)" or model_name == "QS-Nystrom-GMM(diag)":
        # case 7: Nystrom-QS-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [n_kjl],
                                         'nystrom_ds': [d_kjl]
                                         },
                                quickshift={'is_quickshift': True,
                                            'quickshift_ks': [quickshift_k],
                                            # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                                            'quickshift_betas': qs if is_gs else [0.9]
                                            }
                                )
    elif model_name == "Nystrom-MS-GMM(full)" or model_name == "Nystrom-MS-GMM(diag)" or \
            model_name == "MS-Nystrom-GMM(full)" or model_name == "MS-Nystrom-GMM(diag)":
        # case 8: Nystrom-MS-GMM
        model_cfg = create_case(template=TEMPLATE,
                                detector={'detector_name': 'GMM',
                                          'GMM_covariance_type': covariance_type,
                                          'GMM_n_components': []
                                          },
                                is_gs=is_gs,
                                nystrom={'is_nystrom': True,
                                         'nystrom_qs': qs if is_gs else [q],
                                         'nystrom_ns': [n_kjl],
                                         'nystrom_ds': [d_kjl]
                                         },
                                meanshift={'is_meanshift': True,
                                           'meanshift_qs': []
                                           }
                                )
    else:
        msg = model_name
        raise NotImplementedError(msg)

    return model_cfg


MODELS = [  # algorithm name
    "OCSVM(rbf)",
    # "KJL-OCSVM(linear)",
    # "Nystrom-OCSVM(linear)",
    #
    # # "GMM(full)", "GMM(diag)",
    #
    # "KJL-GMM(full)", "KJL-GMM(diag)",
    #
    # "Nystrom-GMM(full)", "Nystrom-GMM(diag)",
    #
    # # quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    # # "QS-KJL-GMM(full)", "QS-KJL-GMM(diag)",
    # # "MS-KJL-GMM(full)", "MS-KJL-GMM(diag)",
    #
    # # "QS-Nystrom-GMM(full)", "QS-Nystrom-GMM(diag)",
    # # "MS-Nystrom-GMM(full)", "MS-Nystrom-GMM(diag)",
    #
    # # quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    "KJL-QS-GMM(full)", #"KJL-QS-GMM(diag)",
    # # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"
    #
    # "Nystrom-QS-GMM(full)", "Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"

]

# if it ues grid search or not. True for best params and False for default params
GSES = [('is_gs', True), ('is_gs', False)]

# which data do we want to extract features? src_dat or src
DIRECTIONS = [('direction', 'src_dst'), ]

# Features.
FEATS = [('feat', 'iat_size'), ('feat', 'stats')]

# if it ues header in feature or not.
HEADERS = [('is_header', True), ('is_header', False)]

# if quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection or after projection
BEFORE_PROJS = [('before_proj', False), ]


def _get_data(data_pth=None, data_name=None, direction='src_dst', feat='iat_size', header=False,
              overwrite=False):
    if overwrite:
        if pth.exists(data_pth): os.remove(data_pth)

    if not pth.exists(data_pth):
        data_pth = generate_data_speed_up(data_name, feat_type=feat, header=header, direction=direction,
                                          out_file=data_pth,
                                          overwrite=overwrite)

    return load_data(data_pth)


class SingleEXP:
    def __init__(self, in_dir=None, n_repeats=5, q=0.3, n_kjl=100, d_kjl=5, n_comp=1,
                 random_state=42, verbose=10, overwrite=False, n_jobs=10):
        self.in_dir = in_dir
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs

        self.q = q
        self.n_kjl = n_kjl
        self.d_kjl = d_kjl
        self.n_comp = n_comp

        self.verbose = verbose
        self.random_state = random_state
        self.overwrite = overwrite

    def get_data(self, data_cfg):
        """ get one data from data_cfg

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
                                   q=self.q, n_kjl=self.n_kjl, d_kjl=self.d_kjl,
                                   n_comp=self.n_comp, n_jobs=self.n_jobs,
                                   random_state=self.random_state, verbose=self.verbose,
                                   overwrite=self.overwrite)
        self.model_cfg = model_cfg
        return self.model_cfg

    def execute(self, model_cfg=None, data_cfg=None):
        """build and evaluate a model on a data

        Parameters
        ----------
        model
        data

        Returns
        -------

        """
        res = single_main(model_cfg, data_cfg)
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


def _main(data_cfg, model_cfg, out_dir =''):
    start = time.time()
    lg.info(f"data_cfg: {data_cfg}, model_cfg: {model_cfg}")
    res=''
    try:
        exp = SingleEXP(in_dir= f'speedup/data', n_repeats=5,
                        q=0.25, n_kjl=100, d_kjl=model_cfg['d_kjl'], n_comp=1,
                        random_state=42, verbose=10, overwrite=False,
                        n_jobs=5)
        data_cfg = exp.get_data(data_cfg)
        model_cfg = exp.get_model_cfg(model_cfg)

        res = exp.execute(model_cfg, data_cfg)
        # exp.save_model()

        # update out_dir
        if 'KJL' in model_cfg['model_name']:
            k_qs_str = model_cfg['k_qs_str']
        else:
            k_qs_str = None
        out_dir = pth.join(out_dir,
                           exp.data_cfg['direction'],
                           exp.data_cfg['feat'] + "-header_" + str(exp.data_cfg['is_header']),
                           exp.data_cfg['data_name'],
                           "before_proj_" + str(exp.model_cfg['before_proj']) + \
                           "-gs_" + str(exp.model_cfg['is_gs']),
                           exp.model_cfg['model_name'] + "-std_" + str(exp.model_cfg['is_std'])
                           + "_center_" + str(exp.model_cfg['is_std_mean']) + "-d_" + str(exp.model_cfg['d_kjl']) \
                           + "-" + str(exp.model_cfg['GMM_covariance_type'])+'-train_size_'
                           + str(exp.model_cfg['train_size']) + f'-k_qs_{k_qs_str}')
        out_file = exp.save_data(res, out_dir=out_dir)
        exp.show_data(res, out_dir=out_dir)
    except Exception as e:
        traceback.print_exc()
        lg.error(f"{data_cfg}, {model_cfg}")

    end = time.time()
    time_token = end - start
    return res, time_token


@execute_time
@profile
def main(directions=[('direction', 'src_dst'), ],
         feats=[('feat', 'iat_size'), ('feat', 'stats')],
         headers=[('is_header', True), ('is_header', False)],
         gses=[('is_gs', True), ('is_gs', False)],
         before_projs=[('before_proj', False), ],
         ds=[('d_kjl', 5), ],
         train_sizes = [('train_size', 5000)],
        k_qs = [('k_qs', 5000**(5/7)), ('k_qs', 5000**(2/3))],
         out_dir = 'speedup/out',
         is_parallel=True):
    # Store all the results
    res = []
    datasets = [('data_name', v) for v in DATASETS]
    datasets_cfg = list(itertools.product(datasets, directions, feats, headers))
    models = [('model_name', v) for v in MODELS]
    models_cfg = list(itertools.product(models, gses, before_projs, ds, train_sizes, k_qs))

    # Total number of experiments
    n_tot = len(list(itertools.product(datasets_cfg, models_cfg)))
    lg.info(f'n_tot: {n_tot}')
    for i, (data_cfg, model_cfg) in enumerate(list(itertools.product(datasets_cfg, models_cfg))):
        lg.info(f'{i}/{n_tot}, {dict(data_cfg)}, {dict(model_cfg)}')
    n_cpus = os.cpu_count()
    lg.info(f'n_cpus: {n_cpus}')
    if is_parallel:
        # It doesn't work well (will be killed by the server). Not sure what's the issue about it.
        # n_jobs(=5) * _single_main(n_job=10) = 5*10, too large
        parallel = Parallel(n_jobs=5, verbose=30)
        with parallel:
            res = parallel(delayed(_main)(dict(data_cfg), dict(model_cfg), out_dir) for data_cfg, model_cfg, in
                           list(itertools.product(datasets_cfg, models_cfg)))
    else:
        # Run each combination in sequence. It's slow but it works.
        for i, (data_cfg, model_cfg) in enumerate(list(itertools.product(datasets_cfg, models_cfg))):
            res_, time_token = _main(dict(data_cfg), dict(model_cfg), out_dir)
            res.append(res_)
            lg.info(f'{i + 1}/{n_tot}, it takes {time_token:.5f}s')

    # dump all data to disk
    dump_data(res, out_file=f'{out_dir}/res.dat')
    lg.info('\n\n***finish!')


if __name__ == '__main__':
    try:
        # # IAT_SIZE
        # out_dir = 'speedup/out_train_sizes/'
        out_dir = 'speedup/out_train_sizes/keep_small_clusters'  # should change _speedup_kjl (self.thres_n = 0), keep all small clusters
        main(feats=[('feat', 'iat_size'), ],
             headers=[('is_header', False)],
             before_projs=[('before_proj', False), ],
             ds=[('d_kjl', 5), ],
             train_sizes= [('train_size', v*1000) for v in list(range(1, 5+1, 1))],
             # train_sizes=[('train_size', 100), ('train_size', 200),('train_size', 400) ,('train_size', 600), ('train_size', 800),('train_size', 1000)],
             k_qs= [('k_qs', '5/7'), ('k_qs', '2/3'), ('k_qs', '3/4')], # [('k_qs', '3/4')],
             # gses=[('is_gs', False)],
             out_dir=out_dir,
             )
        # STATS
        # main(feats=[('feat', 'stats'), ],
        #      headers=[('is_header', True)],
        #      before_projs=[('before_proj', False), ],
        #      # gses=[('is_gs', False)],
        #      ds=[('d_kjl', 5), ],
        #      train_sizes= [('train_size', v*1000) for v in list(range(1, 5+1, 1))],
        #      out_dir='speedup/out_train_sizes',
        #     )
    except Exception as e:
        traceback.print_exc()
        lg.error(e)

    # merge_res(in_dir = 'speedup/calumet_out', datasets= DATASETS,
    #           directions=[('direction', 'src_dst'), ],
    #           feats=[('feat', 'stats'), ],
    #           # headers=[('is_header', False)],
    #           models=MODELS,
    #           # gses=[('is_gs', True), ('is_gs', False)],
    #           before_projs=[('before_proj', False), ],
    #           ds=[('d_kjl', 5), ], )