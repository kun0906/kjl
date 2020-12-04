"""Main entrance
    run under "examples"
    python3 -V
    PYTHONPATH=../:./ python3.8 speedup/main_kjl_parallel.py > speedup/out/main_kjl_parallel.txt 2>&1 &
"""

import itertools
import os
import os.path as pth

from kjl.utils.data import load_data, dump_data, dat2csv
from kjl.utils.tool import execute_time
from speedup._speedup_kjl import get_best_results
from speedup.generate_data import generate_data_speed_up
from speedup.ratio_variance import dat2xlxs, improvement, dat2latex

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


def _generate_datasets(overwrite=False):
    feat = 'iat_size'
    in_dir = f'speedup/data/{feat}'
    dataname_path_mappings = {
        # 'mimic_GMM': f'{in_dir}/mimic_GMM/Xy-normal-abnormal.dat',
        # 'mimic_GMM1': f'{in_dir}/mimic_GMM1/Xy-normal-abnormal.dat',
        # 'UNB1': f'{in_dir}/UNB1/Xy-normal-abnormal.dat',
        # 'UNB2': f'{in_dir}/UNB2/Xy-normal-abnormal.dat',
        #     # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
        #     # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
        #     # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
        #     # # # # #
        #     # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
        #     # # # # # #
        #     'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
        # 'CTU1': f'{in_dir}/CTU1/Xy-normal-abnormal.dat',
        # #     # # #
        # #     # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        # #     # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        'MAWI1': f'{in_dir}/MAWI1/Xy-normal-abnormal.dat',
        # #     # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
        # #     # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
        # #     # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
        # #     # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
        # #     # #
        # #     # # #
        # #     # 'WRCCDC/2020-03-20',
        # #     # 'DEFCON/ctf26',
        # 'ISTS1': f'{in_dir}/ISTS1/Xy-normal-abnormal.dat',
        # 'MACCDC1': f'{in_dir}/MACCDC1/Xy-normal-abnormal.dat',
        #
        # #     # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
        # 'SCAM1': f'{in_dir}/SCAM1/Xy-normal-abnormal.dat',
        # # # #     # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
        # # # #     # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
        # # # #

    }

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


def _generate_experiment(case='', is_gs=True, is_std=False, covariance_type='diag', **kwargs):
    TEMPLATE = {'detector': {'detector_name': 'GMM'},
                'is_gs': False,
                'std': {'is_std': is_std, 'is_std_mean': False},  # default use std
                'kjl': {'is_kjl': False},
                'nystrom': {'is_nystrom': False},
                'quickshift': {'is_quickshift': False},
                'meanshift': {'is_meanshift': False},
                'random_state': kwargs['random_state'],
                'n_repeats': kwargs['n_repeats'],
                # 'q_abnormal_thres': 0.9,
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
                             is_gs=is_gs,
                             # kjl={'is_kjl': True,   # ocsvm use 'linear'
                             #    'kjl_qs': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] if is_gs
                             #                           else [0.3], 'kjl_ns': [100], 'kjl_ds': [10]}
                             ),
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
                                  'kjl_qs': [] if is_gs
                                  else [], 'kjl_ns': [100], 'kjl_ds': [10]}),

        # case 4: GMM-gs:True-nystrom:True   # nystrom will take more time than kjl
        'case4': create_case(template=TEMPLATE,
                             detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                       'GMM_n_components': [1, 5, 10, 15, 20, 25, 30, 35, 40,
                                                            45] if is_gs else [1]},
                             is_gs=is_gs,
                             nystrom={'is_nystrom': True,
                                      'nystrom_qs': [] if is_gs
                                      else [], 'nystrom_ns': [100], 'nystrom_ds': [10]}),
        #
        # # case 5: GMM-gs:True-kjl:True-quickshift:True
        'case5': create_case(template=TEMPLATE,
                             detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                       'GMM_n_components': [] if is_gs else []},
                             is_gs=is_gs,
                             kjl={'is_kjl': True,
                                  'kjl_qs': [] if is_gs
                                  else [], 'kjl_ns': [100], 'kjl_ds': [10]},
                             quickshift={'is_quickshift': True, 'quickshift_ks': [100] if is_gs else [100],
                                         # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                                         'quickshift_betas': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                              0.95] if is_gs else [0.9]}),

        # case 6: GMM-gs:True-kjl:True-meanshift:True
        'case6': create_case(template=TEMPLATE,
                             detector={'detector_name': 'GMM', 'GMM_covariance_type': covariance_type,
                                       'GMM_n_components': [] if is_gs else []},
                             is_gs=is_gs,
                             kjl={'is_kjl': True,
                                  'kjl_qs': [] if is_gs
                                  else [], 'kjl_ns': [100], 'kjl_ds': [10]},
                             meanshift={'is_meanshift': True,
                                        'meanshift_qs': [] if is_gs
                                        else []})
    }

    return experiments[case]


@execute_time
def _main(is_gs=False, is_std=False, covariance_type='diag', n_jobs=16):
    results = {}
    n_repeats = 1
    random_state = 42
    overwrite = False
    verbose = 10
    datasets = _generate_datasets(overwrite=False)

    for dataset in datasets:
        # # 1. get best_qs for OCSVM
        # # # # # case 1: OCSVM-gs:True
        # case = 'case1'
        # experiment = _generate_experiment(case, is_gs=is_gs, is_std=is_std, covariance_type=covariance_type, n_jobs=n_jobs,
        #                                   n_repeats=n_repeats, random_state=random_state,
        #                                   verbose=verbose, overwrite=overwrite)
        # ocsvm_results = single_main(data=dataset, params=(case, experiment))
        # k, v = ocsvm_results[0], ocsvm_results[1]
        # results[k] = v

        # # 2. use the best_qs for GMM
        # # case 2: GMM-gs:True
        # case = 'case2'
        # experiment = _generate_experiment(case, is_gs=is_gs, is_std=is_std, covariance_type=covariance_type, n_jobs=n_jobs,
        #                                   n_repeats=n_repeats, random_state=random_state,
        #                                   verbose=verbose, overwrite=overwrite)
        # _new_results = single_main(data=dataset, params=(case, experiment))
        # k, v = _new_results[0], _new_results[1]
        # results[k] = v

        # case 3: GMM-gs:True-kjl:True
        case = 'case3'
        experiment = _generate_experiment(case, is_gs=is_gs,is_std=is_std, covariance_type=covariance_type, n_jobs=n_jobs,
                                          n_repeats=n_repeats, random_state=random_state,
                                          verbose=verbose, overwrite=overwrite)
        # (_, _), (_best_result, _) = ocsvm_results
        # experiment['kjl_qs'] = [_best_result['params']['OCSVM_q']]
        experiment['kjl_qs'] = [0.1]
        _new_results = single_main(data=dataset, params=(case, experiment))
        k, v = _new_results[0], _new_results[1]
        results[k] = v
        #
        # # case 4: GMM-gs:True-nystrom:True   # nystrom will take more time than kjl
        # case = 'case4'
        # experiment = _generate_experiment(case, is_gs=is_gs, is_std=is_std, covariance_type=covariance_type, n_jobs=n_jobs,
        #                                   n_repeats=n_repeats, random_state=random_state,
        #                                   verbose=verbose, overwrite=overwrite)
        # (_, _), (_best_result, _) = ocsvm_results
        # experiment['nystrom_qs'] = [_best_result['params']['OCSVM_q']]
        # _new_results = single_main(data=dataset, params=(case, experiment))
        # k, v = _new_results[0], _new_results[1]
        # results[k] = v
        #
        # # # case 5: GMM-gs:True-kjl:True-quickshift:True
        # case = 'case5'
        # experiment = _generate_experiment(case, is_gs=is_gs,is_std=is_std, covariance_type=covariance_type, n_jobs=n_jobs,
        #                                   n_repeats=n_repeats, random_state=random_state,
        #                                   verbose=verbose, overwrite=overwrite)
        # (_, _), (_best_result, _) = ocsvm_results
        # # experiment['quickshift_betas'] = [_best_result['params']['OCSVM_q']]
        # experiment['kjl_qs'] = [_best_result['params']['OCSVM_q']]
        # _new_results = single_main(data=dataset, params=(case, experiment))
        # k, v = _new_results[0], _new_results[1]
        # results[k] = v
        #
        # # case 6: GMM-gs:True-kjl:True-meanshift:True
        # case = 'case6'
        # experiment = _generate_experiment(case, is_gs=is_gs, is_std=is_std, covariance_type=covariance_type, n_jobs=n_jobs,
        #                                   n_repeats=n_repeats, random_state=random_state,
        #                                   verbose=verbose, overwrite=overwrite)
        # (_, _), (_best_result, _) = ocsvm_results
        # experiment['meanshift_qs'] = [_best_result['params']['OCSVM_q']]
        # experiment['kjl_qs'] = [_best_result['params']['OCSVM_q']]
        # _new_results = single_main(data=dataset, params=(case, experiment))
        # k, v = _new_results[0], _new_results[1]
        # results[k] = v

        # avoid missing results
        data_str, case = k
        data_name, data_file = data_str.split('|')
        out_file = (f'{os.path.dirname(data_file)}/gs_{is_gs}-{covariance_type}/{case}.dat')
        print(out_file)
        dump_data(results, out_file)
        is_std = experiment['is_std']
        is_std_mean = experiment['is_std_mean']

    print(f'\n\nfinal results: {results}')
    # save results first
    out_dir = f'speedup/out/iat_size-gs_{is_gs}-{covariance_type}-std_{is_std}_center_{is_std_mean}'
    out_file = results2speedup(results, out_dir)
    print(f'{out_file}')
    print("\n\n---finish succeeded!")


@execute_time
def main():
    gses = [False]
    covariance_types = ['diag']
    stds = [True, False]
    combs = len(list(itertools.product(gses, stds, covariance_types)))
    for i, (is_gs, is_std, covariance_type) in enumerate(itertools.product(gses, stds, covariance_types)):
        print(f'\n\n***{i + 1}/{combs}: is_gs:{is_gs}, is_std:{is_std}, covariance_type: {covariance_type}***')
        _main(is_gs=is_gs, is_std=is_std, covariance_type=covariance_type)


if __name__ == '__main__':
    main()
