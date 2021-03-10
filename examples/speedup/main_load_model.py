""" Main function: load built models from disk and evaluate them on test data

    1. Instructions for executing the main file ('main_load_model.py')
        1) Change the current directory to "examples/"
            cd examples/
        2) Check python3 version
            python3 -V
        3) Execute the main file
            mkdir speedup/out
            PYTHONPATH=../:./ python3.7 speedup/main_load_model.py > speedup/out/main_load_model.txt 2>&1 &

    Here, we want to evaluate five models on three datasets.
        1) Five models (fitted on train data with best parameters):
            "OCSVM(rbf)", "KJL-GMM(full)", "Nystrom-GMM(full)",  "KJL-QS-GMM(full)", and "Nystrom-QS-GMM(full)"
        2) Three datasets:
            'UNB345_3', 'CTU1',  and 'AECHO1_2020'

    2. Python libraries
        python==3.7.3

        # Can be installed by pip3
        Cython==0.29.14    # for quickshift++
        matplotlib==3.1.1
        memory-profiler==0.57.0 #
        numpy==1.19.2
        openpyxl==3.0.2
        pandas==0.25.1
        scapy==2.4.4        # parse pcap
        scikit-learn==0.22.1
        scipy==1.4.1
        seaborn==0.11.1
        xlrd==1.2.0
        XlsxWriter==1.2.8
        func_timeout==4.3.5
        joblib==1.0.1

        # install instructions for the following libraries
        1) odet==0.0.1         # extract features
            i) cd "kjl/kjl/dataset/odet-master"
            ii) pip3.7 install .

        2) QuickshiftPP==1.0        # seek modes
            i) cd "kjl/kjl/model/quickshift/"
            iii) python3.7 setup.py build
            iv) python3.7 setup.py install

"""
# Authors: kun.bj@outlook.com
# License: XXX

import itertools
import os.path as pth
import pickle
import traceback

import numpy as np

from kjl.log import get_log
from kjl.utils.data import _get_line
from kjl.utils.tool import load_data

# create a customized log instance that can print the information.
lg = get_log(level='info')

DATASETS = [
    ### Final datasets for the paper
    # 'UNB345_3',  # Combine UNB3, UNB3 and UNB5 attack data as attack data and only use UNB3's normal as normal data
    # 'CTU1',  # Two different abnormal data
    # 'MAWI1_2020', # Two different normal data
    # 'MACCDC1',    # Two different normal data
    # 'SFRIG1_2020', #  Two different normal data
    'AECHO1_2020',  # Two different normal data
    # 'DWSHR_WSHR_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
]

MODELS = [
    ### Algorithm name
    "OCSVM(rbf)",
    # "KJL-OCSVM(linear)",
    # "Nystrom-OCSVM(linear)",

    # "GMM(full)", "GMM(diag)",

    # "KJL-GMM(full)",  # "KJL-GMM(diag)",
    # "Nystrom-GMM(full)",  # "Nystrom-GMM(diag)",
    #
    # ### quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    # # "QS-KJL-GMM(full)", "QS-KJL-GMM(diag)",
    # # "MS-KJL-GMM(full)", "MS-KJL-GMM(diag)",
    #
    # # "QS-Nystrom-GMM(full)", "QS-Nystrom-GMM(diag)",
    # # "MS-Nystrom-GMM(full)", "MS-Nystrom-GMM(diag)",
    #
    # ### quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    # "KJL-QS-GMM(full)",  # "KJL-QS-GMM(diag)",
    # # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"
    #
    # "Nystrom-QS-GMM(full)",  # "Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"

]


def _res2csv(f, data, feat_set='iat_size', data_name='', model_name=''):
    """ Format the data in one line and save it to the csv file (f)

    Parameters
    ----------
    f: file object
    data: dict
    feat_set
    data_name
    model_name

    Returns
    -------

    """
    try:
        # best_auc = data['best_auc']
        aucs = data['aucs']
        params = data['params'][-1]
        train_times = data['train_times']
        test_times = data['test_times']
        space_sizes = data['space_sizes']

        _prefix, _line, _suffex = _get_line(data, feat_set=feat_set)
        # line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs: {aucs} with best_params: {params}: {_suffex}'

        aucs_str = "-".join([str(v) for v in aucs])
        train_times_str = "-".join([str(v) for v in train_times])
        test_times_str = "-".join([str(v) for v in test_times])
        space_size_str = "-".join([str(v) for v in space_sizes])

        try:
            n_comps = [int(v['GMM_n_components']) for v in data['params']]
            mu_n_comp = np.mean(n_comps)
            std_n_comp = np.std(n_comps)
            n_comp_str = f'{mu_n_comp:.2f}+/-{std_n_comp:.2f}'
            n_comp_str2 = "-".join([str(v) for v in n_comps])
            tot_clusters = []
            n_clusters = []
            for ps in data['params']:
                qs_res = ps['qs_res']
                tot_clusters.append(len(qs_res['tot_clusters']))
                n_clusters.append(qs_res['n_clusters'])
            tot_clusters_str = f'{np.mean(tot_clusters):.2f}+/-{np.std(tot_clusters):.2f}'
            tot_clusters_str2 = "-".join([str(v) for v in tot_clusters])
            n_clusters_str = f'{np.mean(n_clusters):.2f}+/-{np.std(n_clusters):.2f}'
            n_clusters_str2 = "-".join([str(v) for v in n_clusters])
        except Exception as e:
            n_comps = []
            n_comp_str = f'-'
            n_comp_str2 = f'-'
            tot_clusters_str = 0
            tot_clusters_str2 = 0
            n_clusters_str = 0
            n_clusters_str2 = 0

        if 'qs_res' in params.keys():
            if 'tot_clusters' in params['qs_res'].keys():
                params['qs_res']['tot_clusters'] = '-'.join(
                    [f'{k}:{v}' for k, v in params['qs_res']['tot_clusters'].items()])
        line = f'{data_name}|, {model_name}, {_prefix}, {_line}, => aucs:{aucs_str}, ' \
               f'train_times:{train_times_str}, test_times:{test_times_str}, n_comp: {n_comp_str}, ' \
               f'{n_comp_str2}, space_sizes: {space_size_str},tot_clusters:{tot_clusters_str}, {tot_clusters_str2},' \
               f'n_clusters: {n_clusters_str}, {n_clusters_str2}, with params: {params}: {_suffex}'

    except Exception as e:
        traceback.print_exc()
        line = ''
    f.write(line + '\n')


def res2csv(result, out_file, feat_set='iat_size'):
    """ save results to a csv file

    Parameters
    ----------
    result: dict
        all results
    out_file: csv

    feat_set: str
        "iat_size"

    Returns
    -------
    out_file: csv

    """
    with open(out_file, 'w') as f:
        for data_name, values in result.items():
            for model_name, vs in values.items():
                _res2csv(f, vs, feat_set, data_name, model_name)

    return out_file


def load_model(model_file):
    """Load model from disk

    Parameters
    ----------
    model_file:

    Returns
    -------
        a model instance

    """
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        lg.debug(model)
    return model


def _main(in_dir, data_name, model_name, feat_set='iat_size'):
    """ Get the result by one model on one dataset

    Parameters
    ----------
    data_cfg
    model_cfg
    out_dir

    Returns
    -------

    """

    if 'OCSVM' in model_name:
        GMM_covariance_type = 'None'
    elif 'diag' in model_name:
        GMM_covariance_type = 'diag'
    else:
        GMM_covariance_type = 'full'

    in_dir = pth.join(in_dir,
                      'src_dst',
                      feat_set + "-header_False",
                      data_name,
                      "before_proj_False" + \
                      "-gs_False",
                      model_name + "-std_False"
                      + "_center_False" + "-d_5" \
                      + f"-{GMM_covariance_type}")
    n_repeats = 5
    train_times = []
    aucs = []
    test_times = []
    params = []
    space_sizes = []
    X_train_shape = ''
    X_val_shape = ''
    X_test_shape = ''
    for i in range(n_repeats):
        try:
            # load model from file
            model_file = pth.join(in_dir, f'repeat_{i}.model')
            model = load_model(model_file)
            train_times.append(model.train_time)
            params.append(model.params)
            space_sizes.append(model.space_size)
            X_train_shape = f'({model.N}, {model.D})'
            X_val_shape = ''

            # load test set from file
            test_set_file = pth.join(in_dir, f'Test_set-repeat_{i}.dat')
            X_test, y_test = load_data(test_set_file)
            X_test_shape = f'{X_test.shape}'

            auc, test_time = model.test(X_test, y_test)
            aucs.append(auc)
            test_times.append(test_time)
        except Exception as e:
            traceback.print_exc()
            lg.error(f"Error: {data_name}, {model_name}")

    res = {'train_times': train_times, 'test_times': test_times, 'aucs': aucs,
           'params': params,
           'space_sizes': space_sizes,
           'X_train_shape': X_train_shape, 'X_val_shape': X_val_shape, 'X_test_shape': X_test_shape}

    return res


def main():
    res = {}
    feat_set = 'iat_size'
    in_dir = 'speedup/out/kjl-dev'

    # Load models and evaluate them on test data
    for i, (data_name, model_name) in enumerate(itertools.product(DATASETS, MODELS)):
        out = _main(in_dir, data_name, model_name, feat_set)
        if data_name not in res.keys():
            res[data_name] = {model_name: out}
        else:
            res[data_name][model_name] = out
    lg.debug(res)

    # Save results
    res2csv(res, out_file=f'{in_dir}/res.csv', feat_set=feat_set)

    lg.info('finish!')


if __name__ == '__main__':
    main()
