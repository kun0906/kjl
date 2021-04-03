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

        3) test memory
            run under "examples"
            PYTHONPATH=../:./ ~/.local/bin/scalene speedup/main_load_model-dev.py

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
import copy
import itertools
# Authors: kun.bj@outlook.com
# License: XXX
import os
import os.path as pth
import pickle
import time
import traceback

import numpy as np
from joblib import delayed, Parallel
# from scalene.scalene_profiler import Scalene
from sklearn import metrics
from sklearn.metrics import roc_curve

from kjl.log import get_log
from kjl.model.gmm import GMM
from kjl.model.kjl import KJL
from kjl.model.nystrom import NYSTROM
from kjl.model.ocsvm import OCSVM
from kjl.utils.data import _get_line, dump_data
from kjl.utils.tool import load_data, check_path

# create a customized log instance that can print the information.
from speedup.ratio_variance import dat2xlxs_new, improvement

lg = get_log(level='info')

DATASETS = [
    ### Final datasets for the paper
    'UNB345_3',  # Combine UNB3, UNB3 and UNB5 attack data as attack data and only use UNB3's normal as normal data
    'CTU1',  # Two different abnormal data
    'MAWI1_2020', # Two different normal data
    'MACCDC1',    # Two different normal data
    'SFRIG1_2020', #  Two different normal data
    'AECHO1_2020',  # Two different normal data
    'DWSHR_WSHR_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
]

MODELS = [
    ### Algorithm name
    "OCSVM(rbf)",
    "KJL-OCSVM(linear)",
    "Nystrom-OCSVM(linear)",

    # "GMM(full)", "GMM(diag)",

    "KJL-GMM(full)",  # "KJL-GMM(diag)",
    "Nystrom-GMM(full)",  # "Nystrom-GMM(diag)",
    #
    # ### quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    # # "QS-KJL-GMM(full)", "QS-KJL-GMM(diag)",
    # # "MS-KJL-GMM(full)", "MS-KJL-GMM(diag)",
    #
    # # "QS-Nystrom-GMM(full)", "QS-Nystrom-GMM(diag)",
    # # "MS-Nystrom-GMM(full)", "MS-Nystrom-GMM(diag)",
    #
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


# @profile(precision=8)
def _test(model, X_test, y_test, params, project):
    """Evaluate the model on the X_test, y_test

    Parameters
    ----------
    model
    X_test
    y_test

    Returns
    -------
       y_score: abnormal score
       testing_time, auc, apc
    """

    test_time = 0

    #####################################################################################################
    # 1. standardization
    # start = time.time()
    # # if self.params['is_std']:
    # #     X_test = self.scaler.transform(X_test)
    # # else:
    # #     pass
    # end = time.time()
    # self.std_test_time = end - start
    std_test_time = 0
    test_time += std_test_time

    #####################################################################################################
    # 2. projection
    start = time.time()
    if 'is_kjl' in params.keys() and params['is_kjl']:
        X_test = project.transform(X_test)
    elif 'is_nystrom' in params.keys() and params['is_nystrom']:
        X_test = project.transform(X_test)
    else:
        pass
    end = time.time()
    proj_test_time = end - start
    test_time += proj_test_time

    # no need to do seek in the testing phase
    seek_test_time = 0

    #####################################################################################################
    # 3. prediction
    start = time.time()
    # For inlier, a small value is used; a larger value is for outlier (positive)
    # it must be abnormal score because we use y=1 as abnormal and roc_acu(pos_label=1)
    y_score = model.decision_function(X_test)
    end = time.time()
    model_test_time = end - start
    test_time += model_test_time

    # For binary  y_true, y_score is supposed to be the score of the class with greater label.
    # auc = roc_auc_score(y_test, y_score)  # NORMAL(inliers): 0, ABNORMAL(outliers: positive): 1
    # pos_label = 1, so y_score should be the corresponding score (i.e., abnormal score)
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    lg.debug(f"AUC: {auc}")

    lg.info(f'Total test time: {test_time} <= std_test_time: {std_test_time}, '
            f'seek_test_time: {seek_test_time}, proj_test_time: {proj_test_time}, '
            f'model_test_time: {model_test_time}')

    return auc, test_time


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
        model_spaces = data['model_spaces']

        _prefix, _line, _suffex = _get_line(data, feat_set=feat_set)
        # line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs: {aucs} with best_params: {params}: {_suffex}'

        aucs_str = "-".join([str(v) for v in aucs])
        train_times_str = "-".join([str(v) for v in train_times])
        test_times_str = "-".join([str(v) for v in test_times])
        space_size_str = "-".join([str(v) for v in space_sizes])
        model_spaces_str = "-".join([str(v) for v in model_spaces])

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
               f'{n_comp_str2}, space_sizes: {space_size_str}|model_spaces: {model_spaces_str},' \
               f'tot_clusters:{tot_clusters_str}, {tot_clusters_str2},' \
               f'n_clusters: {n_clusters_str}, {n_clusters_str2}, with params: {params}: {_suffex}'

    except Exception as e:
        traceback.print_exc()
        line = f'{data_name}|, {model_name}, X_train, X_test, auc, train, test, => aucs:, ' \
               f'train_times:, test_times:, n_comp:, ' \
               f', space_sizes: |model_spaces: ,' \
               f'tot_clusters:, ,' \
               f'n_clusters: , , with params: : '
    f.write(line + '\n')
    return line


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
    outs = {}
    with open(out_file, 'w') as f:
        for data_name, values in result.items():
            for model_name, vs in values.items():
                line = _res2csv(f, vs, feat_set, data_name, model_name)

                line = np.asarray(line.split(','))
                if model_name not in outs.keys():
                    outs[model_name] = [line]
                else:
                    outs[model_name].append(line)
    lg.info(out_file)
    return out_file, outs


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


# @Scalene.profile(' --cpu-only    ')
# @profile(precision=8)
def minimal_model_cost(model_name, model_params_file, test_file, params, project_params_file, is_parallel=False):
    #######################################################################################################
    # 1. create a new model from saved parameters
    params = {'is_kjl': False, 'is_nystrom': False}
    if 'OCSVM' in model_name:
        #######################################################################################################
        # load params
        model_params = load_data(model_params_file)

        #######################################################################################################
        # create a new model
        # 'OCSVM(rbf)':
        oc = OCSVM()
        oc.kernel = model_params['kernel']
        oc._gamma = model_params['_gamma']  # only used for 'rbf', 'linear' doeesn't need '_gamma'
        oc.gamma = oc._gamma

        oc.support_vectors_ = model_params['support_vectors_']
        oc._dual_coef_ = model_params['_dual_coef_']  # Coefficients of the support vectors in the decision function.
        oc.dual_coef_ = oc._dual_coef_
        oc._intercept_ = model_params['_intercept_']
        oc.intercept_ = oc._intercept_

        oc.support_ = np.zeros(oc.support_vectors_.shape[0],
                               dtype=np.int32)  # np.empty((1,), dtype=np.int32) #  # model_params['support_']  # Indices of support vectors.
        oc._n_support = model_params['_n_support']  # Number of support vectors for each class.
        oc._sparse = model_params['_sparse']  # spare_kernel_compute or dense_kernel_compute
        oc.shape_fit_ = model_params['shape_fit_']  # to check if the dimension of train set and test set is the same.
        oc.probA_ = np.zeros(
            0)  # model_params['probA_']  # /* pairwise probability information */, not used. its values = [].
        oc.probB_ = np.zeros(
            0)  # model_params['probB_']  # /* pairwise probability information */, not used its values = [].
        oc.offset_ = -1 * model_params['_intercept_']  # model_params['offset_']

        project = None
        if 'KJL' in model_name:  # KJL-OCSVM
            # load params
            project_params = load_data(project_params_file)

            project = KJL(None)
            project.sigma = project_params['sigma']
            project.Xrow = project_params['Xrow']
            project.U = project_params['U']
            params['is_kjl'] = True
        elif 'Nystrom' in model_name:  # Nystrom-OCSVM
            # load params
            project_params = load_data(project_params_file)
            project = NYSTROM(None)
            project.sigma = project_params['sigma']
            project.Xrow = project_params['Xrow']
            project.eigvec_lambda = project_params['eigvec_lambda']
            params['is_nystrom'] = True

    elif 'GMM' in model_name:
        #######################################################################################################
        # load params
        model_params = load_data(model_params_file)

        # GMM params
        oc = GMM()
        oc.covariance_type = model_params['covariance_type']
        oc.weights_ = model_params['weights_']
        oc.means_ = model_params['means_']
        # oc.precisions_ = model_params['precisions_']
        oc.precisions_cholesky_ = model_params['precisions_cholesky_']

        project = None
        if 'KJL' in model_name:  # KJL-GMM
            # load params
            project_params = load_data(project_params_file)

            project = KJL(None)
            project.sigma = project_params['sigma']
            project.Xrow = project_params['Xrow']
            project.U = project_params['U']
            params['is_kjl'] = True
        elif 'Nystrom' in model_name:  # Nystrom-GMM
            # load params
            project_params = load_data(project_params_file)
            project = NYSTROM(None)
            project.sigma = project_params['sigma']
            project.Xrow = project_params['Xrow']
            project.eigvec_lambda = project_params['eigvec_lambda']
            params['is_nystrom'] = True
    else:
        raise NotImplementedError()

    #######################################################################################################
    # 2. load test set and evaluate the model
    X_test, y_test = load_data(test_file)

    # Evaluation
    # average time
    # minimal model cost
    num = 1
    if is_parallel:
        with Parallel(n_jobs=10, verbose=0, backend='loky', pre_dispatch=1, batch_size=1) as parallel:
            # outs = parallel(delayed(_test)(oc, X_test, y_test, params=params, project=project) for _ in range(num))
            outs = parallel(delayed(_test)(copy.deepcopy(oc), copy.deepcopy(X_test), copy.deepcopy(y_test),
                                           params=copy.deepcopy(params), project=copy.deepcopy(project)) for _ in
                            range(num))
        auc, test_time = list(zip(*outs))
        auc = np.mean(auc)
        test_time = np.mean(test_time)
    else:
        auc = []
        test_time = []
        for _ in range(num):
            # auc_, test_time_ = _test(oc, X_test, y_test, params, project)
            auc_, test_time_ = _test(copy.deepcopy(oc), copy.deepcopy(X_test), copy.deepcopy(y_test),
                                     params=copy.deepcopy(params), project=copy.deepcopy(project))
            auc.append(auc_)
            test_time.append(test_time_)
        auc = np.mean(auc)
        test_time = np.mean(test_time)

    return auc, X_test, test_time


def save_model_params(model, out_file):
    model_params = {}
    project_params = {}
    if 'OCSVM' in model.params['model_name']:
        # if model.params['model_name'] == 'OCSVM(rbf)':
        model_params['kernel'] = model.model.kernel
        if model_params['kernel'] == 'rbf':
            model_params['_gamma'] = model.model._gamma
        else:
            model_params['_gamma'] = 0

        model_params['support_vectors_'] = model.model.support_vectors_
        model_params['_dual_coef_'] = model.model._dual_coef_
        model_params['_intercept_'] = model.model._intercept_

        # other parameters
        model_params['_sparse'] = model.model._sparse
        model_params['shape_fit_'] = model.model.shape_fit_
        model_params['_n_support'] = model.model._n_support
        # model_params['support_'] = model.model.support_
        # model_params['probA_'] = model.model.probA_
        # model_params['probB_'] = model.model.probB_
        # model_params['offset_'] = model.model.offset_

        if 'KJL' in model.params['model_name']:  # KJL-OCSVM(linear)
            project_params['sigma'] = model.project.sigma
            project_params['Xrow'] = model.project.Xrow
            project_params['U'] = model.project.U
        elif 'Nystrom' in model.params['model_name']:  # Nystrom-OCSVM(linear)
            project_params['sigma'] = model.project.sigma
            project_params['Xrow'] = model.project.Xrow
            project_params['eigvec_lambda'] = model.project.eigvec_lambda

    elif 'GMM' in model.params['model_name']:

        # GMM params
        model_params['covariance_type'] = model.model.covariance_type
        model_params['weights_'] = model.model.weights_
        model_params['means_'] = model.model.means_
        # model_params['precisions_'] = model.model.precisions_
        model_params['precisions_cholesky_'] = model.model.precisions_cholesky_

        if 'KJL' in model.params['model_name']:  # KJL-GMM
            project_params['sigma'] = model.project.sigma
            project_params['Xrow'] = model.project.Xrow
            project_params['U'] = model.project.U
        elif 'Nystrom' in model.params['model_name']:  # Nystrom-GMM
            project_params['sigma'] = model.project.sigma
            project_params['Xrow'] = model.project.Xrow
            project_params['eigvec_lambda'] = model.project.eigvec_lambda


    else:
        raise NotImplementedError()

    # save model params to disk
    model_params_file = out_file + '.model_params'
    dump_data(model_params, model_params_file)

    # save projection params to disk
    project_params_file = out_file + '.project_params'
    dump_data(project_params, project_params_file)
    lg.info(project_params_file)

    return model_params_file, project_params_file


def get_model_space(model_params_file, project_params_file, unit='KB'):
    """ Return the size, in bytes, of path.

    Parameters
    ----------
    model_params_file
    project_params_file

    Returns
    -------

    """
    space = os.path.getsize(model_params_file) + os.path.getsize(project_params_file)
    if unit == 'KB':
        space /= 1e+3
    elif unit == 'MB':
        space /= 1e+6
    else:
        pass

    return space


# @profile(precision=8)
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
    lg.info(f'\n***{model_name}')

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
                      "-gs_True",
                      model_name + "-std_False"
                      + "_center_False" + "-d_5" \
                      + f"-{GMM_covariance_type}")
    lg.info(f'{in_dir}')
    n_repeats = 5
    train_times = []
    aucs = []
    test_times = []
    params = []
    space_sizes = []
    X_train_shape = ''
    X_val_shape = ''
    X_test_shape = ''
    model_spaces = []
    unit = 'KB'
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

            # # load test set from file
            test_set_file = pth.join(in_dir, f'Test_set-repeat_{i}.dat')
            # X_test, y_test = load_data(test_set_file)
            # X_test_shape = f'{X_test.shape}'
            #
            # #######################################################################################################
            # # average time
            # num = 10
            # auc =[]
            # test_time = []
            # for _ in range(num):
            #     auc_, test_time_ = model.test(X_test, y_test)
            #     auc.append(auc_)
            #     test_time.append(test_time_)
            # auc = np.mean(auc)
            # test_time = np.mean(test_time)
            # lg.info(f'auc: {auc}, test_time: {test_time}')

            model_name = model.params['model_name']
            model_params_file, project_params_file = save_model_params(model, out_file=model_file)
            model_spaces.append(get_model_space(model_params_file, project_params_file, unit=unit))
            auc, X_test, test_time = minimal_model_cost(model_name, model_params_file, test_set_file, None,
                                                project_params_file, is_parallel=False)
            X_test_shape = f'{X_test.shape}'
            lg.info(f'without_parallel: auc: {auc}, test_time: {test_time}')

            # auc, test_time = minimal_model_cost(model_name, model_params_file, test_set_file, None,
            #                                     project_params_file, is_parallel=True)
            # lg.info(f'with_parallel: auc: {auc}, test_time: {test_time}')

            aucs.append(auc)
            test_times.append(test_time)
        except Exception as e:
            traceback.print_exc()
            lg.error(f"Error: {data_name}, {model_name}")

    lg.info(f'model_spaces: {np.mean(model_spaces):.2f}+/-{np.std(model_spaces):.2f} ({unit})')
    lg.info(f'auc: {np.mean(aucs):.2f}+/-{np.std(aucs):.2f}')
    res = {'train_times': train_times, 'test_times': test_times, 'aucs': aucs,
           'params': params,
           'space_sizes': space_sizes, 'model_spaces': model_spaces,
           'X_train_shape': X_train_shape, 'X_val_shape': X_val_shape, 'X_test_shape': X_test_shape}

    return res


# @profile()
def main():
    res = {}
    feat_set = 'iat_size'
    # in_dir = 'speedup/out/kjl_serial_ind_32_threads-cProfile_perf_counter'
    in_dir = 'speedup/out/kjl_joblib_parallel_30'

    # Load models and evaluate them on test data
    for i, (data_name, model_name) in enumerate(itertools.product(DATASETS, MODELS)):
        out = _main(in_dir, data_name, model_name, feat_set)
        if data_name not in res.keys():
            res[data_name] = {model_name: out}
        else:
            res[data_name][model_name] = out
        # if model_name not in res.keys():
        #     res[model_name] = [out]
        # else:
        #     res[model_name].append(out)
    lg.debug(res)

    # Save results
    out_file = f'{in_dir}/res.csv'
    check_path(out_file)
    # save as csv
    _, outs = res2csv(res, out_file, feat_set=feat_set)
    out_file_dat = out_file + '.dat'
    dump_data(outs, out_file=out_file_dat)
    lg.info(out_file)
    # save as xlsx
    out_xlsx = dat2xlxs_new(out_file_dat, out_file=out_file_dat + '.xlsx', models=MODELS)
    # compute ratio OCSVM/GMM
    out_xlsx_ratio = improvement(out_xlsx, feat_set=feat_set,
                                 out_file=os.path.splitext(out_file_dat)[0] + '-ratio.xlsx')
    print(out_xlsx_ratio)

    lg.info('finish!')


if __name__ == '__main__':
    main()
