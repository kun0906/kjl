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
# Authors: kun.bj@outlook.com
# License: XXX

import itertools
import os
import os.path as pth
import shutil
import traceback

from kjl.log import get_log

DATASETS = [
    ### Final datasets for the paper
    'UNB345_3',  # Combine UNB3, UNB3 and UNB5 attack data as attack data and only use UNB3's normal as normal data
    'CTU1',  # Two different abnormal data
    'MAWI1_2020',  # Two different normal data
    'MACCDC1',  # Two different normal data
    'SFRIG1_2020',  # Two different normal data
    'AECHO1_2020',  # Two different normal data
    'DWSHR_WSHR_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
]

ALL_MODELS = [
    ### Algorithm name
    "OCSVM(rbf)",
    "KJL-OCSVM(linear)",
    "Nystrom-OCSVM(linear)",

    # "GMM(full)", "GMM(diag)",

    "KJL-GMM(full)",   "KJL-GMM(diag)",
    "Nystrom-GMM(full)",   "Nystrom-GMM(diag)",
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
    "KJL-QS-GMM(full)", "KJL-QS-GMM(diag)",
    # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"

    "Nystrom-QS-GMM(full)", "Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"
    #
    # ################################################################################################################
    # # 4. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection and initialize GMM (set 'GMM_is_init_all'=True)
    "KJL-QS-init_GMM(full)",   "KJL-QS-init_GMM(diag)",
    "Nystrom-QS-init_GMM(full)",   "Nystrom-QS-init_GMM(diag)",

]

MODELS = [ "OCSVM(rbf)",
           "KJL-OCSVM(linear)",
           "Nystrom-OCSVM(linear)",
           "KJL-GMM(full)", "KJL-GMM(diag)",
           "Nystrom-GMM(full)", "Nystrom-GMM(diag)",
           "KJL-QS-init_GMM(full)", "KJL-QS-init_GMM(diag)",
           "Nystrom-QS-init_GMM(full)", "Nystrom-QS-init_GMM(diag)",
           ]

# create a customized log instance that can print the information.
lg = get_log(level='info')


def clean(in_dir, dataset_name="CTU1", model_name="OCSVM(rbf)", feat_set='iat_size', is_gs=True, start_time=None):
    """

    Parameters
    ----------
    dataset_name:
    model_name
    start_time

    Returns
    -------
        out_file: dump the result to the disk
    """

    # lg.info(f'***{dataset_name}, {model_name}, {feat_set}')
    ##############################################################################################################
    # 1. Initialization parameters
    # in_dir = 'speedup/out/kjl_serial_ind_32_threads-cProfile_perf_counter'

    if 'OCSVM' in model_name:
        GMM_covariance_type = 'None'
    elif 'diag' in model_name:
        GMM_covariance_type = 'diag'
    else:
        GMM_covariance_type = 'full'

    tmp_dir = pth.join(in_dir,'src_dst','stats-header_False')
    if 'stats' in tmp_dir:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    tmp_dir = pth.join(in_dir, 'src_dst', 'stats-header_True')
    if 'stats' in tmp_dir:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    in_dir = pth.join(in_dir,
                      'src_dst',
                      feat_set + "-header_False",
                      dataset_name,
                      "before_proj_False" + \
                      f"-gs_{is_gs}",
                      model_name + "-std_False"
                      + "_center_False" + "-d_5" \
                      + f"-{GMM_covariance_type}")

    if 'gs_False' in in_dir:
        if os.path.exists(in_dir):
            tmp_dir = os.path.dirname(in_dir)
            # tmp_dir = os.path.dirname(tmp_dir)
            lg.warning(f'remove: {tmp_dir}')
            shutil.rmtree(tmp_dir)
    # if 'diag' in in_dir or 'OCSVM(linear)' in in_dir:
    if model_name not in MODELS:
        if os.path.exists(in_dir):
            lg.warning(f'remove: {in_dir}')
            shutil.rmtree(in_dir)

    lg.info(f'***{dataset_name}, {model_name}, {feat_set}, {in_dir}')

    n_repeats = 5
    ##############################################################################################################
    # 2. Recreate a new model from the saved parameters and evaluate it on the test set.
    for i in range(n_repeats):
        try:
            # 2.1 load the model and project parameters from file
            for tmp_file in [f'repeat_{i}.model', f'Test_set-repeat_{i + 1}.dat', 'results.dat', 'res.dat.csv',
                             'res.dat']:
                tmp_file = pth.join(in_dir, tmp_file)
                if os.path.exists(tmp_file):
                    lg.warning(f'remove: {tmp_file}')
                    os.remove(tmp_file)
        except Exception as e:
            traceback.print_exc()
            lg.error(f"Error: {dataset_name}, {model_name}")

    ##############################################################################################################
    # 3. Save the result to the disk
    out_file = f'{in_dir}/{dataset_name}-{model_name}.dat'
    if os.path.exists(out_file): os.remove(out_file)
    return out_file


if __name__ == '__main__':
    in_dir_src = 'speedup/out/kjl_serial_ind_32_threads-cProfile_perf_counter-20times-4'
    in_dir = 'speedup/data/models'
    if os.path.exists(in_dir):
        # os.makedirs(in_dir)
        shutil.rmtree(in_dir)
    shutil.copytree(in_dir_src, in_dir)

    for dataset_name, model_name in itertools.product(DATASETS, ALL_MODELS):
        clean(in_dir, dataset_name=dataset_name, model_name=model_name, start_time=None, is_gs=True)
        clean(in_dir, dataset_name=dataset_name, model_name=model_name, start_time=None, is_gs=False)
