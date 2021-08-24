import itertools
import os

import shutil

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

MODELS = [
    ### Algorithm name
    # "OCSVM(rbf)",
    "KJL-OCSVM(linear)",
    "Nystrom-OCSVM(linear)",

    # "GMM(full)", "GMM(diag)",

    # "KJL-GMM(full)",  "KJL-GMM(diag)",
    # "Nystrom-GMM(full)",   "Nystrom-GMM(diag)",
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
    # "KJL-QS-GMM(full)", #"KJL-QS-GMM(diag)",
    # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"

    # "Nystrom-QS-GMM(full)", #"Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"
    #
    # ################################################################################################################
    # # 4. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection and initialize GMM (set 'GMM_is_init_all'=True)
    # "KJL-QS-init_GMM(full)",  "KJL-QS-init_GMM(diag)",
    # "Nystrom-QS-init_GMM(full)", "Nystrom-QS-init_GMM(diag)",

]


def main(src_dir, dst_dir, feat_set, is_gs, is_header, GMM_covariance_type):
    for data_name, model_name in itertools.product(DATASETS, MODELS):
        if 'OCSVM' in model_name: GMM_covariance_type = 'None'
        sub_dir = os.path.join(
                      'src_dst',
                      feat_set + f"-header_{is_header}",
                      data_name,
                      "before_proj_False" + \
                      f"-gs_{is_gs}",
                      model_name + "-std_False"
                      + "_center_False" + "-d_5" \
                      + f"-{GMM_covariance_type}")
        src_dir_tmp = os.path.join(src_dir, sub_dir)
        dst_dir_tmp = os.path.join(dst_dir, sub_dir)
        if not os.path.exists(dst_dir_tmp):
            shutil.copytree(src_dir_tmp, dst_dir_tmp)


src_dir =   'offline/out/kjl_serial_ind_32_threads-cProfile_perf_counter-20times-41'
dst_dir =   'offline/out/kjl_serial_ind_32_threads-cProfile_perf_counter-20times-4'

for feat_set in ['iat_size', 'stats']:  # 'iat_size',
    for is_gs in [True, False]:
        for is_header in [True, False]:
            for GMM_covariance_type in ['full', 'diag']:
                if feat_set == 'iat_size' and is_header: continue
                if feat_set == 'stats' and not is_header: continue
                main(src_dir, dst_dir, feat_set, is_gs, is_header, GMM_covariance_type)

