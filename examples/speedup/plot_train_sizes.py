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

import copy
import os.path as pth
import traceback
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kjl.log import get_log
from kjl.utils.tool import dump_data, check_path

# get log

# sns.set_style("darkgrid")

# colorblind for diff
sns.set_palette("bright")  # for feature+header
sns.palplot(sns.color_palette())

# get log
lg = get_log(level='info')

DATASETS = [
    # # 'UNB3',
    # # 'UNB_5_8_Mon',  # auc: 0.5
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

    # 'UNB123_1',  # combine UNB1, UNB2, UNB3 attacks, only use UNB1 normal
    # 'UNB134_1',
    # 'UNB145_1',

    # 'UNB245_2',

    # 'UNB35_3',
    # 'UNB234_2',  # combine UNB2, UNB3, UNB4 attacks, only use UNB2 normal
    # 'UNB345_3',
    # # 'UNB35_3',
    # # 'UNB34_3',
    # # 'UNB35_3',
    #
    # # 'UNB24',
    # 'CTU1',
    # 'CTU21', # normal + abnormal (botnet) # normal 10.0.0.15
    # # 'CTU22',  # normal + abnormal (coinminer)
    # 'CTU31',  # normal + abnormal (botnet)   # 192.168.1.191
    # 'CTU32',  # normal + abnormal (coinminer)
    'MAWI1_2020',
    # 'MAWI32_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32',
    # 'MAWI32-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32-2',
    # 'MAWI165-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.7.165-2',  # ~25000 (flows src_dst)
    # 'ISTS1',
    # 'MACCDC1',
    # 'SFRIG1_2020',
    # 'AECHO1_2020',
]

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
    # "KJL-QS-GMM(full)", "KJL-QS-GMM(diag)",
    # # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"
    #
    # "Nystrom-QS-GMM(full)", "Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"

]


def show_train_sizes(in_file, out_file='', title='auc', n_repeats=5):
    font_size = 15
    values_dict = pd.read_csv(in_file).values
    if out_file == '':
        out_file = in_file + '.pdf'
    print(out_file)

    # show_detectors = []
    # sns.set_style("darkgrid")
    # # create plots

    # close preivous figs
    plt.close()

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    colors = ['blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green'][:2]
    labels = ['KJL', 'Nystrom']

    sub_dataset = []
    yerrs = []
    data_name = ''
    j = 0
    for i, vs in enumerate(values_dict):
        try:
            data_name = vs[0].split('|')[0]
            train_size = vs[2].split('-')[0].split('(')[-1]
            m_auc, std_auc = vs[4].split(':')[-1].split('+/-')
            vs = [data_name, vs[1], int(train_size), float(m_auc)]
            yerrs.append(float(std_auc) / np.sqrt(n_repeats))
        except Exception as e:
            print(e)
            # traceback.print_exc()
            vs = ['0', '0', 0, 0]
            yerrs.append(0)
        sub_dataset.append(vs)

    df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'train_size', 'auc'])
    # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
    print('yerrs:', yerrs)
    plt.errorbar(df['train_size'].values, df['auc'].values, yerrs, capsize=3, color=colors[j], ecolor='tab:red',
                 label=labels[j])
    plt.xticks(df['train_size'].values, fontsize=font_size - 1)
    plt.yticks(fontsize=font_size - 1)
    # fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
    plt.xlabel('Train size ($n$)', fontsize=font_size)
    plt.ylabel('AUC', fontsize=font_size)
    # plt.title(f'{data_name}', fontsize = font_size-2)
    plt.legend(labels=['OCSVM'], loc='upper right', fontsize=font_size - 2)
    plt.ylim([0.95, 1])
    # n_groups = len(show_datasets)
    # index = np.arange(n_groups)
    # # print(index)
    # # plt.xlim(xlim[0], n_groups)
    # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
    plt.tight_layout()
    #

    # try:
    #     if r == 1:
    #         plt.subplots_adjust(bottom=0.35)
    #     else:
    #         # if appendix:
    #         #     if "GMM" in ",".join(show_detectors):
    #         #         plt.subplots_adjust(bottom=0.2)
    #         #     else:
    #         #         plt.subplots_adjust(bottom=0.10)
    #         # else:
    #         plt.subplots_adjust(bottom=0.13, top=0.95)
    # except Warning as e:
    #     raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(out_file + '.pdf')  # should use before plt.show()
    plt.show()
    plt.close()

    # sns.reset_orig()
    # sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file


def merge_train_sizes_res(in_dir='', datasets=['SFRIG1_2020', 'AECHO1_2020'],
                          directions=[('direction', 'src_dst'), ],
                          feats=[('feat', 'stats'), ],
                          headers=[('is_header', True)],
                          models="",
                          gses=[('is_gs', False)],
                          before_projs=[('before_proj', False), ],
                          ds=[('d_kjl', 5)],
                          train_sizes=[('train_size', v * 1000) for v in list(range(1, 5 + 1, 1))],
                          covariances=[('covariance_type', 'None')]
                          ):
    # model_name="OCSVM(rbf)", data_name="UNB24",
    def _merge_datasets(datasets, pth_cfg):
        """ merge all dataset to one csv

        Parameters
        ----------
        datasets
        pth_cfg

        Returns
        -------

        """
        (feat, is_header, is_before_proj, is_gs, d, covariance_type, model_name, train_size) = pth_cfg
        data = []
        for data_name in datasets:
            data_file = pth.join(in_dir, feat + "-header_" + str(is_header),
                                 data_name,
                                 "before_proj_" + str(is_before_proj) + "-gs_" + str(is_gs),
                                 model_name + f"-std_False_center_False-d_{str(d)}-{str(covariance_type)}-train_size_{train_size}",
                                 'res.dat.csv')
            print(f'{data_file}: {pth.exists(data_file)}')
            if 'GMM' in data_file:
                if f'GMM({covariance_type})' not in data_file:
                    continue
            try:
                df = pd.read_csv(data_file, header=None)
                for i, v in enumerate(df.values):
                    v[1] = f"{v[1].strip()}|d_{d}"
                    data.extend([v])
            except Exception as e:
                data.extend(np.array(
                    [[f'{data_name}|{data_file}', f'{model_name}|d_{d}', 'X_train_shape', 'X_test_shape', feat]]))
                print(f'Error({e}): {data_file}')

        # try:
        #     # combine all files in the list
        #     combined_csv = pd.concat(data)
        #     # # export to csv
        #     # combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')
        #     data = combined_csv.values
        # except Exception as e:
        #     data = None

        return data

    def _merge_models(feat, is_header, is_before_proj, is_gs, d, covariance_type, dataset_name, model_name,
                      train_sizes):
        """ merge all models and datasets to one csv

        Parameters
        ----------
        feat
        is_header
        is_gs
        d

        Returns
        -------

        """
        res_ = {}
        vs = []  # store for csv
        for i, ts_tup in enumerate(train_sizes):
            _, ts = ts_tup
            if "OCSVM" in model_name:
                pth_cfg = (feat, is_header, is_before_proj, is_gs, d, None, model_name, ts)
                data = _merge_datasets([dataset_name], pth_cfg)
                res_[ts_tup] = data
            elif 'GMM' in model_name:
                pth_cfg = (feat, is_header, is_before_proj, is_gs, d, covariance_type, model_name, ts)
                data = _merge_datasets([dataset_name], pth_cfg)
                res_[ts_tup] = data
            else:
                msg = ts_tup
                raise NotImplementedError(msg)

            # store for csv
            if i == 0:
                vs = copy.deepcopy(data)
            else:
                vs.extend(data)
        # print(vs)
        #  'feat-header_false-before_proj_False-gs_True-diag-std_False_center_False-d_5'
        out_file_ = pth.join(in_dir, feat + "-header_" + str(is_header),
                             "before_proj_" + str(is_before_proj) + "-gs_" + str(is_gs),
                             f"std_False_center_False-{str(covariance_type)}",
                             f'{dataset_name}-{model_name}.csv')
        print(f'data_models: {out_file_}')
        check_path(out_file_)
        out_file_dat = out_file_ + '.dat'
        dump_data(res_, out_file=out_file_dat)
        # save as csv
        pd.DataFrame(vs).to_csv(out_file_, index=False, encoding='utf-8-sig')
        # # save as xlsx
        # out_xlsx = dat2xlxs_new(out_file_dat, out_file=out_file_dat + '.xlsx', models=models)
        # # compute ratio OCSVM/GMM
        # out_xlsx_ratio = improvement(out_xlsx, feat_set=feat,
        #                              out_file=os.path.splitext(out_file_dat)[0] + '-ratio.xlsx')
        # print(out_xlsx)
        #
        # # for paper
        # out_latex = dat2latex(out_xlsx_ratio, out_file=os.path.splitext(out_file_dat)[0] + '-latex.xlsx')
        # print(out_latex)

        # show(in_file=out_file_)   # show model separately
        return out_file_

    res = {}
    for direction_tup in directions:
        in_dir = pth.join(in_dir, direction_tup[1])
        for feat_tup in feats:
            for header_tup in headers:
                for before_proj_tup in before_projs:
                    for gs_tup in gses:
                        for covariance_type_tup in covariances:
                            for dataset_name in datasets:
                                try:
                                    in_files = []
                                    for model_name in [f'OCSVM(rbf)']:
                                        out_file = _merge_models(feat_tup[1], header_tup[1], before_proj_tup[1],
                                                                 gs_tup[1], ds[0][1],
                                                                 covariance_type_tup[1],
                                                                 dataset_name, model_name, train_sizes)
                                        show_train_sizes(out_file, out_file='')
                                        in_files.append(out_file)

                                except Exception as e:
                                    traceback.print_exc()
                                    lg.error(f"Error: {e}")


if __name__ == '__main__':
    in_dir = 'speedup/out_train_sizes'  # neon
    # in_dir = 'speedup/calumet_out_ds-20210201'    # calumet
    merge_train_sizes_res(in_dir=in_dir, datasets=DATASETS,
                          directions=[('direction', 'src_dst'), ],
                          feats=[('feat', 'iat_size'), ],
                          headers=[('is_header', False)],
                          models=MODELS,
                          gses=[('is_gs', False)],
                          before_projs=[('before_proj', False), ],
                          ds=[('d_kjl', 5)],
                          train_sizes=[('train_size', v * 1000) for v in list(range(1, 5 + 1, 1))],
                          )
