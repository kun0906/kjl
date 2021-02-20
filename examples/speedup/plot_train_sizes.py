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
    # 'UNB35_3',
    #
    # # 'UNB24',
    # 'CTU1',
    # 'CTU21', # normal + abnormal (botnet) # normal 10.0.0.15
    # # 'CTU22',  # normal + abnormal (coinminer)
    # 'CTU31',  # normal + abnormal (botnet)   # 192.168.1.191
    # 'CTU32',  # normal + abnormal (coinminer)
    # 'MAWI1_2020',
    # 'MAWI32_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32',
    # 'MAWI32-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32-2',
    # 'MAWI165-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.7.165-2',  # ~25000 (flows src_dst)
    # 'ISTS1',
    # 'MACCDC1',
    'SFRIG1_2020',
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
    print(sub_dataset, 'yerrs:', yerrs)
    plt.errorbar(df['train_size'].values, df['auc'].values, yerrs, capsize=3, color=colors[j], ecolor='tab:red',
                 label=labels[j])
    plt.xticks(df['train_size'].values, fontsize=font_size - 1)
    plt.yticks(fontsize=font_size - 1)
    # fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
    plt.xlabel('Train size ($n$)', fontsize=font_size)
    plt.ylabel('AUC', fontsize=font_size)
    # plt.title(f'{data_name}', fontsize = font_size-2)
    plt.legend(labels=['OCSVM'], loc='upper right', fontsize=font_size - 2)
    plt.ylim([0.0, 1])
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
                                 model_name + f"-std_False_center_False-d_{str(d)}-{str(covariance_type)}-train_size_{train_size}-k_qs_None",
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



def show_3train_sizes(in_files, out_file='', title='auc', n_repeats=5):
    font_size = 15

    # show_detectors = []
    # sns.set_style("darkgrid")
    # # create plots

    # close preivous figs
    plt.close()

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    colors = ['blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']
    labels = ['KJL', 'Nystrom']

    fig = plt.figure()

    for i, ((model_name, k_qs), in_file) in enumerate(in_files.items()):
        values_dict = pd.read_csv(in_file).values

        if i == 0 and out_file == '':
            out_file = in_file + '.pdf'
            print(out_file)

        sub_dataset = []
        yerrs = []
        data_name = ''
        j = 0
        for _, vs in enumerate(values_dict):
            try:
                data_name = vs[0].split('|')[0]
                train_size = vs[2].split('-')[0].split('(')[-1]
                m_auc, std_auc = vs[4].split(':')[-1].split('+/-')
                vs = [data_name, vs[1], int(train_size), float(m_auc)]
                yerrs.append(float(std_auc))
            except Exception as e:
                print(e)
                # traceback.print_exc()
                vs = ['0', '0', 0, 0]
                yerrs.append(0)
            sub_dataset.append(vs)

        df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'train_size', 'auc'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print(f'{model_name}-{k_qs}', df['auc'].values, 'yerrs:', yerrs)
        label_ = f'{model_name}-{k_qs}' if 'KJL' in model_name else model_name
        plt.errorbar(df['train_size'].values, df['auc'].values, yerrs, capsize=3, color=colors[i], ecolor='tab:red',
                     label=label_, marker='o')
        # plt.legend(labels=[label_], loc='upper right', fontsize=font_size - 2)

    plt.legend(loc='lower right', fontsize=font_size - 2)
    plt.xticks(df['train_size'].values, fontsize=font_size - 1)
    plt.yticks(fontsize=font_size - 1)
    # fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
    plt.xlabel('Train size ($n$)', fontsize=font_size)
    plt.ylabel('AUC', fontsize=font_size)
    # plt.title(f'{data_name}', fontsize = font_size-2)

    plt.ylim([0.0, 1])
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



def show_3train_sizes_n_clusters(in_files, out_file='', title='auc', n_repeats=5, show_name='n_clusters'):
    font_size = 15

    # show_detectors = []
    # sns.set_style("darkgrid")
    # # create plots

    # close preivous figs
    plt.close()

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    colors = ['c', 'blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']

    fig = plt.figure()

    for i, ((model_name, k_qs), in_file) in enumerate(in_files.items()):
        if 'OCSVM' in model_name: continue

        values_dict = pd.read_csv(in_file).values

        if i == 0 and out_file == '':
            out_file = in_file + f'-{show_name}.pdf'
            print(out_file)

        sub_dataset = []
        yerrs = []
        data_name = ''
        j = 0
        for _, vs in enumerate(values_dict):
            try:
                data_name = vs[0].split('|')[0]
                train_size = vs[2].split('-')[0].split('(')[-1]
                # m_auc, std_auc = vs[4].split(':')[-1].split('+/-')    # auc
                if show_name == 'tot_n_clusters':
                    mu_, std_ = vs[13].split(':')[-1].split('+/-')      # tot_n_clusters
                else:
                    mu_, std_ = vs[15].split(':')[-1].split('+/-')  # n_clusters
                vs = [data_name, vs[1], int(train_size), float(mu_)]
                yerrs.append(float(std_))
            except Exception as e:
                print(e)
                # traceback.print_exc()
                vs = ['0', '0', 0, 0]
                yerrs.append(0)
            sub_dataset.append(vs)

        df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'train_size','n_clusters'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print(f'{model_name}-{k_qs}-{show_name}:',df['n_clusters'].values, 'yerrs:', yerrs)
        label_ = f'{model_name}-{k_qs}' if 'KJL' in model_name else model_name
        plt.errorbar(df['train_size'].values, df['n_clusters'].values, yerrs, capsize=3, color=colors[i], ecolor='tab:red',
                     label=label_, marker='o')
        # plt.legend(labels=[label_], loc='upper right', fontsize=font_size - 2)

    plt.legend(loc='upper right', fontsize=font_size - 2)
    plt.xticks(df['train_size'].values, fontsize=font_size - 1)
    plt.yticks(fontsize=font_size - 1)
    # fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
    plt.xlabel('Train size ($n$)', fontsize=font_size)
    plt.ylabel('Number of clusters', fontsize=font_size)
    if show_name =='tot_n_clusters':
        title='Total number of clusters'
    elif show_name =='n_clusters':
        title = 'Top 20 clusters'
    else:
        title =''
    plt.title(f'{title}', fontsize = font_size-2)

    # plt.ylim([0, 1])
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



def show_3train_sizes_cluster_size_histgram(in_files, out_file='', title='auc', n_repeats=5, show_name='n_clusters'):
    font_size = 15

    # show_detectors = []
    # sns.set_style("darkgrid")
    # # create plots

    # close preivous figs
    plt.close()

    show_detectors = []
    sns.set_style("darkgrid")
    # create plots
    num_figs = 3
    appendix = False
    if not appendix:
        c = 2  # cols of subplots in each row
        if num_figs > c:
            if num_figs % c == 0:
                r = int(num_figs // c)
            else:
                r = int(num_figs // c) + 1  # in each row, it show 4 subplot
            fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
            axes = axes.reshape(r, -1)
        else:
            r = 1
            fig, axes = plt.subplots(r, num_figs, figsize=(8, 5))  # (width, height)
            axes = np.asarray([axes]).reshape(1, 1)
    else:
        c = 1  # cols of subplots in each row
        if num_figs > c:
            if num_figs % c == 0:
                r = int(num_figs // c)
            else:
                r = int(num_figs // c)  # in each row, it show 4 subplot
            if "GMM" in ",".join(show_detectors):
                fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
            else:
                fig, axes = plt.subplots(r, c, figsize=(18, 18))  # (width, height)
            axes = axes.reshape(r, -1)
        else:
            r = 1
            fig, axes = plt.subplots(r, num_figs, figsize=(8, 5))  # (width, height)
            axes = np.asarray([axes]).reshape(1, 1)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0


    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    colors = ['c', 'blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']

    # fig = plt.figure()

    # for i, ((model_name, k_qs), in_file) in enumerate(in_files.items()):
    #     if 'OCSVM' in model_name: continue
    #
    #     values_dict = pd.read_csv(in_file).values
    #
    #     if i == 0 and out_file == '':
    #         out_file = in_file + f'-{show_name}-histgram.pdf'
    #         print(out_file)
    #
    #     sub_dataset = []
    #     yerrs = []
    #     data_name = ''
    #     j = 0
    #     for _, vs in enumerate(values_dict):
    #         try:
    #             data_name = vs[0].split('|')[0]
    #             train_size = vs[2].split('-')[0].split('(')[-1]
    #             # m_auc, std_auc = vs[4].split(':')[-1].split('+/-')    # auc
    #
    #             v1= vs[42].split('\'tot_n_clusters\':')[-1].split('-')  # tot_n_clusters
    #             v1 = [ (v_.split(':')[0].replace('\'', ''), int(v_.split(':')[-1].replace('\'', ''))) for v_ in v1][:20]    # top 20 cluster sizes
    #             vs = [data_name, vs[1], int(train_size), v1]
    #             yerrs.append(0)
    #         except Exception as e:
    #             print(e)
    #             # traceback.print_exc()
    #             vs = ['0', '0', 0, 0]
    #             yerrs.append(0)
    #         sub_dataset.append(vs)
    #     data = sub_dataset[-1][-1]  # only show the result on 5000 train size
    #     print(f'cluster size: {data}')
    #     # x, y = list(zip(*data))
    #     new_data = []
    #     for i, (x_, y_) in enumerate(data):
    #         new_data.extend([f'c_{i}'] * y_)
    #     # the histogram of the data
    #     n, bins, patches = plt.hist(new_data, 50, density=True, stacked=True, facecolor='g', alpha=0.75)
    #
    #     df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'train_size','n_clusters'])
    #     # # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
    #     # print(f'{model_name}-{k_qs}-{show_name}:',df['n_clusters'].values, 'yerrs:', yerrs)
    #     # label_ = f'{model_name}-{k_qs}' if 'KJL' in model_name else model_name
    #     # plt.errorbar(df['train_size'].values, df['n_clusters'].values, yerrs, capsize=3, color=colors[i], ecolor='tab:red',
    #     #              label=label_, marker='o')
    #     # plt.legend(labels=[label_], loc='upper right', fontsize=font_size - 2)


    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    # metrics = ['5/7', '2/3', '3/4']
    # metrics = ['3/4']
    if ('OCSVM(rbf)', None)  in in_files.keys():
        del in_files[('OCSVM(rbf)', None) ]

    j = 0
    for i, ((model_name, k_qs), in_file) in enumerate(in_files.items()):
        values_dict = pd.read_csv(in_file).values

        if i == 0 and out_file == '':
            out_file = in_file + f'-{show_name}-histgram.pdf'
            print(out_file)

        sub_dataset = []
        yerrs = []
        data_name = ''

        for _, vs in enumerate(values_dict):
            try:
                data_name = vs[0].split('|')[0]
                train_size = vs[2].split('-')[0].split('(')[-1]
                # m_auc, std_auc = vs[4].split(':')[-1].split('+/-')    # auc

                v1 = vs[42].split('\'tot_n_clusters\':')[-1].split('-')  # tot_n_clusters
                v1 = [(v_.split(':')[0].replace('\'', ''), int(v_.split(':')[-1].replace('\'', ''))) for v_ in v1][
                     :20]  # top 20 cluster sizes
                vs = [data_name, vs[1], int(train_size), v1]
                yerrs.append(0)
            except Exception as e:
                print(e)
                # traceback.print_exc()
                vs = ['0', '0', 0, 0]
                yerrs.append(0)
            sub_dataset.append(vs)
        data = sub_dataset[-1][-1]  # only show the result on 5000 train size
        print(f'train_size=5000-{k_qs}, Top 20 clusters ({len(data)}): {data}')
        # _, X = list(zip(*data))
        # y = X
        X = []
        y = []
        top_20_n_clusters = len(data)
        for i_, (x_, y_) in enumerate(data):
            X.extend([i] * y_)
            y.extend([f'c{i_}'] * y_)

        sub_dataset = list(zip(X, y))
        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        # print(f'{metric_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['x', 'y'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.histplot(data=df, x="y", hue="y", ax=axes[t, i % c], stat='count', legend=f'{model_name}-{k_qs}') # 'density', probability


        # g = sns.histplot(data=df, x="x", hue="y", ax=axes[t, i % c], stat='count',
        #                  legend=f'{model_name}-{k_qs}')  # 'density', probability
        # # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, ci=None,
        # #                 capsize=.2, ax=axes)
        # ys = []
        # xs = []
        # width = 0
        # sub_fig_width = 0
        # for i_p, p in enumerate(g.patches):
        #     height = p.get_height()
        #     # g.text(p.get_x() + p.get_width() / 2.,
        #     #        height,
        #     #        '{:0.3f}'.format(new_yerrs[i_p]),
        #     #        ha="center")
        #     width = p.get_width()
        #     ys.append(height)
        #     xs.append(p.get_x())
        #     # yerr.append(i_p + p.get_height())
        #
        #     num_bars = df['model_name'].nunique()
        #     # print(f'num_bars:',num_bars)
        #     if i_p == 0:
        #         pre = p.get_x() + p.get_width() * num_bars
        #         sub_fig_width = p.get_bbox().width
        #     if i_p < df['dataset'].nunique() and i_p > 0:
        #         cur = p.get_x()
        #         # g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, alpha=0.3)
        #         g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
        #         pre = cur + p.get_width() * num_bars
        #
        # axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
        #                         yerr=yerrs, fmt='none', c='b', capsize=3)

        s = 0
        pre_s = 0
        for p in g.patches:
            height = p.get_height()
            if height == 0: continue
            # # print(height)
            # if  height < 0:
            #     height= height - 0.03
            # else:
            #     height = p.get_height() + 0.03
            # # g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., height), ha='center',
            #                va='center', xytext=xytext, textcoords='offset points')
            s += height
            g.annotate(format(p.get_height(), '.0f'), xy=(p.get_x() + p.get_width() / 2., height), xytext=(0, 10),
                       ha='center', va='center',
                       textcoords="offset points", fontsize=font_size + 0)
            n = int(train_size)
            if (n-s)/n < 0.9 and (n-(s-height))/n >=0.9:
                l = g.axvline(x=p.get_x()+p.get_width(), ymin=0, ymax=0.9, color = 'b')
                g.text(p.get_x() + p.get_width(), g.get_ylim()[1]*0.8, '(>=90%)', fontsize=font_size)
            if (n-s)/n < 0.95 and (n-(s-height))/n >=0.95:
                l = g.axvline(x=p.get_x()+p.get_width(), ymin=0, ymax=0.9, color = 'c')
                g.text(p.get_x()+p.get_width(), g.get_ylim()[1]*0.7, '(>=95%)', fontsize=font_size)



        # g.set(xlabel=detector_name)       # set name at the bottom
        # g.set(xlabel=None)
        # g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        # g.set_ylabel(diff_name, fontsize=font_size + 4)
        g.set_xlabel('Cluster index', fontsize=font_size + 4)
        g.set_ylabel('Cluster size', fontsize=font_size + 4)
        # if appendix:
        #     if j < len(show_detectors) - 1:
        #         g.set_xticklabels([])
        #     else:
        #         g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # else:
        fig.canvas.draw()   # https://stackoverflow.com/questions/41122923/getting-empty-tick-labels-before-showing-a-plot-in-matplotlib
        g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 1, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        # y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        # y_v = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # g.set_yticks(y_v)  # set value locations in y axis
        # g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
        #                   fontsize=font_size + 6)  # set the number of each value in y axis
        g.set_yticklabels([int(v) for v in g.get_yticks()], fontsize=font_size + 4)  # set the number of each value in y axis
        # print(g.get_yticks(), y_v)
        # if j % c != 0:
        #     # g.get_yaxis().set_visible(False)
        #     g.set_yticklabels(['' for v_tmp in y_v])
        #     g.set_ylabel('')

        g.set_title(f'{model_name}-{k_qs}-Top 20 cluster sizes-({top_20_n_clusters})', fontsize=font_size + 4)
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        g.get_legend().set_visible(False)
        # g.get_legend().set_visible(True)
        # handles, labels = g.get_legend_handles_labels()
        # axes[t, j % c].legend(handles, labels, loc="upper right",fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        j+=1
    # # get the legend only from the last 'ax' (here is 'g')
    # handles, labels = g.get_legend_handles_labels()
    # labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    # pos1 = axes[-1, -1].get_position()  # get the original position
    # # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # # ax.set_position(pos2) # set a new position
    # loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    # print(f'loc: {loc}, pos1: {pos1.bounds}')
    # # axes[-1, -1].legend(handles, labels, loc=2, # upper right
    # #             ncol=1, prop={'size': font_size-13})  # loc='lower right',  loc = (0.74, 0.13)
    # axes[-1, -1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.0, 0.95, 1, 0.5),borderaxespad=0, fancybox=True, # upper right
    #                     ncol=1, prop={'size': font_size - 4})  # loc='lower right',  loc = (0.74, 0.13)

    # share one legend
    # fig.legend(handles, labels, loc='lower center',  # upper left
    #            ncol=3, prop={'size': font_size - 2})  # l

    # figs.legend(handles, labels, title='Representation', bbox_to_anchor=(2, 1), loc='upper right', ncol=1)
    # remove subplot
    # fig.delaxes(axes[1][2])
    # axes[-1, -1].set_axis_off()

    # j += 1
    # while t < r:
    #     if j % c == 0:
    #         t += 1
    #         if t >= r:
    #             break
    #         j = 0
    #     # remove subplot
    #     # fig.delaxes(axes[1][2])
    #     axes[t, j % c].set_axis_off()
    #     j += 1

    # # plt.xlabel('Catagory')
    # plt.ylabel('AUC')
    # plt.ylim(0,1.07)
    # # # plt.title('F1 Scores by category')
    # n_groups = len(show_datasets)
    # index = np.arange(n_groups)
    # # print(index)
    # # plt.xlim(xlim[0], n_groups)
    # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
    plt.tight_layout()
    #

    try:
        if r == 1:
            plt.subplots_adjust(bottom=0.35)
        else:
            if appendix:
                if "GMM" in ",".join(show_detectors):
                    plt.subplots_adjust(bottom=0.2)
                else:
                    plt.subplots_adjust(bottom=0.10)
            else:
                plt.subplots_adjust(bottom=0.13, top=0.95)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    print(f'--{out_file}')
    plt.savefig(out_file + '.pdf')  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    # sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file



def show_3train_sizes_cluster_size_histgram2(in_files, out_file='', title='auc', n_repeats=5, show_name='n_clusters'):
    font_size = 15

    # show_detectors = []
    # sns.set_style("darkgrid")
    # # create plots

    # close preivous figs
    plt.close()

    show_detectors = []
    sns.set_style("darkgrid")
    # create plots
    num_figs = 3
    appendix = False
    if not appendix:
        c = 2  # cols of subplots in each row
        if num_figs > c:
            if num_figs % c == 0:
                r = int(num_figs // c)
            else:
                r = int(num_figs // c) + 1  # in each row, it show 4 subplot
            fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
            axes = axes.reshape(r, -1)
        else:
            r = 1
            fig, axes = plt.subplots(r, num_figs, figsize=(8, 5))  # (width, height)
            axes = np.asarray([axes]).reshape(1, 1)
    else:
        c = 1  # cols of subplots in each row
        if num_figs > c:
            if num_figs % c == 0:
                r = int(num_figs // c)
            else:
                r = int(num_figs // c)  # in each row, it show 4 subplot
            if "GMM" in ",".join(show_detectors):
                fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
            else:
                fig, axes = plt.subplots(r, c, figsize=(18, 18))  # (width, height)
            axes = axes.reshape(r, -1)
        else:
            r = 1
            fig, axes = plt.subplots(r, num_figs, figsize=(8, 5))  # (width, height)
            axes = np.asarray([axes]).reshape(1, 1)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0


    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    colors = ['c', 'blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']

    # fig = plt.figure()

    # for i, ((model_name, k_qs), in_file) in enumerate(in_files.items()):
    #     if 'OCSVM' in model_name: continue
    #
    #     values_dict = pd.read_csv(in_file).values
    #
    #     if i == 0 and out_file == '':
    #         out_file = in_file + f'-{show_name}-histgram.pdf'
    #         print(out_file)
    #
    #     sub_dataset = []
    #     yerrs = []
    #     data_name = ''
    #     j = 0
    #     for _, vs in enumerate(values_dict):
    #         try:
    #             data_name = vs[0].split('|')[0]
    #             train_size = vs[2].split('-')[0].split('(')[-1]
    #             # m_auc, std_auc = vs[4].split(':')[-1].split('+/-')    # auc
    #
    #             v1= vs[42].split('\'tot_n_clusters\':')[-1].split('-')  # tot_n_clusters
    #             v1 = [ (v_.split(':')[0].replace('\'', ''), int(v_.split(':')[-1].replace('\'', ''))) for v_ in v1][:20]    # top 20 cluster sizes
    #             vs = [data_name, vs[1], int(train_size), v1]
    #             yerrs.append(0)
    #         except Exception as e:
    #             print(e)
    #             # traceback.print_exc()
    #             vs = ['0', '0', 0, 0]
    #             yerrs.append(0)
    #         sub_dataset.append(vs)
    #     data = sub_dataset[-1][-1]  # only show the result on 5000 train size
    #     print(f'cluster size: {data}')
    #     # x, y = list(zip(*data))
    #     new_data = []
    #     for i, (x_, y_) in enumerate(data):
    #         new_data.extend([f'c_{i}'] * y_)
    #     # the histogram of the data
    #     n, bins, patches = plt.hist(new_data, 50, density=True, stacked=True, facecolor='g', alpha=0.75)
    #
    #     df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'train_size','n_clusters'])
    #     # # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
    #     # print(f'{model_name}-{k_qs}-{show_name}:',df['n_clusters'].values, 'yerrs:', yerrs)
    #     # label_ = f'{model_name}-{k_qs}' if 'KJL' in model_name else model_name
    #     # plt.errorbar(df['train_size'].values, df['n_clusters'].values, yerrs, capsize=3, color=colors[i], ecolor='tab:red',
    #     #              label=label_, marker='o')
    #     # plt.legend(labels=[label_], loc='upper right', fontsize=font_size - 2)


    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    # metrics = ['5/7', '2/3', '3/4']
    # metrics = ['3/4']
    if ('OCSVM(rbf)', None)  in in_files.keys():
        del in_files[('OCSVM(rbf)', None) ]

    j = 0
    for i, ((model_name, k_qs), in_file) in enumerate(in_files.items()):
        values_dict = pd.read_csv(in_file).values

        if i == 0 and out_file == '':
            out_file = in_file + f'-{show_name}-histgram2.pdf'
            print(out_file)

        sub_dataset = []
        yerrs = []
        data_name = ''

        for _, vs in enumerate(values_dict):
            try:
                data_name = vs[0].split('|')[0]
                train_size = vs[2].split('-')[0].split('(')[-1]
                # m_auc, std_auc = vs[4].split(':')[-1].split('+/-')    # auc

                v1 = vs[42].split('\'tot_n_clusters\':')[-1].split('-')  # tot_n_clusters
                v1 = [(v_.split(':')[0].replace('\'', ''), int(v_.split(':')[-1].replace('\'', ''))) for v_ in v1][
                     :20]  # top 20 cluster sizes
                vs = [data_name, vs[1], int(train_size), v1]
                yerrs.append(0)
            except Exception as e:
                print(e)
                # traceback.print_exc()
                vs = ['0', '0', 0, 0]
                yerrs.append(0)
            sub_dataset.append(vs)
        data = sub_dataset[-1][-1]  # only show the result on 5000 train size
        print(f'train_size=5000-{k_qs}, Top 20 clusters ({len(data)}): {data}')
        _, X = list(zip(*data))
        y = X
        print(X)
        # X = []
        # y = []
        top_20_n_clusters = len(data)
        # for i_, (x_, y_) in enumerate(data):
        #     X.extend([f'c_{i_}'] * y_)
        #     y.extend([f'c_{i_}'] * y_)

        sub_dataset = list(zip(X, y))
        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        # print(f'{metric_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['x', 'y'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        # g = sns.histplot(data=df, x="x", hue="y", ax=axes[t, i % c], stat='probability', legend=f'{model_name}-{k_qs}') # 'density', probability


        # g = sns.histplot(data=df, x="x", hue=None, ax=axes[t, i % c], stat='count', bins= sorted(set(X)),
        #                  legend=f'{model_name}-{k_qs}')  # 'density', probability

        bins = [0, 5, 10, 50, 100, 500, 2000, max(X) if max(X) > 3000 else 3000]

        hist, bin_edges = np.histogram(X, bins)
        data = [[v1, v2] for v1, v2 in zip(hist, bin_edges)]
        df = pd.DataFrame(data, columns=['height', 'x-interval'])
        g = sns.barplot(y='height', x='x-interval', hue=None, data=df, ax=axes[t, i % c])
        pre_v = bin_edges[0]
        labels = []
        for v in bin_edges[1:]:
            labels.append(f'{pre_v}-{v}')
            pre_v = v +1
        g.set_xticklabels(labels,fontsize=font_size + 4, rotation = 30)


        # # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, ci=None,
        # #                 capsize=.2, ax=axes)
        # ys = []
        # xs = []
        # width = 0
        # sub_fig_width = 0
        # for i_p, p in enumerate(g.patches):
        #     height = p.get_height()
        #     # g.text(p.get_x() + p.get_width() / 2.,
        #     #        height,
        #     #        '{:0.3f}'.format(new_yerrs[i_p]),
        #     #        ha="center")
        #     width = p.get_width()
        #     ys.append(height)
        #     xs.append(p.get_x())
        #     # yerr.append(i_p + p.get_height())
        #
        #     num_bars = df['model_name'].nunique()
        #     # print(f'num_bars:',num_bars)
        #     if i_p == 0:
        #         pre = p.get_x() + p.get_width() * num_bars
        #         sub_fig_width = p.get_bbox().width
        #     if i_p < df['dataset'].nunique() and i_p > 0:
        #         cur = p.get_x()
        #         # g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, alpha=0.3)
        #         g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
        #         pre = cur + p.get_width() * num_bars
        #
        # axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
        #                         yerr=yerrs, fmt='none', c='b', capsize=3)

        for p in g.patches:
            height = p.get_height()
            if height == 0: continue
            # # print(height)
            # if  height < 0:
            #     height= height - 0.03
            # else:
            #     height = p.get_height() + 0.03
            # # g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., height), ha='center',
            #                va='center', xytext=xytext, textcoords='offset points')
            g.annotate(format(p.get_height(), '.0f'), xy=(p.get_x() + p.get_width() / 2., height),  xytext=(0,5), ha='center',va='center',
                       textcoords="offset points", fontsize = font_size+4)

        # g.set(xlabel=detector_name)       # set name at the bottom
        # g.set(xlabel=None)
        # g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        # g.set_ylabel(diff_name, fontsize=font_size + 4)
        g.set_xlabel('Cluster size', fontsize=font_size + 4)
        g.set_ylabel('Number of clusters', fontsize=font_size + 4)
        # if appendix:
        #     if j < len(show_detectors) - 1:
        #         g.set_xticklabels([])
        #     else:
        #         g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # else:
        # g.set_xticklabels(g.get_xticks(), fontsize=font_size + 4, rotation=90, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_xticks(sorted(set(X)))
        # print(g.get_xticks())
        # g.set_xticklabels(g.get_xticks(), fontsize=font_size + 4, rotation = 30)
        # g.set_xlim(.78, 1.)  # outliers only
        # g.set_xlim(0, .22)  # most of the data

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        # y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        # y_v = [0, 0.5, 1.0]
        # g.set_yticks(y_v)  # set value locations in y axis
        # g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
        #                   fontsize=font_size + 6)  # set the number of each value in y axis
        g.set_yticklabels([int(v) for v in g.get_yticks()], fontsize=font_size + 4)  # set the number of each value in y axis
        # print(g.get_yticks(), y_v)
        # if j % c != 0:
        #     # g.get_yaxis().set_visible(False)
        #     g.set_yticklabels(['' for v_tmp in y_v])
        #     g.set_ylabel('')

        g.set_title(f'{model_name}-{k_qs}-Top 20 cluster sizes-({top_20_n_clusters})', fontsize=font_size + 4)
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        # g.get_legend().set_visible(False)
        # g.get_legend().set_visible(True)
        # handles, labels = g.get_legend_handles_labels()
        # axes[t, j % c].legend(handles, labels, loc="upper right",fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        j+=1
    # # get the legend only from the last 'ax' (here is 'g')
    # handles, labels = g.get_legend_handles_labels()
    # labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    # pos1 = axes[-1, -1].get_position()  # get the original position
    # # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # # ax.set_position(pos2) # set a new position
    # loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    # print(f'loc: {loc}, pos1: {pos1.bounds}')
    # # axes[-1, -1].legend(handles, labels, loc=2, # upper right
    # #             ncol=1, prop={'size': font_size-13})  # loc='lower right',  loc = (0.74, 0.13)
    # axes[-1, -1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.0, 0.95, 1, 0.5),borderaxespad=0, fancybox=True, # upper right
    #                     ncol=1, prop={'size': font_size - 4})  # loc='lower right',  loc = (0.74, 0.13)

    # share one legend
    # fig.legend(handles, labels, loc='lower center',  # upper left
    #            ncol=3, prop={'size': font_size - 2})  # l

    # figs.legend(handles, labels, title='Representation', bbox_to_anchor=(2, 1), loc='upper right', ncol=1)
    # remove subplot
    # fig.delaxes(axes[1][2])
    # axes[-1, -1].set_axis_off()

    # j += 1
    # while t < r:
    #     if j % c == 0:
    #         t += 1
    #         if t >= r:
    #             break
    #         j = 0
    #     # remove subplot
    #     # fig.delaxes(axes[1][2])
    #     axes[t, j % c].set_axis_off()
    #     j += 1

    # # plt.xlabel('Catagory')
    # plt.ylabel('AUC')
    # plt.ylim(0,1.07)
    # # # plt.title('F1 Scores by category')
    # n_groups = len(show_datasets)
    # index = np.arange(n_groups)
    # # print(index)
    # # plt.xlim(xlim[0], n_groups)
    # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
    plt.tight_layout()
    #

    try:
        if r == 1:
            plt.subplots_adjust(bottom=0.35)
        else:
            if appendix:
                if "GMM" in ",".join(show_detectors):
                    plt.subplots_adjust(bottom=0.2)
                else:
                    plt.subplots_adjust(bottom=0.10)
            else:
                plt.subplots_adjust(bottom=0.13, top=0.95)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    print(f'--{out_file}')
    plt.savefig(out_file + '.pdf')  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    # sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file

def merge_train_sizes_3res(in_dir='', datasets=['SFRIG1_2020', 'AECHO1_2020'],
                          directions=[('direction', 'src_dst'), ],
                          feats=[('feat', 'stats'), ],
                          headers=[('is_header', True)],
                          models="",
                          gses=[('is_gs', False)],
                          before_projs=[('before_proj', False), ],
                          ds=[('d_kjl', 5)],
                          train_sizes=[('train_size', v * 1000) for v in list(range(1, 5 + 1, 1))],
                           k_qs=[('k_qs', '5/7'), ('k_qs', '2/3')],
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
        (feat, is_header, is_before_proj, is_gs, d, covariance_type, (model_name, k_qs), train_size) = pth_cfg
        data = []
        for data_name in datasets:
            if '5/7' in str(k_qs):
                k_qs_str = f'{int(train_size ** (5 / 7))}-' + k_qs.replace('/', '_')
            elif '2/3' in str(k_qs):
                k_qs_str = f'{int(train_size ** (2 / 3))}-' + k_qs.replace('/', '_')
            elif '3/4' in str(k_qs):
                k_qs_str = f'{int(train_size ** (3 / 4))}-' + k_qs.replace('/', '_')
            else:
                k_qs_str = 500 if 'qs' in model_name else None
            data_file = pth.join(in_dir, feat + "-header_" + str(is_header),
                                 data_name,
                                 "before_proj_" + str(is_before_proj) + "-gs_" + str(is_gs),
                                 model_name + f"-std_False_center_False-d_{str(d)}-{str(covariance_type)}-train_size_{train_size}-k_qs_{k_qs_str}",
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
            if "OCSVM" in model_name[0]:
                pth_cfg = (feat, is_header, is_before_proj, is_gs, d, None, model_name, ts)
                data = _merge_datasets([dataset_name], pth_cfg)
                res_[ts_tup] = data
            elif 'GMM' in model_name[0]:
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
                             f'{dataset_name}-{model_name[0]}-k_qs_{model_name[1]}.csv')
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
                                    in_files = {}
                                    for model_name in models:
                                        if 'OCSVM' in model_name[0]:
                                            covariance_type= None
                                        elif 'KJL-QS-GMM(full)' in model_name[0]:
                                            covariance_type = 'full'
                                        else:
                                            msg = model_name
                                            raise NotImplementedError(msg)
                                        out_file = _merge_models(feat_tup[1], header_tup[1], before_proj_tup[1],
                                                                 gs_tup[1], ds[0][1],
                                                                 covariance_type,
                                                                 dataset_name, model_name, train_sizes)
                                        in_files[model_name] = out_file
                                    # show_3train_sizes(in_files, out_file='')
                                    # show_3train_sizes_n_clusters(in_files, out_file='', show_name='tot_n_clusters')
                                    # show_3train_sizes_n_clusters(in_files, out_file='', show_name='n_clusters')
                                    show_3train_sizes_cluster_size_histgram(in_files, out_file='', show_name='n_clusters')
                                    show_3train_sizes_cluster_size_histgram2(in_files, out_file='',
                                                                            show_name='n_clusters')
                                except Exception as e:
                                    traceback.print_exc()
                                    lg.error(f"Error: {e}")

if __name__ == '__main__':

    only_show_ocsvm = False
    if only_show_ocsvm:
        in_dir = 'speedup/out_train_sizes'  # neon
        # in_dir = 'speedup/out_train_sizes/keep_small_clusters'  # neon
        # in_dir = 'speedup/calumet_out_ds-20210201'    # calumet
        merge_train_sizes_res(in_dir=in_dir, datasets= ['DWSHR_WSHR_2020'], #['UNB345_3', 'CTU1', 'MAWI1_2020', 'ISTS1', 'MACCDC1',  'SFRIG1_2020', 'AECHO1_2020', 'DWSHR_WSHR_2020'],
                              directions=[('direction', 'src_dst'), ],
                              feats=[('feat', 'iat_size'), ],
                              headers=[('is_header', False)],
                              models=MODELS,
                              gses=[('is_gs', False)],
                              before_projs=[('before_proj', False), ],
                              ds=[('d_kjl', 5)],
                              # train_sizes=[('train_size', v * 1000) for v in list(range(1, 5 + 1, 1))],
                              train_sizes=[('train_size', 100), ('train_size', 200), ('train_size', 400),
                                           ('train_size', 600), ('train_size', 800), ('train_size', 1000)],
                              )
    else:
        # in_dir = 'speedup/out_train_sizes'  # neon
        in_dir = 'speedup/out_train_sizes/keep_small_clusters' # neon
        merge_train_sizes_3res(in_dir=in_dir, datasets= ['CTU1'], #['MACCDC1, 'UNB345_3', 'CTU1', 'MAWI1_2020', 'SFRIG1_2020', 'AECHO1_2020', 'DWSHR_WSHR_2020'],
                              directions=[('direction', 'src_dst'), ],
                              feats=[('feat', 'iat_size'), ],
                              headers=[('is_header', False)],
                              models= [('OCSVM(rbf)', None),  ("KJL-QS-GMM(full)", '5/7'),  ("KJL-QS-GMM(full)", '2/3'), ("KJL-QS-GMM(full)", '3/4')], # [("KJL-QS-GMM(full)", '3/4')], #
                              gses=[('is_gs', True)],
                              before_projs=[('before_proj', False), ],
                              ds=[('d_kjl', 5)],
                              train_sizes=[('train_size', v * 1000) for v in list(range(1, 5 + 1, 1))],
                              k_qs=[('k_qs', '5/7'), ('k_qs', '2/3')],
                              )
