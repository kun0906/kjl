"""Data postprocess including data format transformation, highlight data

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import os, sys
import pickle
import traceback
from collections import OrderedDict
import pandas as pd
import numpy as np
from kjl.utils.tool import check_path

lib_path = os.path.abspath('.')
sys.path.append(lib_path)
print(f"add \'{lib_path}\' into sys.path: {sys.path}")
#
# import matplotlib
# matplotlib.use('TkAgg')     # pycharm can't use subplots_adjust() to show the legend and modified subplots correctly.

import copy
from operator import itemgetter

import xlsxwriter
from matplotlib import rcParams
from pandas import ExcelWriter

import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import ListedColormap
from itertools import chain
from fractions import Fraction

import seaborn as sns

# sns.set_style("darkgrid")

# colorblind for diff
sns.set_palette("bright")  # for feature+header
sns.palplot(sns.color_palette())


# plt.show()
# plt.close()
#
#
# def csv2xlsx(filename, detector_name='OCSVM', out_file='example.xlsx'):
#     read_file = pd.read_csv(filename, header=0, index_col=False)  # index_col=False: not use the first columns as index
#     read_file.to_excel(out_file, sheet_name=detector_name, index=0, header=True)
#
#     return out_file


def seaborn_palette(feat_type='', fig_type='raw'):
    # 1 bright used for basic
    # Set the palette to the "pastel" default palette:
    sns.set_palette("bright")
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_bright = sns.color_palette()

    # muted for FFT
    sns.set_palette("muted")  # feature+size
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_muted = sns.color_palette()

    # dark for feature + size
    sns.set_palette("dark")
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_dark = sns.color_palette()

    # deep for feature + header
    sns.set_palette("deep")  # for feature+header
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_deep = sns.color_palette()

    # colorblind for diff
    sns.set_palette("colorblind")  # for feature+header
    # sns.palplot(sns.color_palette());
    # plt.show()
    colors_colorblind = sns.color_palette()

    # construct cmap
    # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    # my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    colors_bright = ListedColormap(colors_bright.as_hex()).colors
    colors_dark = ListedColormap(colors_dark.as_hex()).colors
    colors_muted = ListedColormap(colors_muted.as_hex()).colors
    colors_deep = ListedColormap(colors_deep.as_hex()).colors
    colors_colorblind = ListedColormap(colors_colorblind.as_hex()).colors
    # colors_bright = ListedColormap(colors_bright.as_hex())

    feat_type = feat_type.upper()
    fig_type = fig_type.upper()

    C_STATS = 4  # purple
    C_IAT = 2  # green
    C_SIZE = 0  # blue
    C_SAMP_NUM = 3  # red
    C_SAMP_SIZE = 5  # brown

    raw_feat = {'STATS': colors_bright[C_STATS], 'IAT': colors_bright[C_IAT], 'SIZE': colors_bright[C_SIZE],
                'SAMP-NUM': colors_bright[C_SAMP_NUM],
                'SAMP-SIZE': colors_bright[C_SAMP_SIZE]}
    unique_flg = False
    if unique_flg:  # each feature has one color
        if feat_type == "basic_representation".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS': raw_feat['STATS'], 'IAT': raw_feat['IAT'], 'IAT-FFT': colors_dark[C_IAT],
                          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-NUM-FFT': colors_deep[C_SAMP_NUM]
                          }  # red
            elif fig_type == 'diff'.upper():
                # 'IAT' vs. IAT-FFT
                colors = {'IAT vs. IAT-FFT': colors_colorblind[C_IAT],  # green
                          'SAMP-NUM vs. SAMP-NUM-FFT': colors_colorblind[C_SAMP_NUM]}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)

        elif feat_type == "effect_size".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS': raw_feat['STATS'], 'SIZE': raw_feat['SIZE'], 'IAT': raw_feat['IAT'],
                          'IAT+SIZE': colors_dark[C_IAT],
                          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-SIZE': raw_feat['SAMP-SIZE']
                          }  # red
            elif fig_type == 'diff'.upper():
                colors = {'IAT vs. IAT+SIZE': colors_colorblind[C_IAT],  # green
                          'SAMP-NUM vs. SAMP-SIZE': colors_colorblind[C_SAMP_SIZE]}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)
        elif feat_type == "effect_header".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS (wo. header)': raw_feat['STATS'], 'STATS (w. header)': colors_deep[C_STATS],
                          'IAT+SIZE (wo. header)': colors_dark[C_IAT], 'IAT+SIZE (w. header)': colors_deep[C_IAT],
                          # green
                          'SAMP-SIZE (wo. header)': raw_feat['SAMP-SIZE'],
                          'SAMP-SIZE (w. header)': colors_deep[C_SAMP_SIZE]}  # red
            elif fig_type == 'diff'.upper():
                colors = {'STATS (wo. header) vs. STATS (w. header)': colors_colorblind[C_STATS],
                          'IAT+SIZE (wo. header) vs. IAT+SIZE (w. header)': colors_colorblind[C_SIZE],  # green
                          'SAMP-SIZE (wo. header) vs. SAMP-SIZE (w. header)': colors_bright[6]}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)

    else:  # for the paper
        if feat_type == "basic_representation".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS': raw_feat['STATS'], 'IAT': raw_feat['IAT'], 'IAT-FFT': colors_dark[C_IAT],
                          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-NUM-FFT': colors_dark[C_SAMP_NUM]
                          }  # red
            elif fig_type == 'diff'.upper():
                # 'IAT' vs. IAT-FFT
                colors = {'IAT vs. IAT-FFT': raw_feat['IAT'],  # green
                          'SAMP-NUM vs. SAMP-NUM-FFT': raw_feat['SAMP-NUM']}  # purple
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)

        elif feat_type == "effect_size".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS': raw_feat['STATS'], 'SIZE': raw_feat['SIZE'], 'IAT': raw_feat['IAT'],
                          'IAT+SIZE': colors_dark[C_SIZE],
                          'SAMP-NUM': raw_feat['SAMP-NUM'], 'SAMP-SIZE': raw_feat['SAMP-SIZE']
                          }  # red
            elif fig_type == 'diff'.upper():
                colors = {'IAT vs. IAT+SIZE': raw_feat['IAT'],  # green
                          'SAMP-NUM vs. SAMP-SIZE': raw_feat['SAMP-SIZE']}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)
        elif feat_type == "effect_header".upper():
            if fig_type == 'raw'.upper():
                colors = {'STATS (wo. header)': raw_feat['STATS'], 'STATS (w. header)': colors_dark[C_STATS],
                          'IAT+SIZE (wo. header)': colors_dark[C_SIZE], 'IAT+SIZE (w. header)': colors_deep[C_SIZE],
                          # green
                          'SAMP-SIZE (wo. header)': raw_feat['SAMP-SIZE'],
                          'SAMP-SIZE (w. header)': colors_deep[C_SAMP_SIZE]}  # red
            elif fig_type == 'diff'.upper():
                colors = {'STATS (wo. header) vs. STATS (w. header)': raw_feat['STATS'],
                          'IAT+SIZE (wo. header) vs. IAT+SIZE (w. header)': raw_feat['IAT'],  # green
                          'SAMP-SIZE (wo. header) vs. SAMP-SIZE (w. header)': raw_feat['SAMP-SIZE']}  # red
            else:
                msg = f'{feat_type} is not implemented yet.'
                raise ValueError(msg)

        else:
            msg = f'{feat_type} is not implemented yet.'
            raise ValueError(msg)

    return colors


MODELS = [  # algorithm name
    "OCSVM(rbf)",
    "KJL-OCSVM(linear)",

    "GMM(full)", "GMM(diag)",

    "KJL-GMM(full)", "KJL-GMM(diag)",

    "Nystrom-GMM(full)", "Nystrom-GMM(diag)",

    # quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    "QS-KJL-GMM(full)", "QS-KJL-GMM(diag)",
    "MS-KJL-GMM(full)", "MS-KJL-GMM(diag)",

    "QS-Nystrom-GMM(full)", "QS-Nystrom-GMM(diag)",
    "MS-Nystrom-GMM(full)", "MS-Nystrom-GMM(diag)",

    # quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    "KJL-QS-GMM(full)", "KJL-QS-GMM(diag)",
    "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"

    "Nystrom-QS-GMM(full)", "Nystrom-QS-GMM(diag)",
    "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"

]

DATASETS = [
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
    # # 'UNB45_4',
    #
    # 'UNB123_1',  # combine UNB1, UNB2, UNB3 attacks, only use UNB1 normal
    # 'UNB134_1',
    # 'UNB145_1',

    # 'UNB245_2',

    # # 'UNB234_2',  # combine UNB2, UNB3, UNB4 attacks, only use UNB2 normal
    'UNB35_3',  # combine  UNB3, UNB5 attacks, only use UNB3 normal
    'UNB345_3',  # combine UNB3, UNB3, UNB5 attacks, only use UNB3 normal
    # #
    # # 'UNB24',
    'CTU1',
    # # # # # 'CTU21', # normal + abnormal (botnet) # normal 10.0.0.15 (too few normal flows)
    # # # # # # 'CTU22',  # normal + abnormal (coinminer)
    # # # 'CTU31',  # normal + abnormal (botnet)   # 192.168.1.191
    'CTU32',  # normal + abnormal (coinminer)
    'MAWI1_2020',
    # # # # # # 'MAWI32_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32',
    # # # # # 'MAWI32-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.4.32-2',
    # # # 'MAWI165-2_2020',  # 'MAWI/WIDE_2020/pc_203.78.7.165-2',  # ~25000 (flows src_dst)
    'ISTS1',
    'MACCDC1',
    'SFRIG1_2020',
    'AECHO1_2020',
]

def parse_xlsx(input_file='xxx.dat-ratio.xlsx', tab_type = f'iat_size-header_False-gs_False-KJL-QS+Nystrom-QS'):

    df = pd.read_excel(input_file, header=0,index_col=None)  # index_col=False: not use the first columns as index
    vs = df.values
    res = {}

    # find the first_data line index, ignore the first few line in xlsx
    for i, line in enumerate(vs):
        if 'Xy-normal-abnormal.dat' in str(line[0]):
            idx_first_line = i
            break
    start_idx = 3
    while idx_first_line <= vs.shape[0] - i:
        line = vs[idx_first_line]
        data_name = str(line[0]).split('|')[0]   # (data_name, data_file)
        if data_name in DATASETS:
            res[data_name] = {}
            if tab_type == 'iat_size-header_False-gs_True-KJL-OCSVM+OCSVM-full' or tab_type == 'iat_size-header_False-gs_False-KJL-OCSVM+OCSVM-full' or \
                 tab_type == 'iat_size-header_False-gs_True-KJL-OCSVM+OCSVM-diag' or tab_type == 'iat_size-header_False-gs_False-KJL-OCSVM+OCSVM-diag':
                Needed_MOLDES = {'OCSVM': start_idx, 'KJL-OCSVM': start_idx + 1, 'Nystrom-OCSVM': start_idx + 2}  # (name:idx)
            elif tab_type == 'iat_size-header_False-gs_True-KJL-GMM-KJL-OCSVM(linear)-full'  or \
                    tab_type == 'iat_size-header_False-gs_True-KJL-GMM-KJL-OCSVM(linear)-diag' or \
                    tab_type == 'iat_size-header_False-gs_True-Nystrom-GMM-Nystrom-OCSVM(linear)-full'  or \
                    tab_type == 'iat_size-header_False-gs_True-Nystrom-GMM-Nystrom-OCSVM(linear)-diag':
                if 'KJL' in tab_type:
                    Needed_MOLDES = {'OC-KJL': 6, 'OC-KJL-SVM': 4}  # (name:idx)
                elif 'Nystrom' in tab_type:
                    Needed_MOLDES = {'OC-Nystrom': 7, 'OC-Nystrom-SVM': 5}  # (name:idx)
            elif tab_type == 'iat_size-header_False-gs_False-KJL-QS-Nystrom-QS-full' or \
                     tab_type == 'iat_size-header_False-gs_False-KJL-QS-Nystrom-QS-diag' :
                    Needed_MOLDES = {'OC-KJL-QS': -2, 'OC-Nystrom-QS': -1} # (name:idx)

            for model_name, t in Needed_MOLDES.items():
                # if model_name not in res[data_name].keys():
                res[data_name][model_name]={'Speedup AUC': vs[idx_first_line][t], 'Speedup training':vs[idx_first_line+1][t],
                                        'Speedup testing':vs[idx_first_line+2][t], 'Saving space': vs[idx_first_line+3][t]}
            idx_first_line +=4 + 1  # one blank line between two results
            continue
        else: # avoid dead loop
            idx_first_line +=1

    return res


def process_value(v):
    v_t = v.split("(")[0]
    v_t = f"{float(v_t):.2f}"
    return v_t



def show_KJL_QS_Nystrom_QS(input_file='', out_file='output_data', tab_type='', fig_flg='main_paper',
                      n_repeats = 5, gs =False,
                      verbose=1):
    """

    Parameters
    ----------
    input_file
    tab_latex
    caption
    do_header
    num_feat
    verbose

    Returns
    -------

    """
    # parse value from xlsx
    values_dict = parse_xlsx(input_file, tab_type)
    if out_file == '':
        out_file = input_file + "-" + tab_type + '-latex-figures.txt'
    print(out_file)

    show_detectors = []
    sns.set_style("darkgrid")
    # create plots
    num_figs = 4
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
            fig, axes = plt.subplots(r, num_figs, figsize=(18, 5))  # (width, height)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    metrics = ['Speedup AUC', 'Speedup training', 'Speedup testing', 'Saving space']
    # metrics = ['Speedup AUC']
    for j, diff_name in enumerate(metrics):
        sub_dataset = []
        yerrs = []  # std/sqrt(n_repeats)
        for k, vs in values_dict.items():
            if 'UNB' in k:
                k = 'UNB'
            elif 'CTU1' in k:
                k = 'CTU'
            elif 'MAWI' in k:
                k = 'MAWI'
            elif 'ISTS' in k:
                k = 'ISTS'
            elif 'MACCDC' in k:
                k = 'MACCDC'
            elif 'SFRIG' in k:
                k = 'SFRIG'
            elif 'AECHO' in k:
                k = 'AECHO'
            for model_name, diff_vs in vs.items():
                v= diff_vs[diff_name].split('+/-')
                mean_, std_ = float(v[0]), float(v[1].split("(")[0])
                if diff_name == 'Speedup AUC':
                    sub_dataset.append([k, f"{model_name} / OCSVM", mean_])
                elif diff_name in ['Speedup training', 'Speedup testing', 'Saving space']:
                    sub_dataset.append([k, f"OCSVM / {model_name}", mean_])
                yerrs.append(std_)

        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        print(f'{diff_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df,  ci=None,
                        capsize=.2, ax=axes[t, j % c])
        ys = []
        xs = []
        width = 0
        sub_fig_width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            # g.text(p.get_x() + p.get_width() / 2.,
            #        height,
            #        '{:0.3f}'.format(new_yerrs[i_p]),
            #        ha="center")
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            # yerr.append(i_p + p.get_height())

            num_bars = df['model_name'].nunique()
            # print(f'num_bars:',num_bars)
            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['dataset'].nunique() and i_p > 0:
                cur = p.get_x()
                # g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, alpha=0.3)
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width() * num_bars

        axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=yerrs, fmt='none', c='b', capsize=3)



        # for p in g.patches:
        #     height = p.get_height()
        #     # print(height)
        #     if  height < 0:
        #         height= height - 0.03
        #         xytext = (0, 0)
        #     else:
        #         height = p.get_height() + 0.03
        #         xytext = (0, 0)
        #     g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., height), ha='center',
        #                    va='center', xytext=xytext, textcoords='offset points')

        # g.set(xlabel=detector_name)       # set name at the bottom
        g.set(xlabel=None)
        g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        font_size = 20
        # g.set_ylabel(diff_name, fontsize=font_size + 4)
        g.set_ylabel(diff_name, fontsize=font_size + 4)
        if appendix:
            if j < len(show_detectors) - 1:
                g.set_xticklabels([])
            else:
                g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        else:
            g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        # y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        # v_max = max([ int(np.ceil(v)) for v in g.get_yticks()])
        # step = v_max/5
        # step = int(step) if step > 1 else step
        # y_v = [v for v in range(0, v_max, step)]
        if 'AUC' in diff_name:
            y_v = [0.0, 0.5, 1.0, 1.5, 2.0]
        elif 'train' in diff_name:
            y_v = [0.0, 0.5, 1.0, 1.5]
        elif 'test' in diff_name:
            y_v = [0, 25, 50, 75]
        elif 'space' in diff_name:
            y_v = [0,  10,  20, 30]

        g.set_yticks(y_v)  # set value locations in y axis
        # g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
        #                   fontsize=font_size + 6)  # set the number of each value in y axis
        g.set_yticklabels(y_v, fontsize=font_size + 4)  # set the number of each value in y axis
        print(g.get_yticks(), y_v)
        # if j % c != 0:
        #     # g.get_yaxis().set_visible(False)
        #     g.set_yticklabels(['' for v_tmp in y_v])
        #     g.set_ylabel('')

        # g.set_title(diff_name, fontsize=font_size + 8)
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        g.get_legend().set_visible(True)
        handles, labels = g.get_legend_handles_labels()
        axes[t, j % c].legend(handles, labels, loc="upper right",fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
    #
    # # # get the legend only from the last 'ax' (here is 'g')
    # handles, labels = g.get_legend_handles_labels()
    # labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    # # pos1 = axes[-1, -1].get_position()  # get the original position
    # # # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # # # ax.set_position(pos2) # set a new position
    # # loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    # # print(f'loc: {loc}, pos1: {pos1.bounds}')
    # # # axes[-1, -1].legend(handles, labels, loc=2, # upper right
    # # #             ncol=1, prop={'size': font_size-13})  # loc='lower right',  loc = (0.74, 0.13)
    # # axes[-1, -1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.0, 0.95, 1, 0.5),borderaxespad=0, fancybox=True, # upper right
    # #                     ncol=1, prop={'size': font_size - 4})  # loc='lower right',  loc = (0.74, 0.13)
    #
    # # share one legend
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
                plt.subplots_adjust(bottom=0.15, top=0.95)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    print(f'--{out_file}')
    plt.savefig(out_file + '.pdf')  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file





def show_KJL_GMM_KJL_OCSVM(input_file='', out_file='output_data', tab_type='', fig_flg='main_paper',
                      n_repeats = 5, gs =False,
                      verbose=1):
    """

    Parameters
    ----------
    input_file
    tab_latex
    caption
    do_header
    num_feat
    verbose

    Returns
    -------

    """
    # parse value from xlsx
    values_dict = parse_xlsx(input_file, tab_type)
    if out_file == '':
        out_file = input_file + "-" + tab_type + '-latex-figures.txt'
    print(out_file)

    show_detectors = []
    sns.set_style("darkgrid")
    # create plots
    num_figs = 1
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
            axes = np.asarray([axes]).reshape(1,1)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    metrics = ['Speedup AUC', 'Speedup training', 'Speedup testing', 'Saving space']
    metrics= ['Speedup AUC']
    for j, diff_name in enumerate(metrics):
        sub_dataset = []
        yerrs = []  # std/sqrt(n_repeats)
        for k, vs in values_dict.items():
            if 'UNB' in k:
                k = 'UNB'
            elif 'CTU1' in k:
                k = 'CTU'
            elif 'MAWI' in k:
                k = 'MAWI'
            elif 'ISTS' in k:
                k = 'ISTS'
            elif 'MACCDC' in k:
                k = 'MACCDC'
            elif 'SFRIG' in k:
                k = 'SFRIG'
            elif 'AECHO' in k:
                k = 'AECHO'
            for model_name, diff_vs in vs.items():
                v= diff_vs[diff_name].split('+/-')
                mean_, std_ = float(v[0]), float(v[1].split("(")[0])
                if diff_name == 'Speedup AUC':
                    sub_dataset.append([k, f"{model_name}", mean_])
                elif diff_name in ['Speedup training', 'Speedup testing', 'Saving space']:
                    sub_dataset.append([k, f"{model_name}", mean_])
                yerrs.append(std_)

        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        print(f'{diff_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df,  ci=None,
                        capsize=.2, ax=axes[t, j % c])
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, ci=None,
        #                 capsize=.2, ax=axes)
        ys = []
        xs = []
        width = 0
        sub_fig_width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            # g.text(p.get_x() + p.get_width() / 2.,
            #        height,
            #        '{:0.3f}'.format(new_yerrs[i_p]),
            #        ha="center")
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            # yerr.append(i_p + p.get_height())

            num_bars = df['model_name'].nunique()
            # print(f'num_bars:',num_bars)
            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['dataset'].nunique() and i_p > 0:
                cur = p.get_x()
                # g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, alpha=0.3)
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width() * num_bars

        axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=yerrs, fmt='none', c='b', capsize=3)



        # for p in g.patches:
        #     height = p.get_height()
        #     # print(height)
        #     if  height < 0:
        #         height= height - 0.03
        #         xytext = (0, 0)
        #     else:
        #         height = p.get_height() + 0.03
        #         xytext = (0, 0)
        #     g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., height), ha='center',
        #                    va='center', xytext=xytext, textcoords='offset points')

        # g.set(xlabel=detector_name)       # set name at the bottom
        g.set(xlabel=None)
        g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        font_size = 20
        # g.set_ylabel(diff_name, fontsize=font_size + 4)
        g.set_ylabel('AUC', fontsize=font_size + 4)
        if appendix:
            if j < len(show_detectors) - 1:
                g.set_xticklabels([])
            else:
                g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        else:
            g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        # y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        y_v = [ 0, 0.5, 1.0, 1.5]
        g.set_yticks(y_v)  # set value locations in y axis
        # g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
        #                   fontsize=font_size + 6)  # set the number of each value in y axis
        g.set_yticklabels(y_v, fontsize=font_size + 4)  # set the number of each value in y axis
        print(g.get_yticks(), y_v)
        # if j % c != 0:
        #     # g.get_yaxis().set_visible(False)
        #     g.set_yticklabels(['' for v_tmp in y_v])
        #     g.set_ylabel('')

        # g.set_title(diff_name, fontsize=font_size + 8)
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        g.get_legend().set_visible(False)
        # handles, labels = g.get_legend_handles_labels()
        # axes[t, j % c].legend(handles, labels, loc="upper right",fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)

    # # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
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
    fig.legend(handles, labels, loc='lower center',  # upper left
               ncol=3, prop={'size': font_size - 2})  # l

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
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file





def parse_xlsx_col(input_file='xxx.dat-ratio.xlsx', model_name  = 'KJL'):

    df = pd.read_excel(input_file, header=0,index_col=None)  # index_col=False: not use the first columns as index
    vs = df.values
    res = {}

    # find the first_data line index, ignore the first few line in xlsx
    for i, line in enumerate(vs):
        if 'Xy-normal-abnormal.dat' in str(line[0]):
            idx_first_line = i
            break

    while idx_first_line <= vs.shape[0] - i:
        line = vs[idx_first_line]
        data_name = str(line[0]).split('|')[0]   # (data_name, data_file)
        start_idx = 3
        if data_name in DATASETS:
            res[data_name] = {}
            if model_name == 'KJL-QS-GMM':
                Needed_MOLDES = {'KJL-QS-GMM': -2} # (name:idx)
            elif model_name == 'Nystrom-QS-GMM':
                Needed_MOLDES = {'Nystrom-QS-GMM': -1}  # (name:idx)
            # elif model_name == 'OCSVM':
            #     Needed_MOLDES = {'OCSVM': start_idx}  # (name:idx)
            # elif model_name == 'KJL-OCSVM':
            #     Needed_MOLDES = {'KJL-OCSVM': start_idx+1}  # (name:idx)
            # elif model_name == 'Nystrom-OCSVM':
            #     Needed_MOLDES = {'Nystrom-OCSVM': start_idx+2}  # (name:idx)
            for model_name, t in Needed_MOLDES.items():
                # if model_name not in res[data_name].keys():
                res[data_name][model_name]={'Speedup AUC': vs[idx_first_line][t], 'Speedup training':vs[idx_first_line+1][t],
                                        'Speedup testing':vs[idx_first_line+2][t], 'Saving space': vs[idx_first_line+3][t]}
            idx_first_line +=4 + 1  # one blank line between two results
            continue
        else: # avoid dead loop
            idx_first_line +=1

    return res


model_name_mapping={
    'OCSVM(rbf)': 'OCSVM',
    "KJL-OCSVM(linear)": "OC-KJL-OCSVM",
    "Nystrom-OCSVM(linear)": "OC-Nystrom-OCSVM",

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

    # quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    "KJL-QS-GMM(full)": "OC-KJL-QS",
    "KJL-QS-GMM(diag)": "OC-KJL-QS-diag",
    # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"

    "Nystrom-QS-GMM(full)": "OC-Nystrom-QS",
    "Nystrom-QS-GMM(diag)": "OC-Nystrom-QS-diag",
    # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"
}
def show_full_diag(full_file, diag_file, out_file='output_data', model_name='KJL', tab_type='', fig_flg='main_paper',
                      n_repeats = 5, gs =False,
                      verbose=1):
    """

    Parameters
    ----------
    input_file
    tab_latex
    caption
    do_header
    num_feat
    verbose

    Returns
    -------

    """
    # parse value from xlsx
    full_dict = parse_xlsx_col(full_file, model_name)
    diag_dict = parse_xlsx_col(diag_file, model_name)
    if out_file == '':
        out_file = full_file + "-" + tab_type + '-latex-figures.txt'
    print(out_file)

    show_detectors = []
    sns.set_style("darkgrid")
    # create plots
    num_figs = 4
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
            fig, axes = plt.subplots(r, num_figs, figsize=(18, 5))  # (width, height)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    metrics = ['Speedup AUC', 'Speedup training', 'Speedup testing', 'Saving space']
    # metrics = ['Speedup AUC']
    for j, diff_name in enumerate(metrics):
        sub_dataset = []
        yerrs = []  # std/sqrt(n_repeats)
        for k, vs in full_dict.items():
            if 'UNB' in k:
                k_abbr = 'UNB'
            elif 'CTU1' in k:
                k_abbr = 'CTU'
            elif 'MAWI' in k:
                k_abbr = 'MAWI'
            elif 'ISTS' in k:
                k_abbr = 'ISTS'
            elif 'MACCDC' in k:
                k_abbr = 'MACCDC'
            elif 'SFRIG' in k:
                k_abbr = 'SFRIG'
            elif 'AECHO' in k:
                k_abbr = 'AECHO'
            for model_name, diff_vs in vs.items():
                # full result
                v= diff_vs[diff_name].split('+/-')
                mean_full, std_full = float(v[0]), float(v[1].split("(")[0])
                # diag result
                v = diag_dict[k][model_name][diff_name].split('+/-')
                mean_diag, std_diag = float(v[0]), float(v[1].split("(")[0])
                if diff_name == 'Speedup AUC':
                    sub_dataset.append([k_abbr, f"{model_name_mapping[f'{model_name}(full)']} / OCSVM", mean_full])
                    sub_dataset.append([k_abbr, f"{model_name_mapping[f'{model_name}(diag)']} / OCSVM", mean_diag])
                elif diff_name in ['Speedup training', 'Speedup testing', 'Saving space']:
                    sub_dataset.append([k_abbr, f"OCSVM / {model_name_mapping[f'{model_name}(full)']}", mean_full])
                    sub_dataset.append([k_abbr, f"OCSVM / {model_name_mapping[f'{model_name}(diag)']}", mean_diag])
                yerrs.append(std_full)
                yerrs.append(std_diag )

        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        print(f'{diff_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df,  ci=None,
                        capsize=.2, ax=axes[t, j % c])
        ys = []
        xs = []
        width = 0
        sub_fig_width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            # g.text(p.get_x() + p.get_width() / 2.,
            #        height,
            #        '{:0.3f}'.format(new_yerrs[i_p]),
            #        ha="center")
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            # yerr.append(i_p + p.get_height())

            num_bars = df['model_name'].nunique()
            # print(f'num_bars:',num_bars)
            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['dataset'].nunique() and i_p > 0:
                cur = p.get_x()
                # g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, alpha=0.3)
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width() * num_bars

        axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=yerrs, fmt='none', c='b', capsize=3)



        # for p in g.patches:
        #     height = p.get_height()
        #     # print(height)
        #     if  height < 0:
        #         height= height - 0.03
        #         xytext = (0, 0)
        #     else:
        #         height = p.get_height() + 0.03
        #         xytext = (0, 0)
        #     g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., height), ha='center',
        #                    va='center', xytext=xytext, textcoords='offset points')

        # g.set(xlabel=detector_name)       # set name at the bottom
        g.set(xlabel=None)
        g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        font_size = 20
        g.set_ylabel(diff_name, fontsize=font_size + 4)
        if appendix:
            if j < len(show_detectors) - 1:
                g.set_xticklabels([])
            else:
                g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        else:
            g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        # y_v = [float(f'{x:.2f}') for x in g.get_yticks()]

        if 'AUC' in diff_name:
            y_v = [0.0, 0.5, 1.0, 1.5, 2.0]
        elif 'train' in diff_name:
            y_v = [0.0, 0.5, 1.0, 1.5]
        elif 'test' in diff_name:
            y_v = [0, 20, 40, 60]
        elif 'space' in diff_name:
            y_v = [0, 10, 20, 30]

        g.set_yticks(y_v)  # set value locations in y axis
        # g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
        #                   fontsize=font_size + 6)  # set the number of each value in y axis
        g.set_yticklabels(y_v, fontsize=font_size + 4)  # set the number of each value in y axis
        print(g.get_yticks(), y_v)
        # if j % c != 0:
        #     # g.get_yaxis().set_visible(False)
        #     g.set_yticklabels(['' for v_tmp in y_v])
        #     g.set_ylabel('')

        # g.set_title(diff_name, fontsize=font_size + 8)
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        # #     g.get_legend().set_visible()
        g.get_legend().set_visible(True)
        handles, labels = g.get_legend_handles_labels()
        axes[t, j % c].legend(handles, labels, loc="upper right",fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)

    # # # get the legend only from the last 'ax' (here is 'g')
    # handles, labels = g.get_legend_handles_labels()
    # labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    # # pos1 = axes[-1, -1].get_position()  # get the original position
    # # # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # # # ax.set_position(pos2) # set a new position
    # # loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    # # print(f'loc: {loc}, pos1: {pos1.bounds}')
    # # # axes[-1, -1].legend(handles, labels, loc=2, # upper right
    # # #             ncol=1, prop={'size': font_size-13})  # loc='lower right',  loc = (0.74, 0.13)
    # # axes[-1, -1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.0, 0.95, 1, 0.5),borderaxespad=0, fancybox=True, # upper right
    # #                     ncol=1, prop={'size': font_size - 4})  # loc='lower right',  loc = (0.74, 0.13)
    #
    # # share one legend
    # fig.legend(handles, labels, loc='lower center',  # upper left
    #            ncol=3, prop={'size': font_size - 4})  # l

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
                plt.subplots_adjust(bottom=0.15, top=0.95)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    print(f'{out_file}')
    plt.savefig(out_file + '.pdf')  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file


def get_diff(values_1, values_2):
    if ('sf' not in str(values_1[0])) or ('q_flow' not in str(values_1[0])) or ('interval' not in str(values_1[0])):
        return values_1

    values = []
    start_idx = 4
    for i, (v_1, v_2) in enumerate(zip(values_1, values_2)):
        if i < start_idx:
            values.append(v_1)
            continue
        if '(' in str(v_1):
            try:
                v_1 = v_1.split('(')[0]
                v_2 = v_2.split('(')[0]
                values.append(float(f'{float(v_1) - float(v_2):.4f}'))
            except Exception as e:
                print(f'i {i}, Error: {e}')
                values.append('-')
        else:
            values.append(v_1)

    return values


def diff_files(file_1, file_2, out_file='diff.xlsx', file_type='xlsx'):
    if file_type in ['xlsx']:
        xls = pd.ExcelFile(file_1)
        # Now you can list all sheets in the file
        print(f'xls.sheet_names:', xls.sheet_names)
        with ExcelWriter(out_file) as writer:
            for i, sheet_name in enumerate(xls.sheet_names):
                print(i, sheet_name)
                df_1 = pd.read_excel(file_1, sheet_name=sheet_name, header=None,
                                     index_col=None)  # index_col=False: not use the first columns as index

                df_2 = pd.read_excel(file_2, sheet_name=sheet_name, header=None,
                                     index_col=None)  # index_col=False: not use the first columns as index

                values = []
                for j, (v_1, v_2) in enumerate(list(zip(df_1.values, df_2.values))):
                    values.append(get_diff(v_1, v_2))
                # Generate dataframe from list and write to xlsx.
                pd.DataFrame(values).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                # styled_df = df_1.style.apply(highlight_cell, axis=1)
                # styled_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            writer.save()

    return out_file


def merge_xlsx(input_files=[], out_file='merged.xlsx'):
    workbook = xlsxwriter.Workbook(out_file)
    for t, (sheet_name, input_file) in enumerate(input_files):
        print(t, sheet_name, input_file)
        worksheet = workbook.add_worksheet(sheet_name)
        if not os.path.exists(input_file):
            # for i in range(rows):
            #     worksheet.write_row(i, 0, [str(v) if str(v) != 'nan' else '' for v in
            #                                    list(values[i])])  # write a list from (row, col)
            pass
        else:
            df = pd.read_excel(input_file, header=None, index_col=None)
            values = df.values
            rows, cols = values.shape
            # add column index
            # worksheet.write_row(0, 0, [str(i) for i in range(cols)])
            for i in range(rows):
                worksheet.write_row(i, 0, [str(v) if str(v) != 'nan' else '' for v in
                                           list(values[i])])  # write a list from (row, col)
            # worksheet.write(df.values)
    workbook.close()

    return out_file


def clean_xlsx(input_file, out_file='clean.xlsx'):
    xls = pd.ExcelFile(input_file)
    # Now you can list all sheets in the file
    print(f'xls.sheet_names:', xls.sheet_names)

    def process_line(v_arr):
        if 'sf:True-q_flow_dur:' in str(v_arr[1]):
            for i, v_t in enumerate(v_arr):
                # print(i,v_t)
                v_t = str(v_t)
                if 'q=' in v_t and 'dim=' in v_t:
                    v_arr[i] = v_t.split('(')[0]
                elif 'q_samp' in v_t and 'std' in v_t:
                    ts = v_t.split('(')
                    min_t = min([float(t) for t in ts[1].split('|')[4].split('=')[1].split('+')])
                    v_arr[i] = ts[0] + '|' + f'{min_t:.4f}'
                else:
                    pass
        return v_arr

    with ExcelWriter(out_file) as writer:
        for i, sheet_name in enumerate(xls.sheet_names):
            print(i, sheet_name)
            df = pd.read_excel(input_file, sheet_name=sheet_name, header=None,
                               index_col=None)  # index_col=False: not use the first columns as index
            values = []
            for j, v in enumerate(df.values):
                values.append(process_line(v))
            # Generate dataframe from list and write to xlsx.
            pd.DataFrame(values).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            # styled_df = df_1.style.apply(highlight_cell, axis=1)
            # styled_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        writer.save()

    return out_file


def txt_to_csv(txt_file, csv_file):
    with open(csv_file, 'w') as out_f:
        head = "ts,uid,id.orig_h(srcIP),id.orig_p(sport),id.resp_h(dstIP),id.resp_p(dport),proto,service,duration,orig_bytes,resp_bytes,conn_state,local_orig,local_resp,missed_bytes,history,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,tunnel_parents,label,detailed-label"
        out_f.write(head + '\n')
        with open(txt_file, 'r') as in_f:
            line = in_f.readline()
            while line:
                arr = line.split()
                l = ",".join(arr)
                out_f.write(l + '\n')

                line = in_f.readline()


def plot_bar_difference_seaborn_vs_stats(data=[], show_detectors=['OCSVM', 'AE'],
                                         show_datasets=['PC3(UNB)', 'MAWI', 'SFrig(private)'], gs=True,
                                         show_repres=['IAT', 'IAT-FFT', 'SIZE'],
                                         colors=['tab:brown', 'tab:green', 'm', 'c', 'b', 'r'],
                                         out_file="F1_for_all.pdf", xlim=[-0.1, 1]):
    # import seaborn as sns
    # sns.set_style("darkgrid")

    # create plots
    fig_cols = len(show_datasets) // 2 + 1
    figs, axes = plt.subplots(2, fig_cols)
    bar_width = 0.13
    opacity = 1

    s = min(len(show_repres), len(colors))
    for ind, dataset_name in enumerate(show_datasets):
        s_data = []
        for j, detector_name in enumerate(show_detectors):
            sub_dataset = []
            new_colors = []
            for i, (repres, color) in enumerate(zip(show_repres[:s], colors[:s])):
                if 'SAMP-' in repres.upper():
                    max_auc, min_auc = data[dataset_name][detector_name][gs][-1][i]
                    aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                else:
                    aucs = [float(data[dataset_name][detector_name][gs][-1][i]), 0]
                if i == 0:
                    pre_aucs = aucs
                    continue
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                sub_dataset.append((detector_name, repres, diff[0]))
                # #rects = plt.bar((ind) + (i) * bar_width, height=diff[0], width=bar_width, alpha=opacity, color=color,
                #  #               label='Frank' + str(i))
                # # g = sns.barplot(x='day', y='tip', data=groupedvalues, palette=np.array(pal[::-1])[rank])
                # sns.boxplot(y="b", x="a", data=diff, orient='v', ax=axes[ind])
                # # autolabel(rects, aucs=aucs, pre_aucs=pre_aucs)
                # pre_aucs = aucs
                new_colors.append(color)
            s_data.extend(sub_dataset)

        if ind % fig_cols == 0:
            if ind == 0:
                t = 0
            else:
                t += 1
        df = pd.DataFrame(s_data, columns=['detector', 'repres', 'diff'])
        g = sns.barplot(y="diff", x='detector', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set_ylim(-1, 1)
        g.set_title(dataset_name)
        g.get_legend().set_visible(False)

    # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    labels = ["\n".join(textwrap.wrap(v, width=13)) for v in labels]
    pos1 = axes[-1, -1].get_position()  # get the original position
    # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # ax.set_position(pos2) # set a new position
    loc = (pos1.x0 + 0.05, pos1.y0 + 0.05)
    print(f'loc: {loc}, pos1: {pos1.bounds}')
    figs.legend(handles, labels, title='Representation', loc=loc,
                ncol=1, prop={'size': 8})  # loc='lower right',  loc = (0.74, 0.13)

    ind += 1
    while t < 2:
        if ind % fig_cols == 0:
            t += 1
            if t >= 2:
                break
            ind = 0
        # remove subplot
        # fig.delaxes(axes[1][2])
        axes[t, ind % fig_cols].set_axis_off()
        ind += 1

    # # plt.xlabel('Catagory')
    # plt.ylabel('AUC')
    # plt.ylim(0,1.07)
    # # # plt.title('F1 Scores by category')
    # n_groups = len(show_datasets)
    # index = np.arange(n_groups)
    # # print(index)
    # # plt.xlim(xlim[0], n_groups)
    # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
    # plt.tight_layout()
    #

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(out_file)  # should use before plt.show()
    plt.show()

    del fig  # must have this.

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})


def plot_bar_difference_seaborn(data=[], show_detectors=['OCSVM', 'AE'],
                                show_datasets=['PC3(UNB)', 'MAWI', 'SFrig(private)'], gs=True,
                                show_repres=['IAT', 'IAT-FFT', 'SIZE'],
                                colors=['tab:brown', 'tab:green', 'm', 'c', 'b', 'r'],
                                out_file="F1_for_all.pdf", xlim=[-0.1, 1], tab_type='', appendix=False):
    sns.set_style("darkgrid")
    print(show_detectors)
    # create plots
    num_figs = len(show_detectors)
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
            fig, axes = plt.subplots(r, num_figs, figsize=(18, 5))  # (width, height)
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
            fig, axes = plt.subplots(r, num_figs, figsize=(18, 5))  # (width, height)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0
    s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    for j, detector_name in enumerate(show_detectors):
        sub_dataset = []
        new_colors = []
        yerrs = []
        for ind, dataset_name in enumerate(show_datasets):
            n = 100
            test_size = sum([int(v) for v in data[detector_name][dataset_name]['test_size']])
            yerr = []
            if dataset_name not in new_data.keys():
                new_data[dataset_name] = {detector_name: {}, 'test_size': [], 'yerr': []}

            if tab_type.upper() == "basic_representation".upper():
                features = ['STATS', 'IAT', 'IAT-FFT', 'SAMP-NUM', 'SAMP-NUM-FFT','SAMP-SIZE', 'SAMP-SIZE-FFT']
                f_dict = dict(zip(features, [i for i in range(len(features))]))
                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'IAT'
                B = 'IAT-FFT'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-NUM'
                B = 'SAMP-NUM-FFT'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]
                # new_colors = ['b', 'r']
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-SIZE'
                B = 'SAMP-SIZE-FFT'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]
                # new_colors = ['b', 'r']
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()
                yerr = [1 / np.sqrt(test_size)] * 3

                # yerr = [1 / np.sqrt(test_size)] * 2

            elif tab_type.upper() == "effect_size".upper():
                features = ['STATS', 'SIZE', 'IAT', 'IAT+SIZE', 'SAMP-NUM', 'SAMP-SIZE']
                f_dict = dict(zip(features, [i for i in range(len(features))]))

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'IAT'
                B = 'IAT+SIZE'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-NUM'
                B = 'SAMP-SIZE'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ {A}"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]

                new_colors = seaborn_palette(tab_type, fig_type='diff').values()
                yerr = [1 / np.sqrt(test_size)] * 2
            elif tab_type.upper() == "effect_header".upper():
                features = ['STATS (wo. header)', 'STATS (w. header)', 'IAT+SIZE (wo. header)', 'IAT+SIZE (w. header)',
                            'SAMP-SIZE (wo. header)', 'SAMP-SIZE (w. header)']
                f_dict = dict(zip(features, [i for i in range(len(features))]))

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'STATS (wo. header)'
                B = 'STATS (w. header)'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ (wo. header)"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}
                new_colors = seaborn_palette(tab_type, fig_type='diff').values()

                A = 'IAT+SIZE (wo. header)'
                B = 'IAT+SIZE (w. header)'
                pre_aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[A]]), 0]
                aucs = [float(data[detector_name][dataset_name][gs][-1][f_dict[B]]), 0]
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ (wo. header)"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                # new_data[dataset_name][detector_name] = {gs: {repres_pair: diff[0]}}
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]

                # datasets[key][d_name][gs] = (fig2, label2, caption2, d)
                A = 'SAMP-SIZE (wo. header)'
                B = 'SAMP-SIZE (w. header)'
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[A]].split('(')[1].split(')')[
                    0].split('-')
                pre_aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                max_auc, min_auc = data[detector_name][dataset_name][gs][-1][f_dict[B]].split('(')[1].split(')')[
                    0].split('-')
                aucs = [float(max_auc), float(min_auc)]  # max and min aucs
                diff = [v - p_v for p_v, v in zip(pre_aucs, aucs)]
                repres_pair = f"{B} \\ (wo. header)"
                sub_dataset.append((dataset_name, repres_pair, diff[0]))
                new_data[dataset_name][detector_name][gs][repres_pair] = diff[0]

                yerr = [1 / np.sqrt(test_size)] * 3

            new_data[dataset_name]['test_size'] = test_size
            new_data[dataset_name]['yerr'] = yerr

            yerrs.append(yerr)
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1
        print(f'{detector_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['dataset', 'repres', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('before yerrs:', yerrs)
        new_yerrs = []
        # yerrs=list(chain.from_iterable(yerrs))
        yerrs = np.asarray(yerrs)
        for c_tmp in range(yerrs.shape[1]):  # extend by columns
            new_yerrs.extend(yerrs[:, c_tmp])
        print('new_yerrs:', new_yerrs)
        g = sns.barplot(y="diff", x='dataset', hue='repres', data=df, palette=new_colors, ci=None,
                        capsize=.2, ax=axes[t, j % c])
        ys = []
        xs = []
        width = 0
        sub_fig_width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            # g.text(p.get_x() + p.get_width() / 2.,
            #        height,
            #        '{:0.3f}'.format(new_yerrs[i_p]),
            #        ha="center")
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            # yerr.append(i_p + p.get_height())

            num_bars = df['repres'].nunique()
            # print(f'num_bars:',num_bars)
            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['dataset'].nunique() and i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.3)
                pre = cur + p.get_width() * num_bars

        axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=new_yerrs, fmt='none', c='b', capsize=3)
        # for p in g.patches:
        #     height = p.get_height()
        #     # print(height)
        #     if  height < 0:
        #         height= height - 0.03
        #         xytext = (0, 0)
        #     else:
        #         height = p.get_height() + 0.03
        #         xytext = (0, 0)
        #     g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., height), ha='center',
        #                    va='center', xytext=xytext, textcoords='offset points')

        # g.set(xlabel=detector_name)       # set name at the bottom
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set_ylim(-1, 1)
        font_size = 20
        g.set_ylabel('AUC difference', fontsize=font_size + 4)
        if appendix:
            if j < len(show_detectors) - 1:
                g.set_xticklabels([])
            else:
                g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        else:
            g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        g.set_yticks(y_v)  # set value locations in y axis
        g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
                          fontsize=font_size + 6)  # set the number of each value in y axis
        print(g.get_yticks(), y_v)
        if j % c != 0:
            # g.get_yaxis().set_visible(False)
            g.set_yticklabels(['' for v_tmp in y_v])
            g.set_ylabel('')

        g.set_title(detector_name, fontsize=font_size + 8)
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        g.get_legend().set_visible(False)

    # # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    labels = ["\n".join(textwrap.wrap(v, width=45)) for v in labels]
    # pos1 = axes[-1, -1].get_position()  # get the original position
    # # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # # ax.set_position(pos2) # set a new position
    # loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    # print(f'loc: {loc}, pos1: {pos1.bounds}')
    # # axes[-1, -1].legend(handles, labels, loc=2, # upper right
    # #             ncol=1, prop={'size': font_size-13})  # loc='lower right',  loc = (0.74, 0.13)
    # axes[-1, -1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.0, 0.95, 1, 0.5),borderaxespad=0, fancybox=True, # upper right
    #                     ncol=1, prop={'size': font_size - 4})  # loc='lower right',  loc = (0.74, 0.13)

    fig.legend(handles, labels, loc='lower center',  # upper left
               ncol=3, prop={'size': font_size - 2})  # l

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
                plt.subplots_adjust(bottom=0.18)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(out_file)  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return new_data, new_colors


def plot_bar_ML_seaborn(new_data=[], new_colors=[], show_detectors=[], gs=True,
                        out_file="F1_for_all.pdf", xlim=[-0.1, 1], tab_type=''):
    # import seaborn as sns
    sns.set_style("darkgrid")
    print(show_detectors)
    # create plots
    num_figs = len(new_data.keys())
    c = 3
    if num_figs >= c:
        if num_figs % c == 0:
            r = int(num_figs // c)
        else:
            r = int(num_figs // c) + 1  # in each row, it show 4 subplot
        figs, axes = plt.subplots(r, c, figsize=(65, 40))  # (width, height)
    else:
        figs, axes = plt.subplots(1, num_figs, figsize=(20, 8))  # (width, height)
        axes = axes.reshape(1, -1)
    print(f'subplots: ({r}, {c})')
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0
    for i, dataset_name in enumerate(new_data.keys()):
        sub_dataset = []
        yerrs = []
        for j, (detector_name, dv_dict) in enumerate(new_data[dataset_name].items()):
            if detector_name not in show_detectors:
                continue
            for _, (repres_pair, diff) in enumerate(dv_dict[gs].items()):
                sub_dataset.append((detector_name, repres_pair, float(diff)))
            yerrs.append(new_data[dataset_name]['yerr'])
        print(f'{dataset_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['detector', 'representation', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        if i % c == 0 and i > 0:
            t += 1
        g = sns.barplot(y="diff", x='detector', data=df, hue='representation', ax=axes[t, i % c],
                        palette=new_colors)  # palette=show_clo
        # for index, row in df.iterrows():
        #     g.text(row.name, row.representation, round(row.auc, 2), color='black', ha="center")
        yerrs = np.asarray(yerrs)
        new_yerrs = []
        for c_tmp in range(yerrs.shape[1]):  # extend by columns
            new_yerrs.extend(yerrs[:, c_tmp])
        print('new_yerrs for ML diff:', new_yerrs)

        ys = []
        xs = []
        width = 0
        sub_fig_width = 0
        for i_p, p in enumerate(g.patches):
            height = p.get_height()
            # g.text(p.get_x() + p.get_width() / 2.,
            #        height,
            #        '{:0.3f}'.format(new_yerrs[i_p]),
            #        ha="center")
            width = p.get_width()
            ys.append(height)
            xs.append(p.get_x())
            # yerr.append(i_p + p.get_height())

            num_bars = df['representation'].nunique()
            # print(f'num_bars:',num_bars)
            if i_p == 0:
                pre = p.get_x() + p.get_width() * num_bars
                sub_fig_width = p.get_bbox().width
            if i_p < df['detector'].nunique() and i_p > 0:
                cur = p.get_x()
                g.axvline(color='black', linestyle='--', x=pre + (cur - pre) / 2, ymin=0, ymax=1, alpha=0.6)
                pre = cur + p.get_width() * num_bars

        axes[t, i % c].errorbar(x=xs + width / 2, y=ys,
                                yerr=new_yerrs, fmt='none', c='b', capsize=3)

        # g.set(xlabel=dataset_name)
        # g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        # # g.set_title(detector_name, y=-0.005)
        # g.get_legend().set_visible(False)

        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set_ylim(-1, 1)
        font_size = 95
        g.set_xticklabels(g.get_xticklabels(), fontsize=font_size - 5)
        # g.set_yticklabels(['{:,.2f}'.format(x) for x in g.get_yticks()], fontsize=font_size - 3)
        # print(g.get_yticks())
        g.set_ylabel('AUC difference', fontsize=font_size + 4)
        y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x % 0.5 == 0]
        g.set_yticks(y_v)  # set value locations in y axis
        g.set_yticklabels(y_v, fontsize=font_size - 3)  # set the number of each value in y axis
        print(g.get_yticks(), y_v)
        if i % c != 0:
            g.get_yaxis().set_visible(False)

        g.set_title(dataset_name, fontsize=font_size)
        g.get_legend().set_visible(False)

    # get the legend only from the last 'ax' (here is 'g')
    handles, labels = g.get_legend_handles_labels()
    # labels = ["\n".join(textwrap.wrap(v, width=26)) for v in labels]
    print('---', labels)
    pos1 = axes[-1, -1].get_position()  # get the original position
    # pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0]
    # ax.set_position(pos2) # set a new position
    loc = (pos1.x0 + (pos1.x1 - pos1.x0) / 2, pos1.y0 + 0.05)
    print(f'loc: {loc}, pos1: {pos1.bounds}')

    # cnt_tmp = r * c - num_figs
    # if cnt_tmp == 0:
    #     last_num = 1
    #     labels = ["\n".join(textwrap.wrap(v, width=26)) for v in labels]
    # else:  # elif cnt_tmp > 0:
    #     last_num = 2
    # gs = axes[r - 1, -last_num].get_gridspec()  # (1, 1): the start point of the new merged subplot
    # for ax in axes[r - 1, -last_num:]:
    #     ax.remove()
    # axbig = figs.add_subplot(gs[r - 1, -last_num:])
    #
    # axbig.legend(handles, labels, loc=2,  # upper left
    #              ncol=1, prop={'size': font_size - 25})  # loc='lower right',  loc = (0.74, 0.13)
    # # figs.legend(handles, labels, title='Representation', bbox_to_anchor=(2, 1), loc='upper rig ht', ncol=1)
    figs.legend(handles, labels, loc='lower center',  # upper left
                ncol=3, prop={'size': font_size - 25})  # l

    # # remove subplot
    # # fig.delaxes(axes[1][2])
    # axbig.set_axis_off()

    i += 1
    while t < r:
        if i % c == 0:
            t += 1
            if t >= r:
                break
            i = 0
        # remove subplot
        # fig.delaxes(axes[1][2])
        axes[t, i % c].set_axis_off()
        i += 1

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
            plt.subplots_adjust(bottom=0.2)
        else:
            plt.subplots_adjust(bottom=0.1)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(out_file)  # should use before plt.show()
    plt.show()
    plt.close(figs)

    # sns.reset_orig()
    # sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})



def main():
    case = 'find_difference'
    case = 'csv2figs'
    out_dir = 'speedup/out/report/src_dst/'
    if case == 'csv2latex2':
        out_file = out_dir + 'results_latex.txt'
        header = False
        num_feat = 3
        # if num_feat == 3:
        #     tab_latex = tab_latex_3
        # else:
        #     tab_latex = tab_latex_7
        # with open(out_file, 'w') as f:
        #     for i, (detector, input_file) in enumerate(input_files.items()):
        #         tab_false, tab_true = csv2latex_previous(input_file=input_file,
        #                                                  tab_latex=tab_latex, caption=detector, do_header=header,
        #                                                  previous_result=True,
        #                                                  num_feat=num_feat,
        #                                                  verbose=0)
        #
        #         # tab_false, tab_true = csv2latex(input_file=input_file,
        #         #                                 tab_latex=tab_latex, caption=detector, do_header=header,
        #         #                                 num_feat=num_feat,
        #         #                                 verbose=0)
        #
        #         f.write(detector + ':' + input_file + '\n')
        #         for line in tab_false:
        #             f.write(line + '\n')
        #         print()
        #
        #         for line in tab_true:
        #             f.write(line + '\n')
        #         print()
        #         f.write('\n')
    elif case == 'csv2figs':
        # # case 1
        # gs = True
        # name = 'show_KJL_GMM_KJL_OCSVM'
        #
        # case 2
        gs = False
        name = 'show_KJL_QS_Nystrom_QS'

        # case 3
        # gs = True
        # name = 'show_full_diag'


        covariance = 'full'
        fig_flg = 'main'
        # feat_header = 'iat_size-header_False'
        feat_header = 'stats-header_True'
        file_name = f'speedup/calumet_out-20210204/src_dst/{feat_header}/before_proj_False-gs_{gs}/std_False_center_False-d_5-{covariance}/res.csv-ratio.xlsx'



        if gs ==True and name == 'show_KJL_GMM_KJL_OCSVM':
            # KJL-GMM vs. KJL-OCSVM(linear)
            for i, name in enumerate(['KJL', 'Nystrom']):
                tab_type = f'{feat_header}-gs_{gs}-{name}-GMM-{name}-OCSVM(linear)-{covariance}'
                out_dir = os.path.dirname(file_name)
                out_file = f'{out_dir}/All_latex_tables_figs.txt'  # for main paper results
                check_path(out_file)
                with open(out_file, 'w') as f:
                    print('\n\n******************')
                    print(i, tab_type, gs)
                    try:

                        # # only for best params
                        # if tab_type == 'iat_size-header_False-gs_True-KJL-GMM-KJL-OCSVM(linear)-full'  or \
                        #         tab_type == 'iat_size-header_False-gs_True-KJL-GMM-KJL-OCSVM(linear)-diag':
                        show_KJL_GMM_KJL_OCSVM(input_file=file_name,
                                               out_file=os.path.dirname(
                                                   out_file) + "/" + tab_type + '-' + fig_flg,
                                               tab_type=tab_type, fig_flg=fig_flg, gs=gs,
                                           n_repeats=5)
                    except Exception as e:
                        print('Error: ', i, e)
                        traceback.print_exc()
                        continue

                # release the matplotlib memory
                # Clear the current figure.
                plt.clf()
                # Closes all the figure windows.
                plt.close('all')

                import gc
                gc.collect()

        elif gs == False and name == 'show_KJL_QS_Nystrom_QS':
            # KJL-QS-GMM, Nystrom-QS-GMM
            # tab_type = f'{feat_header}-gs_{gs}-{name}-GMM-{name}-OCSVM(linear)-{covariance}'
            tab_type = 'iat_size-header_False-gs_False-KJL-QS-Nystrom-QS-full'
            out_dir = os.path.dirname(file_name)
            out_file = f'{out_dir}/All_latex_tables_figs.txt'  # for main paper results
            check_path(out_file)
            i = 0
            with open(out_file, 'w') as f:
                print('\n\n******************')
                print(i, tab_type, gs)
                try:

                    # # only for best params
                    # if tab_type == 'iat_size-header_False-gs_True-KJL-GMM-KJL-OCSVM(linear)-full'  or \
                    #         tab_type == 'iat_size-header_False-gs_True-KJL-GMM-KJL-OCSVM(linear)-diag':
                    show_KJL_QS_Nystrom_QS(input_file=file_name,
                                           out_file=os.path.dirname(
                                               out_file) + "/" + tab_type + '-' + fig_flg,
                                           tab_type=tab_type, fig_flg=fig_flg, gs=gs,
                                           n_repeats=5)
                except Exception as e:
                    print('Error: ', i, e)
                    traceback.print_exc()


            # release the matplotlib memory
            # Clear the current figure.
            plt.clf()
            # Closes all the figure windows.
            plt.close('all')

            import gc
            gc.collect()
        else:
            # tab_type = f'{feat_header}-gs_{gs}-KJL-QS-full+KJL-QS-diag'
            for model_name, tab_type in  [('KJL-QS-GMM', 'iat_size-header_False-gs_True-KJL-QS-full+KJL-QS-diag'),
                              ('Nystrom-QS-GMM', 'iat_size-header_False-gs_True-Nystrom-QS-full+Nystrom-QS-diag')]:

                full_file =   f'speedup/calumet_out-20210204/src_dst/{feat_header}/before_proj_False-gs_True/std_False_center_False-d_5-{covariance}/res.csv-ratio.xlsx'
                diag_file  = f'speedup/calumet_out-20210204/src_dst/{feat_header}/before_proj_False-gs_False/std_False_center_False-d_5-{covariance}/res.csv-ratio.xlsx'


                out_dir = os.path.dirname(full_file)
                out_file = f'{out_dir}/All_latex_tables_figs.txt'  # for main paper results
                check_path(out_file)

                show_full_diag(full_file, diag_file,
                               out_file=os.path.dirname(
                                   out_file) + "/" + tab_type + '-' + fig_flg,
                               tab_type=tab_type, model_name=model_name, gs=gs,
                           n_repeats=5)




if __name__ == '__main__':
    main()
