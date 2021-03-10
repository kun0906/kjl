""" Merge all result to form the final report

"""

import os
import os.path as pth
import traceback
import copy
from collections import OrderedDict

import numpy as np
import pandas as pd

from kjl.log import get_log
from kjl.utils.data import _get_line
from kjl.utils.tool import dump_data, check_path
# get log
from speedup.ratio_variance import improvement, dat2latex, dat2xlxs_new

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

lg = get_log(level='info')


def _dat2csv(result, out_file, feat_set='iat_size'):
    (in_dir, case_str), (best_results, middle_results) = result
    data = best_results
    with open(out_file, 'w') as f:
        try:
            # best_auc = data['best_auc']
            aucs = data['aucs']
            params = data['params'][-1]
            train_times = data['train_times']
            test_times = data['test_times']
            # params = data['params']
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
                if 'qs_res' in data['params'][0]:
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
                else:
                    tot_clusters_str = 0
                    tot_clusters_str2 = 0
                    n_clusters_str = 0
                    n_clusters_str2 = 0
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
                    params['qs_res']['tot_clusters'] = '-'.join([f'{k}:{v}' for k,v in params['qs_res']['tot_clusters'].items()])
            line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs:{aucs_str}, ' \
                   f'train_times:{train_times_str}, test_times:{test_times_str}, n_comp: {n_comp_str}, ' \
                   f'{n_comp_str2}, space_sizes: {space_size_str},tot_clusters:{tot_clusters_str}, {tot_clusters_str2},' \
                   f'n_clusters: {n_clusters_str}, {n_clusters_str2}, with params: {params}: {_suffex}'

        except Exception as e:
            traceback.print_exc()
            line = ''
        f.write(line + '\n')


def merge_res(in_dir='speedup/out', directions=[('direction', 'src_dst'), ], datasets=None, models=None,
              feats=[('feat', 'iat_size')], headers= [('is_header', True), ('is_header', False)],
              before_projs=None, gses=[('is_gs', True), ('is_gs', False)], ds=[('d_kjl', 5), ],
              covariances=[('covariance_type', 'full'), ('covariance_type', 'diag')]):

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
        (feat, is_header, is_before_proj, is_gs, d, covariance_type, model_name) = pth_cfg
        data = []
        for data_name in datasets:
            data_file = pth.join(in_dir, feat + "-header_" + str(is_header),
                                 data_name,
                                 "before_proj_" + str(is_before_proj) + "-gs_" + str(is_gs),
                                 model_name + f"-std_False_center_False-d_{str(d)}-{str(covariance_type)}",
                                 'res.dat.csv')
            print(f'{data_file}: {pth.exists(data_file)}')
            if 'GMM' in data_file:
                if f'GMM({covariance_type})' not in data_file:
                    continue
            try:
                df = pd.read_csv(data_file, header=None)
                data.extend(df.values)
            except Exception as e:
                data.extend(np.array([[f'{data_name}|{data_file}', model_name, 'X_train_shape', 'X_test_shape', feat]]))
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

    def _merge_models(feat, is_header, is_before_proj, is_gs, d, covariance_type):
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
        vs = [] # store for csv
        for i, model_name in enumerate(models):
            if "OCSVM" in model_name:
                pth_cfg = (feat, is_header, is_before_proj, is_gs, d, None, model_name)
                data = _merge_datasets(datasets, pth_cfg)
                res_[model_name] = data
            elif 'GMM' in model_name:
                pth_cfg = (feat, is_header, is_before_proj, is_gs, d, covariance_type, model_name)
                data = _merge_datasets(datasets, pth_cfg)
                res_[model_name] = data
            else:
                msg = model_name
                raise NotImplementedError(msg)

            # store for csv
            if i == 0:
                vs = copy.deepcopy(data)
            else:
                vs.extend(data)

        #  'feat-header_false-before_proj_False-gs_True-diag-std_False_center_False-d_5'
        out_file_ = pth.join(in_dir, feat + "-header_" + str(is_header),
                             "before_proj_" + str(is_before_proj) + "-gs_" + str(is_gs),
                             f"std_False_center_False-d_{str(d)}-{str(covariance_type)}", 'res.csv')
        print(f'data_models: {out_file_}')
        check_path(out_file_)
        out_file_dat = out_file_ + '.dat'
        dump_data(res_, out_file=out_file_dat)
        # save as csv
        pd.DataFrame(vs).to_csv(out_file_, index=False, encoding='utf-8-sig')
        # save as xlsx
        out_xlsx = dat2xlxs_new(out_file_dat, out_file=out_file_dat + '.xlsx', models=models)
        # compute ratio OCSVM/GMM
        out_xlsx_ratio = improvement(out_xlsx, feat_set=feat,
                                     out_file=os.path.splitext(out_file_dat)[0] + '-ratio.xlsx')
        print(out_xlsx)

        # for paper
        out_latex = dat2latex(out_xlsx_ratio, out_file=os.path.splitext(out_file_dat)[0] + '-latex.xlsx')
        print(out_latex)

        return out_file_

    for direction_tup in directions:
        in_dir = pth.join(in_dir, direction_tup[1])
        for feat_tup in feats:
            for header_tup in headers:
                for before_proj_tup in before_projs:
                    for gs_tup in gses:
                        for d_tup in ds:
                            for covariance_type_tup in covariances:
                                try:
                                    _merge_models(feat_tup[1], header_tup[1], before_proj_tup[1],
                                                  gs_tup[1], d_tup[1],
                                                  covariance_type_tup[1])
                                except Exception as e:
                                    traceback.print_exc()
                                    lg(f"Error: {e}", level='error')


def parse_file(in_file='.csv'):
    df = pd.read_csv(in_file)
    vs = df.values
    res = {}

    # find the first_data line index, ignore the first few line in xlsx
    idx_first_line = -1
    for i, line in enumerate(vs):
        if 'Xy-normal-abnormal.dat' in str(line[0]):
            idx_first_line = i
            break
    if idx_first_line ==-1:
        msg = f'{in_file}'
        raise ValueError(msg)

    while idx_first_line < vs.shape[0] - i:
        try:
            line = vs[idx_first_line]
            data_name = str(line[0]).split('|')[0]  # (data_name, data_file)
            model_name, d = line[1].split('|')
            spaces = [float(v)  for v in line[12].split(':')[-1].split('-')]
            diff_values = {'AUC': line[4].split(':')[-1],
                          'Training': line[5].split(':')[-1],
                         'Testing': line[6].split(':')[-1].split(')')[0],  # '0.23)'
                         'Space': f"{np.mean(spaces)}+/-{np.std(spaces)}"}
        except Exception as e:
            diff_values = {'AUC': '0+/-0',
                          'Training': '0+/-0',
                         'Testing':'0+/-0',
                         'Space':'0+/-0'}
        if data_name not in res.keys():
            res[data_name] = {(model_name, d): diff_values}
        else:
            res[data_name][(model_name, d)]= diff_values
        idx_first_line += 1

    return res


def show(in_file, out_file='', title='auc', n_repeats = 5):
    # parse value from xlsx
    values_dict = parse_file(in_file)
    if out_file == '':
        out_file = in_file + '-res.txt'
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

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    for j, diff_name in enumerate(['AUC', 'Training', 'Testing', 'Space']):
        sub_dataset = []
        yerrs = []  # std/sqrt(n_repeats)
        for k, vs in values_dict.items():
            data_name = k
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
            for (model_name, d), diff_vs in vs.items():
                d = float(d.split('_')[-1])
                v = diff_vs[diff_name].split('+/-')
                print(v)
                mean_, std_ = float(v[0]), float(v[1].split("(")[0])
                if diff_name == 'AUC':
                    sub_dataset.append([k, f"{model_name}",d, mean_])
                elif diff_name in ['Training', 'Testing', 'Space']:
                    sub_dataset.append([k, f"{model_name}",d, mean_])
                yerrs.append(std_ / np.sqrt(n_repeats))

        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        print(f'{diff_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=[ 'dataset', 'model_name', 'd', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.lineplot(y="diff", x='d', data=df, ci=None, hue = 'model_name',
                        ax=axes[t, j % c])
        axes[t, j % c].errorbar(df['d'].values, df['diff'].values, yerrs, capsize=3,  color='tab:blue', ecolor='tab:red')
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

        # # g.set(xlabel=detector_name)       # set name at the bottom
        g.set(xlabel=None)
        g.set(ylabel=None)
        # g.set_ylim(-1, 1)
        font_size = 20
        g.set_ylabel(diff_name, fontsize=font_size + 4)
        # if appendix:
        #     if j < len(show_detectors) - 1:
        #         g.set_xticklabels([])
        #     else:
        #         g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # else:
        #     g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        x_v = [int(x) for x in g.get_xticks()]
        g.set_xticks(x_v)  # set value locations in y axis
        g.set_xticklabels(x_v, fontsize=font_size + 2)  # set the number of each value in y axis

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
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
        # axes[t, j % c].legend(handles, labels, loc="upper right", fontsize=font_size - 4)  # bbox_to_anchor=(0.5, 0.5)

    # # # get the legend only from the last 'ax' (here is 'g')
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

    # fig.legend(handles, labels, title='Representation', bbox_to_anchor=(2, 1), loc='upper right', ncol=1)
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
    # plt.title('F1 Scores by category')
    # fig.suptitle("\n".join(["a big long suptitle that runs into the title"]*2), y=0.98, fontsize=font_size)
    fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
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
            # if appendix:
            #     if "GMM" in ",".join(show_detectors):
            #         plt.subplots_adjust(bottom=0.2)
            #     else:
            #         plt.subplots_adjust(bottom=0.10)
            # else:
            plt.subplots_adjust(bottom=0.13, top=0.95)
    except Warning as e:
        raise ValueError(e)

    # plt.legend(show_repres, loc='lower right')
    # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
    plt.savefig(out_file + '.pdf')  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file



def get_val(values_dict, diff_name = 'AUC', n_repeats=5):
    sub_dataset = []
    yerrs = []
    for k, vs in values_dict.items():
        data_name = k
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
        for (model_name, d), diff_vs in vs.items():
            d = float(d.split('_')[-1])
            v = diff_vs[diff_name].split('+/-')
            print(v)
            mean_, std_ = float(v[0]), float(v[1].split("(")[0])
            if diff_name == 'AUC':
                sub_dataset.append([k, f"{model_name}", d, mean_])
            elif diff_name in ['Training', 'Testing', 'Space']:
                sub_dataset.append([k, f"{model_name}", d, mean_])
            yerrs.append(std_ / np.sqrt(n_repeats)) #  # std/sqrt(n_repeats)

    return sub_dataset, yerrs, data_name

def show_compare(in_files, out_file='', title='auc', n_repeats = 5):

    font_size = 15

    # parse value from xlsx
    kjl_dict = parse_file(in_files[0])
    nystrom_dict = parse_file(in_files[1])

    if out_file == '':
        out_file = in_files[0] + '-res.txt'
    print(out_file)

    # show_detectors = []
    # sns.set_style("darkgrid")
    # # create plots

    # close preivous figs
    plt.close()

    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    colors = ['blue', 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
    labels = ['KJL', 'Nystrom']
    for j, values_dict in enumerate([kjl_dict, nystrom_dict]):
        sub_dataset, yerrs,  data_name = get_val(values_dict, diff_name='AUC', n_repeats=5)
        df = pd.DataFrame(sub_dataset, columns=[ 'dataset', 'model_name', 'd', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        plt.errorbar(df['d'].values, df['diff'].values, yerrs, capsize=3,  color=colors[j], ecolor='tab:red', label=labels[j])

    # fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
    plt.xlabel('Dimension')
    plt.ylabel('AUC')
    plt.title(f'{data_name}', fontsize = font_size-2)
    plt.legend(loc = 'upper right', fontsize =font_size-2)
    plt.ylim([0,1])
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


def show_train_sizes_seaborn(in_file, out_file='', title='auc', n_repeats = 5):
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
            fig, axes = plt.subplots(r, num_figs, figsize=(18, 5))  # (width, height)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0

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

    diff_name = 'AUC'
    for j in [1]:
        # fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        # print(f'{diff_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['dataset', 'model_name', 'diff'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.lineplot(y="diff", x='dataset', hue='model_name', data=df, ci=None,
                        capsize=.2, ax=axes[t, j % c])
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

        axes[t, j % c].errorbar(x=df['train_size'], y=df['AUC'],
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
        y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
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
    plt.savefig(out_file + '.pdf')  # should use before plt.show()
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})

    return out_file

def merge_ds_res(in_dir='speedup/out', directions=[('direction', 'src_dst'), ], datasets=None, models=None,
              feats=[('feat', 'iat_size'), ('feat', 'stats')], headers= [('is_header', True), ('is_header', False)],
              before_projs=None, gses=[('is_gs', True), ('is_gs', False)], ds=[('d_kjl', 5), ],
              covariances=[('covariance_type', 'full'), ('covariance_type', 'diag')]):

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
        (feat, is_header, is_before_proj, is_gs, d, covariance_type, model_name) = pth_cfg
        data = []
        for data_name in datasets:
            data_file = pth.join(in_dir, feat + "-header_" + str(is_header),
                                 data_name,
                                 "before_proj_" + str(is_before_proj) + "-gs_" + str(is_gs),
                                 model_name + f"-std_False_center_False-d_{str(d)}-{str(covariance_type)}",
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
                data.extend(np.array([[f'{data_name}|{data_file}', f'{model_name}|d_{d}', 'X_train_shape', 'X_test_shape', feat]]))
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

    def _merge_models(feat, is_header, is_before_proj, is_gs, ds, covariance_type,  dataset_name, model_name):
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
        vs = [] # store for csv
        for i, d_tup in enumerate(ds):
            _, d = d_tup
            if "OCSVM" in model_name:
                pth_cfg = (feat, is_header, is_before_proj, is_gs, d, None, model_name)
                data = _merge_datasets([dataset_name], pth_cfg)
                res_[d_tup] = data
            elif 'GMM' in model_name:
                pth_cfg = (feat, is_header, is_before_proj, is_gs, d, covariance_type, model_name)
                data = _merge_datasets([dataset_name], pth_cfg)
                res_[d_tup] = data
            else:
                msg = d_tup
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
                                    for model_name in [f'KJL-QS-GMM({covariance_type_tup[-1]})', f'Nystrom-QS-GMM({covariance_type_tup[-1]})']:
                                        out_file = _merge_models(feat_tup[1], header_tup[1], before_proj_tup[1],
                                                      gs_tup[1], ds,
                                                      covariance_type_tup[1],
                                                            dataset_name, model_name)
                                        in_files.append(out_file)
                                    show_compare(in_files, out_file='')  # compare KJL and Nystrom
                                except Exception as e:
                                    traceback.print_exc()
                                    lg.error(f"Error: {e}")

