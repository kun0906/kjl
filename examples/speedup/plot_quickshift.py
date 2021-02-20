from collections import OrderedDict

from kjl.utils.data import load_data
import numpy as np
import copy
import pandas as pd
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


def check(in_file = '.dat'):

    res = load_data(in_file)
    # print(res)
    best_res, mid_res = res

    # best1: (file_name),
    # outs1: [n_repeats][all_middle_res(tuning res)]
    best1, outs1 = mid_res

    best_avg_auc = -1
    for out in outs1[0]:
        if np.mean(out['auc']) > best_avg_auc:
            best_avg_auc = np.mean(out['auc'])
            best_results = copy.deepcopy(out)

    print('\n', best_avg_auc, best_results)

    return outs1

def _get_valid_clusters(tot_clusters_dict, thres = 10):

    n_clusters = 0
    for k, v in tot_clusters_dict.items():
        if v >= thres:
            n_clusters += 1

    return n_clusters


def parse_qs_res(qs_res, ks = [], kjl_q = 0.3):
    """

    Parameters
    ----------
    qs_res
        a list of 5 repeats, each repeat has all tuning results

    ks

    Returns
    -------

    """
    def _parse_each_repeat(repeat_res, ks, kjl_q=0.3, qs_beta=0.9):
        res_ = {}
        for k in ks:
            for i, vs in enumerate(repeat_res):
                params = vs['params']
                if params['kjl_q'] == kjl_q and params['quickshift_k'] == k and params['quickshift_beta'] == qs_beta:

                    tot_clusters_dict =  params['qs_res']['tot_n_clusters']
                    tot_n_clusters = len(tot_clusters_dict)
                    n_clusters = _get_valid_clusters(tot_clusters_dict, thres=10)

                    res_[(k,  qs_beta, kjl_q)] = {'auc': vs['auc'],
                                                 'tot_n_clusters': tot_n_clusters,
                                                 'n_clusters': n_clusters,
                                                  'top_20_n_clusters': 20 if n_clusters > 20 else n_clusters}  # 20 if params['qs_res']['n_clusters'] > 20 else params['qs_res']['n_clusters']

        return res_

    # def _parse_each_repeat(repeat_res, ks, qs_beta=0.9):
    #     res_ = {}
    #     for k in ks:
    #
    #         best_auc = -1
    #         for i, vs in enumerate(repeat_res):
    #             params = vs['params']
    #             if params['quickshift_k'] == k and params['quickshift_beta'] == qs_beta:
    #                 print(params['kjl_q'])
    #                 if vs['auc'] > best_auc:
    #                     best_auc = vs['auc']
    #                     kjl_q = params['kjl_q']
    #                     res_[(k,  qs_beta, kjl_q)] = {'auc': vs['auc'],
    #                                                  'tot_n_clusters': params['qs_res']['tot_n_clusters'],
    #                                                  'n_clusters':params['qs_res']['n_clusters'],
    #                                                   'top_20_n_clusters': 20 if params['qs_res']['n_clusters'] > 20 else params['qs_res']['n_clusters']}
    #
    #     return res_


    k_res = {}
    for idx_repeat in range(len(qs_res)):
        k_res[idx_repeat] = _parse_each_repeat(qs_res[idx_repeat], ks, kjl_q=kjl_q, qs_beta=0.9)
        # k_res[idx_repeat] = _parse_each_repeat(qs_res[idx_repeat], ks, qs_beta=0.9)

    return k_res

def plot_qs_k(qs_res, out_file ='.pdf', kjl_q=0.6):

    ks =  [10, 70, 140, 210, 280, 350, 420, 490, 560, 630, 700]
    k_res = parse_qs_res(qs_res, ks= ks, kjl_q=kjl_q)
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
    font_size = 20
    # s = min(len(show_repres), len(colors))
    new_data = OrderedDict()
    metrics = [ 'n_clusters', 'tot_n_clusters', 'top_20_n_clusters']     # 'auc',
    # metrics = ['n_clusters']
    for j, metric_name in enumerate(metrics):
        sub_dataset = []
        yerrs = []  # std/sqrt(n_repeats)
        # for idx_repeats, vs in k_res.items():
            # vs = {(qs_k, qs_beta, kjl_q): {'auc', 'tot_n_clusters', 'n_clusters'} }
        idx_repeats = 0
        vs = k_res[idx_repeats]
        for k in ks:
            v_ = vs[(k, 0.9, kjl_q)]
            sub_dataset.append([k,v_['auc'], v_['tot_n_clusters'], v_['n_clusters'], v_['top_20_n_clusters']])
            # yerrs.append(std_)

        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        print(f'{metric_name}: {sub_dataset}')
        df = pd.DataFrame(sub_dataset, columns=['qs_k', 'auc', 'tot_n_clusters', 'n_clusters', 'top_20_n_clusters'])
        # g = sns.barplot(y="diff", x='dataset', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
        print('yerrs:', yerrs)
        # colors = [ 'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.lineplot(y=metric_name, x='qs_k', hue=None, data=df, ci=None,marker="o",
                         ax=axes[t, j % c],  markers=True, dashes=False, linestyle=None)
        for _, (x_, y_) in enumerate( zip(df['qs_k'], df[metric_name])):

            if  metric_name != 'top_20_n_clusters':
                if x_ == 280 or x_ == 70 or x_ == 700 or x_ == 10:
                    axes[t, j % c].annotate((x_, y_), (x_, y_), xycoords='data',
                                            # xytext=(x_ , y_+0.1), #  textcoords='offset points',
                                            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
                                            fontsize=font_size - 4)
            else:
                if 'MAWI' in out_file:
                    if x_ == 280 or x_ == 70 or x_ == 700 or x_ == 10:
                        axes[t, j % c].annotate((x_, y_), (x_, y_), xycoords='data',
                                                # xytext=(x_ , y_+0.1), #  textcoords='offset points',
                                                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
                                                fontsize=font_size - 4)
                else:
                    if x_ == 280 or x_ == 70 or x_ == 700:
                        axes[t, j % c].annotate((x_, y_), (x_, y_), xycoords='data',
                                                # xytext=(x_ , y_+0.1), #  textcoords='offset points',
                                                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
                                                fontsize=font_size - 4)
                if 'MAWI' in out_file and x_ == 140:
                    axes[t, j % c].annotate((x_, y_), (x_, y_),  # xytext=(x_ + 0.1, y_+0.5),
                                            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
                                            fontsize=font_size - 4)
                if 'CTU' in out_file and x_ == 350:
                    axes[t, j % c].annotate((x_, y_), (x_, y_),  # xytext=(x_ + 0.1, y_+0.5),
                                            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
                                            fontsize=font_size - 4)
                if 'SFRIG' in out_file and x_==490:
                    axes[t, j % c].annotate((x_, y_), (x_, y_),  # xytext=(x_ + 0.1, y_+0.5),
                                        arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"), fontsize = font_size-4)
        # linestyle = , '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
        ys = []
        xs = []
        width = 0

        # axes[t, j % c].errorbar(x=xs + width / 2, y=ys,
        #                         yerr=yerrs, fmt='none', c='b', capsize=3)

        # g.set(xlabel=detector_name)       # set name at the bottom
        # g.set(xlabel=None)
        # g.set(ylabel=None)
        # g.set_ylim(-1, 1)

        if metric_name =='n_clusters':
            title = 'Number of clusters (after removing small clusters)'
        elif metric_name == 'tot_n_clusters':
            title = 'Number of total clusters'
        elif metric_name == 'top_20_n_clusters':
            title = 'Top 20 clusters (after removing small clusters)'
        g.set_title(title, fontsize=font_size+4)

        g.set_xlabel('K for quickshift', fontsize=font_size + 4)
        g.set_ylabel('Num. of clusters', fontsize=font_size + 4)
        # if appendix:
        #     if j < len(show_detectors) - 1:
        #         g.set_xticklabels([])
        #     else:
        #         g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # else:
        #     g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=0, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        g.set_xticks(df['qs_k'])
        # # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        # # y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        # # v_max = max([ int(np.ceil(v)) for v in g.get_yticks()])
        # # step = v_max/5
        # # step = int(step) if step > 1 else step
        # # y_v = [v for v in range(0, v_max, step)]
        # if 'AUC' in diff_name:
        #     y_v = [0.0, 0.5, 1.0, 1.5, 2.0]
        # elif 'train' in diff_name:
        #     y_v = [0.0, 0.5, 1.0, 1.5]
        # elif 'test' in diff_name:
        #     y_v = [0, 25, 50, 75]
        # elif 'space' in diff_name:
        #     y_v = [0, 10, 20, 30]
        #
        # g.set_yticks(y_v)  # set value locations in y axis
        # # g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
        # #                   fontsize=font_size + 6)  # set the number of each value in y axis
        # g.set_yticklabels(y_v, fontsize=font_size + 4)  # set the number of each value in y axis
        g.set_yticklabels([int(v) for v in g.get_yticks()], fontsize=font_size + 2)  # set the number of each value in y axis
        g.set_xticklabels([int(v) for v in g.get_xticks()], fontsize=font_size + 2)  # set the number of each value in x axis
        print('yticks', g.get_yticks())
        # # if j % c != 0:
        #     # g.get_yaxis().set_visible(False)
        #     g.set_yticklabels(['' for v_tmp in y_v])
        #     g.set_ylabel('')

        # g.set_title(diff_name, fontsize=font_size + 8)
        # # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        # # g.get_legend().set_visible(True)
        # handles, labels = g.get_legend_handles_labels()
        # axes[t, j % c].legend(handles, labels, loc="upper right", fontsize=font_size - 4)  # bbox_to_anchor=(0.5, 0.5)

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
    axes[-1, -1].set_axis_off()
    # # plt.xlabel('Catagory')
    # plt.ylabel('AUC')
    # plt.ylim(0,1.07)
    # plt.title('F1 Scores by category')
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
    # sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})



def main():


    # MAWI (best_q =0.6 when use KJL-GMM and tune q form [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] and n_repeats =1)
    in_file = 'speedup/out-kjl-tune_qs_k/src_dst/iat_size-header_False/MAWI1_2020/before_proj_False-gs_True/KJL-QS-GMM(full)-std_False_center_False-d_5-full/res.dat'
    res = check(in_file)
    plot_qs_k(res, out_file=in_file+'-.pdf', kjl_q =0.6)

    # CTU (best_q =0.4 when use KJL-GMM and tune q form [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] and n_repeats =1)
    in_file = 'speedup/out-kjl-tune_qs_k/src_dst/iat_size-header_False/CTU1/before_proj_False-gs_True/KJL-QS-GMM(full)-std_False_center_False-d_5-full/res.dat'
    res = check(in_file)
    plot_qs_k(res, out_file=in_file + '-.pdf', kjl_q=0.4)

    # SFRIG1_2020 (best_q =0.9 when use KJL-GMM and tune q form [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] and n_repeats =1)
    in_file = 'speedup/out-kjl-tune_qs_k/src_dst/iat_size-header_False/SFRIG1_2020/before_proj_False-gs_True/KJL-QS-GMM(full)-std_False_center_False-d_5-full/res.dat'
    res = check(in_file)
    plot_qs_k(res, out_file=in_file + '-.pdf', kjl_q=0.9)

    # UNB345_3 (best_q =0.3 when use KJL-GMM and tune q form [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] and n_repeats =1)
    in_file = 'speedup/out-kjl-tune_qs_k/src_dst/iat_size-header_False/UNB345_3/before_proj_False-gs_True/KJL-QS-GMM(full)-std_False_center_False-d_5-full/res.dat'
    res = check(in_file)
    plot_qs_k(res, out_file=in_file + '-.pdf', kjl_q=0.3)



if __name__ == '__main__':
    main()