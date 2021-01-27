import os
import traceback
from collections import OrderedDict

# from pdf2image import convert_from_bytes
from PIL import Image
import pickle

from kjl.utils.tool import load_data
import numpy as np
import matplotlib.pyplot as plt
import os.path as pth

plt.ioff()
import matplotlib as mpl
from matplotlib import rcParams
# print(mpl.is_interactive())
import time
import pandas as pd


def plot_individul_result(result, out_file, fixed_U_size=None, n_point=None):
    # only show the first one
    for i, (dataset_name, v) in enumerate(result.items()):
        if i == 0:
            result = v
            # for j,  (k_case, v1) in enumerate(v.items()):
            #     result = v1
            break

    import matplotlib.pyplot as plt

    def plot_data(ax, x, y, y_err, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):

        # with plt.style.context(('ggplot')):
        # fig, ax = plt.subplots()
        # ax.plot(x, y, '*-', alpha=0.9, label=label)
        # # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
        ax.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='b',
                    label=label)  # marker='*',

        # # plt.xlim([0.0, 1.0])
        # if len(ylim) == 2:
        #     plt.ylim(ylim)  # [0.0, 1.05]
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # # plt.xticks(x)
        # # plt.yticks(y)
        # plt.legend(loc='lower right')
        # plt.title(title)
        #
        # # should use before plt.show()
        # plt.savefig(out_file)

        # plt.show()

    def plot_data2(xs, ys, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):

        # with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        for (x, y) in zip(xs, ys):
            ax.plot(x, y, '*-', alpha=0.9, label=label)
        # ax.plot(x, y, '*-', alpha=0.9)
        # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            plt.ylim(ylim)  # [0.0, 1.05]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        plt.legend(loc='lower right')
        plt.title(title)

        # should use before plt.show()
        plt.savefig(out_file)

        plt.show()

    def plot_times(ax, y_dict, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):

        names = y_dict.keys()
        for name in names:
            # y = [v[name][0] for v in y_dict]
            # y_err = [v[name][1] for v in y_dict]
            y = y_dict[name][0]
            y_err = y_dict[name][1]
            x = range(len(y))
            # with plt.style.context(('ggplot')):
            # ax.plot(x, y, '*-', alpha=0.9, label=name)
            # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
            # plt.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='m',
            #              label=name)
            ax.errorbar(x, y, yerr=y_err, capsize=2, linestyle='-', marker='*', markeredgecolor='m',
                        label=name)
            break

        # # plt.xlim([0.0, 1.0])
        # if len(ylim) == 2:
        #     plt.ylim(ylim)  # [0.0, 1.05]
        # ax.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # # plt.xticks(x)
        # # plt.yticks(y)
        # plt.legend(loc='lower right')
        # plt.title(title)
        #
        # # should use before plt.show()
        # plt.savefig(out_file)

        # plt.show()

    for key_case, v in result.items():
        best_results_arr = [best_res for (best_res, mid_res) in v]
        middle_results_arr = [mid_res for (best_res, mid_res) in v]

        train_times_arr = []
        test_times_arr = []
        n_components_arr = []
        aucs_arr = []
        abnormal_threses_arr = []
        for i, v in enumerate(best_results_arr):
            if i == 0:
                best_results = v

            train_times_arr.append(v['train_times'])
            test_times_arr.append(v['test_times'])
            n_components_arr.append(np.asarray([_v['n_components'] for _v in v['model_params']]))
            aucs_arr.append(np.asarray(v['aucs']))
            abnormal_threses_arr.append(np.asarray(v['abnormal_threses']))

        # # train_times_arr = np.asarray(train_times_arr)
        # # test_times_arr = np.asarray(test_times_arr)
        train_times_dict = {}
        for (k, v) in best_results['train_times'][0].items():
            tmp = []
            for best_res in train_times_arr:
                tmp.append(np.asarray([_v[k] for _v in best_res]))
            tmp = np.asarray(tmp)
            train_times_dict[k] = (np.mean(tmp, axis=0), np.std(tmp, axis=0))

        test_times_dict = {}
        for (k, v) in best_results['test_times'][0].items():
            tmp = []
            for best_res in test_times_arr:
                tmp.append(np.asarray([_v[k] for _v in best_res]))
            tmp = np.asarray(tmp)
            test_times_dict[k] = (np.mean(tmp, axis=0), np.std(tmp, axis=0))

        n_components_arr = np.asarray(n_components_arr)
        aucs_arr = np.asarray(aucs_arr)
        abnormal_threses_arr = np.asarray(abnormal_threses_arr)

        # train_times = np.mean(train_times_arr, axis=0)
        # train_times_std = np.std(train_times_arr, axis=0)
        #
        # test_times = np.mean(test_times_arr, axis=0)
        # test_times_std = np.std(test_times_arr, axis=0)

        n_components = np.mean(n_components_arr, axis=0)
        n_components_std = np.std(n_components_arr, axis=0)

        aucs = np.mean(aucs_arr, axis=0)
        aucs_std = np.std(aucs_arr, axis=0)

        abnormal_threses = np.mean(abnormal_threses_arr, axis=0)
        abnormal_threses_std = np.std(abnormal_threses_arr, axis=0)

        params = best_results['params']
        print(f'\n***{dataset_name}, {key_case}')
        if 'online:False' in key_case:
            online = False
            params.incorporated_points = 0
            params.fixed_U_size = False
        else:
            online = True

        if n_point is None:
            n_point = params.incorporated_points
            n_point = f'{n_point} datapoints' if n_point > 1 else f'{n_point} datapoint'

        fixed_U_size = params.fixed_U_size
        n_components_init = params.n_components
        covariance_type = params.covariance_type
        q_kjl = params.q_kjl
        n_kjl = params.n_kjl
        d_kjl = params.d_kjl
        kjl = params.kjl
        n_repeats = params.n_repeats
        if kjl:
            title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}; n_repeats={n_repeats}'
            if fixed_U_size:
                title = f'online GMM with fixed KJL (incorporated {n_point})\n{title}' if online else f'Batch GMM\n({title})'
            else:
                title = f'online GMM with unfixed KJL (incorporated {n_point})\n{title}' if online else f'Batch GMM\n({title})'
        else:
            title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}; n_repeats={n_repeats}'
            title = f'online GMM\n({title})' if online else f'Batch GMM\n({title})'

        batch_size = params.batch_size
        xlabel = f'The ith batch: i * batch_size({batch_size}) datapoints'
        # if not online:
        #     # y = train_times
        #     # y_err = train_times_dict
        #     # plot_data(range(len(y)), y, y_err, xlabel=xlabel, ylabel='Training time (s)', title=title,
        #     #           out_file=out_file.replace('.pdf', '-train_times.pdf'))
        #     # y = test_times
        #     # y_err = test_times_std
        #     # plot_data(range(len(y)), y, y_err, xlabel=xlabel, ylabel='Testing time (s)', title=title,
        #     #           out_file=out_file.replace('.pdf', '-test_times.pdf'))
        #
        #     # y = [v['n_components'] for v in best_results['model_params']]
        #     y= n_components
        #     y_err=n_components_std
        #     plot_data(range(len(y)), y, y_err, xlabel=xlabel, ylabel='n_components', title=title,
        #               out_file=out_file.replace('.pdf', '-n_components.pdf'))
        #
        # else:
        # # ys = best_results['train_times']

        fig, ax = plt.subplots(nrows=2, ncols=2)

        y = train_times_dict
        plot_times(ax[0, 0], y, xlabel=xlabel, ylabel='Training time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-train_times.pdf'))

        # ys = best_results['test_times']
        y = test_times_dict
        plot_times(ax[0, 1], y, xlabel=xlabel, ylabel='Testing time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-test_times.pdf'))

        # y = best_results['aucs']
        y = aucs
        y_err = aucs_std
        plot_data(ax[1, 0], range(len(y)), y, y_err, xlabel=xlabel, ylabel='AUC', ylim=[0.0, 1.05], title=title,
                  out_file=out_file.replace('.pdf', '-aucs.pdf'))

        # y = [v['n_components'] for v in best_results['model_params']]
        y = n_components
        y_err = n_components_std
        plot_data(ax[1, 1], range(len(y)), y, y_err, xlabel=xlabel, ylabel='n_components', title=title,
                  out_file=out_file.replace('.pdf', '-n_components.pdf'))

        # y = best_results['novelty_threses']
        # plot_data(range(len(y)), y, xlabel=xlabel, ylabel='novelty threshold', title=title,
        #           out_file=out_file.replace('.pdf', '-novelty.pdf'))

        # y = best_results['abnormal_threses']
        y = abnormal_threses
        y_err = abnormal_threses_std
        q_thres = params.q_abnormal_thres
        plot_data(plt, range(len(y)), y, y_err, xlabel=xlabel, ylabel='abnormal threshold', title=title,
                  label=f'{q_thres} quantile of normal scores',
                  out_file=out_file.replace('.pdf', '-abnormal.pdf'))

        # y1 = best_results['novelty_threses']
        # y2 = best_results['abnormal_threses']
        # xs = [range(len(y1)), range(len(y2))]
        # ys = [y1, y2]
        # plot_data2(xs, ys, xlabel=xlabel, ylabel='Threshold', title=title,
        #            out_file=out_file.replace('.pdf', '-threshold.pdf'))

        plt.show()


def _plot_each_result(results, k_dataset=(), experiment_case='', out_file=''):
    def plot_data(ax, x, y, y_err, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):
        ax.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='b',
                    label=label)  # marker='*',

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            ax.set_ylim(ylim)  # [0.0, 1.05]
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        ax.legend(loc='lower right')
        # ax.set_title(title)

    def plot_times(ax, y_dict, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):
        names = y_dict.keys()
        for name in names:
            y = y_dict[name][0]
            y_err = y_dict[name][1]
            x = range(len(y))
            # with plt.style.context(('ggplot')):
            # ax.plot(x, y, '*-', alpha=0.9, label=name)
            # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
            # plt.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='m',
            #              label=name)
            ax.errorbar(x, y, yerr=y_err, capsize=2, linestyle='-', marker='*', markeredgecolor='m',
                        label=name)
            break

        # # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            ax.set_ylim(ylim)  # [0.0, 1.05]
        # ax.xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        ax.legend(loc='upper right')
        # ax.set_title(title)
        #
        # # should use before plt.show()
        # plt.savefig(out_file)

        # plt.show()


def get_average(_result):
    average_result = {''}

    best_results_arr = [best_res for (best_res, mid_res) in _result]
    middle_results_arr = [mid_res for (best_res, mid_res) in _result]

    train_times_arr = []
    test_times_arr = []
    n_components_arr = []
    aucs_arr = []
    abnormal_threses_arr = []
    for i, v in enumerate(best_results_arr):
        if i == 0:
            best_results = v

        train_times_arr.append(v['train_times'])
        test_times_arr.append(v['test_times'])
        n_components_arr.append(np.asarray([_v['n_components'] for _v in v['model_params']]))
        aucs_arr.append(np.asarray(v['aucs']))
        abnormal_threses_arr.append(np.asarray(v['abnormal_threses']))

    # # train_times_arr = np.asarray(train_times_arr)
    # # test_times_arr = np.asarray(test_times_arr)
    train_times_dict = {}
    for (k, v) in best_results['train_times'][0].items():  # {}
        tmp = []
        for best_res in train_times_arr:
            tmp.append(np.asarray([_v[k] for _v in best_res]))
        tmp = np.asarray(tmp)
        train_times_dict[k] = (np.mean(tmp, axis=0), np.std(tmp, axis=0))

    test_times_dict = {}
    for (k, v) in best_results['test_times'][0].items():
        tmp = []
        for best_res in test_times_arr:
            tmp.append(np.asarray([_v[k] for _v in best_res]))
        tmp = np.asarray(tmp)
        test_times_dict[k] = (np.mean(tmp, axis=0), np.std(tmp, axis=0))

    n_components_arr = np.asarray(n_components_arr)
    aucs_arr = np.asarray(aucs_arr)
    abnormal_threses_arr = np.asarray(abnormal_threses_arr)

    # train_times = np.mean(train_times_arr, axis=0)
    # train_times_std = np.std(train_times_arr, axis=0)
    #
    # test_times = np.mean(test_times_arr, axis=0)
    # test_times_std = np.std(test_times_arr, axis=0)

    n_components = np.mean(n_components_arr, axis=0)
    n_components_std = np.std(n_components_arr, axis=0)

    aucs = np.mean(aucs_arr, axis=0)
    aucs_std = np.std(aucs_arr, axis=0)

    # abnormal_threses = np.mean(abnormal_threses_arr, axis=0)
    # abnormal_threses_std = np.std(abnormal_threses_arr, axis=0)

    # average_result = {'train_times': train_times_dict['preprocessing_time'], 'test_times': test_times_dict['preprocessing_time'],
    #                   'aucs': (aucs, aucs_std), 'n_components': (n_components, n_components_std)}
    # average_result = {'train_times': train_times_dict['model_fitting_time'], 'test_times': test_times_dict['prediction_time'],
    #                   'aucs': (aucs, aucs_std), 'n_components': (n_components, n_components_std)}
    # average_result = {'train_times': train_times_dict['rescore_time'], 'test_times': test_times_dict['auc_time'],
    #                   'aucs': (aucs, aucs_std), 'n_components': (n_components, n_components_std)}
    average_result = {'train_times': train_times_dict['train_time'], 'test_times': test_times_dict['test_time'],
                      'aucs': (aucs, aucs_std), 'n_components': (n_components, n_components_std)}

    return average_result


def plot_data(ax, x, y_batch, y_err_batch, y_online, y_err_online, xlabel='range', ylabel='auc', ylim=[], title='',
              out_file='', label='', legend_position='upper right'):
    # x = range(y_batch)
    ax.errorbar(x, y_batch, yerr=y_err_batch, ecolor='r', capsize=2, linestyle='-', marker='.', markeredgecolor='g',
                label=f'Batch', alpha=0.9)  # marker='*',
    ax.errorbar(x, y_online, yerr=y_err_online, ecolor='r', capsize=2, linestyle=':', marker='.', markeredgecolor='b',
                label=f'Online', alpha=0.9)  # marker='*',

    # plt.xlim([0.0, 1.0])
    if len(ylim) == 2:
        ax.set_ylim(ylim)  # [0.0, 1.05]
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.xticks(x)
    # plt.yticks(y)
    ax.legend(loc=legend_position)
    # ax.set_title(title)


def plot_each_result(results, k_dataset=(), out_file=''):
    for param_str, _result in results.items():
        if 'online:False' in param_str:
            result_batch_GMM = get_average(_result)
        elif 'online:True' in param_str:
            # result_batch_GMM = get_average(_result)
            result_online_GMM = get_average(_result)
            experiment_case = param_str
            params = _result[0][0]['params']  # _result[n_repeats] = {best_results, middle_result}
        else:
            msg = param_str
            raise ValueError(msg)

    # params = best_results['params']
    # print(f'\n***{k_dataset}, {experiment_case}')
    if 'online:False' in experiment_case:
        online = False
        params.incorporated_points = 0
        params.fixed_U_size = False
    else:
        online = True

    n_point = params.incorporated_points
    n_point = f'{n_point}' if n_point > 1 else f'{n_point}'

    fixed_kjl = params.fixed_kjl
    fixed_U_size = params.fixed_U_size
    n_components_init = params.n_components
    covariance_type = params.covariance_type
    q_kjl = params.q_kjl
    n_kjl = params.n_kjl
    d_kjl = params.d_kjl
    std = params.std
    kjl = params.kjl
    n_repeats = params.n_repeats
    dataset_name, data_file = k_dataset
    dataset_name = f'{dataset_name} (init_ratio={int(params.percent_first_init * 100)}:{int(round((1 - params.percent_first_init) * 100))})'
    # if kjl:
    #     title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}; n_repeats={n_repeats}'
    #     if fixed_kjl:
    #         title = f'online GMM with fixed KJL on {dataset_name}\n{title}' if online else f'Batch GMM with fixed KJL on {dataset_name}\n({title})'
    #     elif fixed_U_size:
    #         # (replace {n_point} cols and rows of U)
    #         title = f'online GMM with fixed U size on {dataset_name}\n{title}' if online else f'Batch GMM on {dataset_name}\n({title})'
    #     else: # increased_U
    #         title = f'online GMM with unfixed KJL (incorporated {n_point}) on {dataset_name}\n{title}' if online else f'Batch GMM on {dataset_name}\n({title})'
    # else:
    #     title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}; n_repeats={n_repeats}'
    #     title = f'online GMM on {dataset_name}\n({title})' if online else f'Batch GMM on {dataset_name}\n({title})'

    if kjl:
        title = f'n_comp={n_components_init}, {covariance_type}; std={std}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}; n_repeats={n_repeats}'
        if fixed_kjl:
            title = f'Batch GMM vs. Online GMM with fixed KJL on {dataset_name}\n{title}'
        elif fixed_U_size:
            # (replace {n_point} cols and rows of U)
            title = f'Batch GMM vs. Online GMM with fixed U size on {dataset_name}\n{title}'
        else:  # increased_U
            title = f'Batch GMM vs. Online GMM with unfixed KJL on {dataset_name}\n{title}'
    else:
        title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}; n_repeats={n_repeats}'
        title = f'Batch GMM vs. Online GMM on {dataset_name}\n({title})'

    batch_size = params.batch_size
    xlabel = f'The ith batch: i * batch_size({batch_size}) datapoints'

    fig, ax = plt.subplots(nrows=2, ncols=2)
    x = range(len(result_online_GMM['aucs'][0]))
    y_batch, y_err_batch = result_batch_GMM['train_times']
    y_online, y_err_online = result_online_GMM['train_times']
    # plot_times(ax[0,0],x, y_batch_dict, y_online_dict, xlabel=xlabel, ylabel='Training time (s)', title=title,
    #            out_file=out_file.replace('.pdf', '-train_times.pdf'))
    plot_data(ax[0, 0], x, y_batch, y_err_batch, y_online, y_err_online, xlabel=xlabel, ylabel='Training time (s)',
              title=title,
              label='Train_time (s)', legend_position='upper right',
              out_file=out_file.replace('.pdf', '-train_times.pdf'))

    y_batch, y_err_batch = result_batch_GMM['test_times']
    y_online, y_err_online = result_online_GMM['test_times']
    # plot_times(ax[0,1],x, y_batch_dict, y_online_dict, xlabel=xlabel, ylabel='Testing time (s)', title=title,
    #            out_file=out_file.replace('.pdf', '-test_times.pdf'))
    plot_data(ax[0, 1], x, y_batch, y_err_batch, y_online, y_err_online, xlabel=xlabel,
              ylabel='Testing time (s)', title=title, label='Test_time (s)', legend_position='upper right',
              out_file=out_file.replace('.pdf', '-test_times.pdf'))

    y_batch, y_err_batch = result_batch_GMM['aucs']
    y_online, y_err_online = result_online_GMM['aucs']
    plot_data(ax[1, 0], x, y_batch, y_err_batch, y_online, y_err_online, xlabel=xlabel, ylabel='AUC', ylim=[0.0, 1.05],
              title=title,
              label='AUC', legend_position='lower right',
              out_file=out_file.replace('.pdf', '-aucs.pdf'))

    y_batch, y_err_batch = result_batch_GMM['n_components']
    y_online, y_err_online = result_online_GMM['n_components']
    plot_data(ax[1, 1], x, y_batch, y_err_batch, y_online, y_err_online, xlabel=xlabel, ylabel='n_components',
              title=title,
              label='n_components', legend_position='lower right',
              out_file=out_file.replace('.pdf', '-n_components.pdf'))

    fig.suptitle(title, fontsize=11)

    plt.tight_layout()  # rect=[0, 0, 1, 0.95]
    try:
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.975, left=0.12)
    except Warning as e:
        raise ValueError(e)
    #
    # fig.text(.5, 15, "total label", ha='center')
    plt.figtext(0.5, 0.01, f'X-axis:({xlabel})', fontsize=11, va="bottom", ha="center")
    print(out_file)
    fig.savefig(out_file, format='pdf', dpi=300)
    plt.show()
    plt.close(fig)

    # sns.reset_orig()
    # sns.reset_defaults()
    # rcParams.update({'figure.autolayout': True})


def plot_result(result, out_file, fixed_U_size=None, n_point=None):
    # only show the first one
    for i, (k_dataset, v) in enumerate(result.items()):
        if i == 0:
            result = v
            break

    def plot_data(ax, x, y, y_err, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):
        ax.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='b',
                    label=label)  # marker='*',

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            ax.set_ylim(ylim)  # [0.0, 1.05]
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        ax.legend(loc='lower right')
        # ax.set_title(title)

    def plot_times(ax, y_dict, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):
        names = y_dict.keys()
        for name in names:
            y = y_dict[name][0]
            y_err = y_dict[name][1]
            x = range(len(y))
            # with plt.style.context(('ggplot')):
            # ax.plot(x, y, '*-', alpha=0.9, label=name)
            # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
            # plt.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='m',
            #              label=name)
            ax.errorbar(x, y, yerr=y_err, capsize=2, linestyle='-', marker='*', markeredgecolor='m',
                        label=name)
            break

        # # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            ax.set_ylim(ylim)  # [0.0, 1.05]
        # ax.xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        ax.legend(loc='upper right')
        # ax.set_title(title)
        #
        # # should use before plt.show()
        # plt.savefig(out_file)

        # plt.show()

    for key_case, v in result.items():
        best_results_arr = [best_res for (best_res, mid_res) in v]
        middle_results_arr = [mid_res for (best_res, mid_res) in v]

        train_times_arr = []
        test_times_arr = []
        n_components_arr = []
        aucs_arr = []
        abnormal_threses_arr = []
        for i, v in enumerate(best_results_arr):
            if i == 0:
                best_results = v

            train_times_arr.append(v['train_times'])
            test_times_arr.append(v['test_times'])
            n_components_arr.append(np.asarray([_v['n_components'] for _v in v['model_params']]))
            aucs_arr.append(np.asarray(v['aucs']))
            abnormal_threses_arr.append(np.asarray(v['abnormal_threses']))

        # # train_times_arr = np.asarray(train_times_arr)
        # # test_times_arr = np.asarray(test_times_arr)
        train_times_dict = {}
        for (k, v) in best_results['train_times'][0].items():
            tmp = []
            for best_res in train_times_arr:
                tmp.append(np.asarray([_v[k] for _v in best_res]))
            tmp = np.asarray(tmp)
            train_times_dict[k] = (np.mean(tmp, axis=0), np.std(tmp, axis=0))

        test_times_dict = {}
        for (k, v) in best_results['test_times'][0].items():
            tmp = []
            for best_res in test_times_arr:
                tmp.append(np.asarray([_v[k] for _v in best_res]))
            tmp = np.asarray(tmp)
            test_times_dict[k] = (np.mean(tmp, axis=0), np.std(tmp, axis=0))

        n_components_arr = np.asarray(n_components_arr)
        aucs_arr = np.asarray(aucs_arr)
        abnormal_threses_arr = np.asarray(abnormal_threses_arr)

        # train_times = np.mean(train_times_arr, axis=0)
        # train_times_std = np.std(train_times_arr, axis=0)
        #
        # test_times = np.mean(test_times_arr, axis=0)
        # test_times_std = np.std(test_times_arr, axis=0)

        n_components = np.mean(n_components_arr, axis=0)
        n_components_std = np.std(n_components_arr, axis=0)

        aucs = np.mean(aucs_arr, axis=0)
        aucs_std = np.std(aucs_arr, axis=0)

        abnormal_threses = np.mean(abnormal_threses_arr, axis=0)
        abnormal_threses_std = np.std(abnormal_threses_arr, axis=0)

        params = best_results['params']
        print(f'\n***{k_dataset}, {key_case}')
        if 'online:False' in key_case:
            online = False
            params.incorporated_points = 0
            params.fixed_U_size = False
        else:
            online = True

        if n_point is None:
            n_point = params.incorporated_points
            n_point = f'{n_point}' if n_point > 1 else f'{n_point}'

        fixed_kjl = params.fixed_kjl
        fixed_U_size = params.fixed_U_size
        n_components_init = params.n_components
        covariance_type = params.covariance_type
        q_kjl = params.q_kjl
        n_kjl = params.n_kjl
        d_kjl = params.d_kjl
        kjl = params.kjl
        n_repeats = params.n_repeats
        dataset_name, data_file = k_dataset
        dataset_name = f'{dataset_name} (init_ratio={int(params.percent_first_init * 100)}:{int(round((1 - params.percent_first_init) * 100))})'
        if kjl:
            title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}; n_repeats={n_repeats}'
            if fixed_kjl:
                title = f'online GMM with fixed KJL on {dataset_name}\n{title}' if online else f'Batch GMM with fixed KJL on {dataset_name}\n({title})'
            elif fixed_U_size:
                # (replace {n_point} cols and rows of U)
                title = f'online GMM with fixed U size on {dataset_name}\n{title}' if online else f'Batch GMM on {dataset_name}\n({title})'
            else:  # increased_U
                title = f'online GMM with unfixed KJL (incorporated {n_point}) on {dataset_name}\n{title}' if online else f'Batch GMM on {dataset_name}\n({title})'
        else:
            title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}; n_repeats={n_repeats}'
            title = f'online GMM on {dataset_name}\n({title})' if online else f'Batch GMM on {dataset_name}\n({title})'

        batch_size = params.batch_size
        xlabel = f'The ith batch: i * batch_size({batch_size}) datapoints'

        fig, ax = plt.subplots(nrows=2, ncols=2)

        y = train_times_dict
        plot_times(ax[0, 0], y, xlabel=xlabel, ylabel='Training time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-train_times.pdf'))

        # ys = best_results['test_times']
        y = test_times_dict
        plot_times(ax[0, 1], y, xlabel=xlabel, ylabel='Testing time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-test_times.pdf'))

        # y = best_results['aucs']
        y = aucs
        y_err = aucs_std
        plot_data(ax[1, 0], range(len(y)), y, y_err, xlabel=xlabel, ylabel='AUC', ylim=[0.0, 1.05], title=title,
                  label='AUC', out_file=out_file.replace('.pdf', '-aucs.pdf'))

        # y = [v['n_components'] for v in best_results['model_params']]
        y = n_components
        y_err = n_components_std
        plot_data(ax[1, 1], range(len(y)), y, y_err, xlabel=xlabel, ylabel='n_components', title=title,
                  label='n_components', out_file=out_file.replace('.pdf', '-n_components.pdf'))

        fig.suptitle(title, fontsize=11)

        plt.tight_layout()  # rect=[0, 0, 1, 0.95]
        try:
            plt.subplots_adjust(top=0.9, bottom=0.1, right=0.975, left=0.08)
        except Warning as e:
            raise ValueError(e)
        #
        # fig.text(.5, 15, "total label", ha='center')
        plt.figtext(0.5, 0.01, f'X-axis:({xlabel})', fontsize=11, va="bottom", ha="center")
        print(out_file)
        fig.savefig(out_file, format='pdf', dpi=300)
        plt.show()
        plt.close(fig)

        # sns.reset_orig()
        # sns.reset_defaults()
        # rcParams.update({'figure.autolayout': True})


def write_imgs2excel(worksheet, img_pth, position):
    """
    https://stackoverflow.com/questions/51601031/python-writing-images-and-dataframes-to-the-same-excel-file/51608720
    Parameters
    ----------
    worksheet
    img_pth
    position

    Returns
    -------

    """
    # Insert an image.
    # worksheet.insert_image('D3', 'logo.png')
    worksheet.insert_image(position, img_pth)


def _imgs2xls(xls_pth, res):
    """
    https://stackoverflow.com/questions/33672833/set-width-and-height-of-an-image-when-inserting-via-worksheet-insert-image
    Parameters
    ----------
    xls_pth
    img_lst

    Returns
    -------

    """
    # Create a Pandas dataframe from some data.
    df = pd.DataFrame({'Data': [10]})

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(xls_pth, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    row_start = 3
    col_start = 3
    for ratio, img_lst in res.items():
        # Insert an image.
        for i, img in enumerate(img_lst):
            print(i, img, os.path.exists(img))
            # worksheet.insert_image('D3', 'logo.png')
            if img is None or not os.path.exists(img):
                worksheet.insert_image(row_start, i + col_start, '')
            else:
                im = Image.open(img)
                image_width, image_height = im.size

                # image_width = 140.0
                # image_height = 182.0

                cell_width = 150.0
                cell_height = 100.0

                x_scale = cell_width / image_width
                y_scale = cell_height / image_height

                worksheet.insert_image(row_start, i + col_start, img, {'x_scale': x_scale, 'y_scale': y_scale})
                worksheet.set_column(1, 100, width=cell_width /5)
                worksheet.set_row(row_start, height=cell_height)

        row_start += 1
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def main(online=True):
    # out_dir = 'out/data/data_reprst/pcaps/mimic_GMM_dataset'
    # results = load_data( f'{out_dir}/all-online_{online}-results.dat')
    out_dir = 'out'
    out_file = f'{out_dir}/all-results.dat'
    out_file = f'{out_dir}/data/feats/mimic_GMM_dataset/Xy-normal-abnormal.dat-case0.dat'
    results = load_data(out_file)

    ##########################################################################################
    out_file = f'{out_dir}/online_{online}.pdf'
    # # Todo: format the results
    # plot_result(results, out_file, fixed_U_size=True, n_point=None)
    plot_each_result(results, k_dataset=('mimic_GMM', 'path'), out_file=out_file)
    print("\n\n---finish succeeded!")


def _pdf2img(pdf_pth, img_pth):
    try:
        # print(pdf_pth, os.path.exists(pdf_pth))
        """
        install pdf2image, poppler
        https://github.com/Belval/pdf2image
        https://stackoverflow.com/questions/54990306/how-to-write-pandas-dataframe-and-insert-image-or-chart-into-the-same-excel-wor
        """
        from pdf2image import convert_from_path, convert_from_bytes
        # pages = convert_from_path(pdf_pth, 500)
        pages = convert_from_bytes(open(pdf_pth, 'rb').read())
        for page in pages:
            page.save(img_pth, 'JPEG')
            break
    except:
        traceback.print_exc()
        img_pth = None
    return img_pth


def find_img(out_dataname, out_ratio, out_key, root_dir, outs):
    img = ''
    for i, v1 in enumerate(outs):
        for (dataset_name, data_file), v2 in v1.items():
            for ratio, v3 in v2.items():
                for experiment_case, v4 in v3.items():
                    # online = v3['online']
                    # batch = v3['batch']
                    p = v4['online'][0]['params']
                    sub_dir = f'gs={p.gs}-std={p.std}_center={p.with_means_std}-' \
                              f'kjl={p.kjl}-d_kjl={p.d_kjl}-n_kjl={p.n_kjl}-c_kjl={p.centering_kjl}-' \
                              f'ms={p.meanshift}-before_kjl={p.before_kjl_meanshift}-fixed_kjl={p.fixed_kjl}-seed={p.random_state}'
                    tmp_dir = f'{root_dir}/out/{pth.dirname(data_file)}/{sub_dir}'
                    tmp_dir_key =  f'{root_dir}/out/{pth.dirname(pth.dirname(data_file))}/{sub_dir}'
                    if tmp_dir_key in out_key and out_dataname == dataset_name and out_ratio == ratio:
                        # # display(v4, out_file, key)
                        # res[tmp_dir]
                        img = tmp_dir + f'/case0-ratio_{ratio}.dat.pdf.png'
                        return img
    return img

def imgs2xlsx(in_file, data_path_mappings,  outfile='out/imgs.xlsx', root_dir='online'):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    with open(in_file, 'rb') as f:
        outs = pickle.load(f)

    res = []
    # find key = os.path.dirname(out_file)
    out_key = os.path.dirname(outfile)
    res = OrderedDict()
    for i, ratio in enumerate([0.5, 0.8, 0.9, 0.95, 1.0]):
        img_lst = []
        for data_name, value in data_path_mappings.items():
            # img = 'online/out/online/' + value + f'-case0-ratio_{ratio}.dat.pdf'
            # # img_lst.append(_pdf2img(img, img + '.png'))
            img_lst.append(find_img(data_name, ratio, out_key, root_dir, outs))
        res[ratio] = img_lst
        _imgs2xls(outfile, res)


def main2():
    n_init_train = 1000
    in_dir = f'data/feats/n_init_train_{n_init_train}'
    # in_dir = 'data/feats/'

    dataname_file_mappings = {
        # 'DEMO_IDS': 'DEMO_IDS/DS-srcIP_192.168.10.5',
        # 'mimic_GMM': f'{in_dir}/mimic_GMM_dataset/Xy-normal-abnormal.dat',
        #
        'UNB1_UNB2': f'{in_dir}/UNB1_UNB2/Xy-normal-abnormal.dat',
        'UNB1_UNB3': f'{in_dir}/UNB1_UNB3/Xy-normal-abnormal.dat',
        'UNB1_UNB4': f'{in_dir}/UNB1_UNB4/Xy-normal-abnormal.dat',
        'UNB1_UNB5': f'{in_dir}/UNB1_UNB5/Xy-normal-abnormal.dat',
        'UNB2_UNB3': f'{in_dir}/UNB2_UNB3/Xy-normal-abnormal.dat',
        'UNB1_CTU1': f'{in_dir}/UNB1_CTU1/Xy-normal-abnormal.dat',
        'UNB1_MAWI1': f'{in_dir}/UNB1_MAWI1/Xy-normal-abnormal.dat',
        'UNB2_CTU1': f'{in_dir}/UNB2_CTU1/Xy-normal-abnormal.dat',
        'UNB2_MAWI1': f'{in_dir}/UNB2_MAWI1/Xy-normal-abnormal.dat',
         'UNB2_FRIG1': f'{in_dir}/UNB2_FRIG1/Xy-normal-abnormal.dat', # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
         'UNB2_FRIG2': f'{in_dir}/UNB_FRIG2/Xy-normal-abnormal.dat',    # UNB1+Fridge: (normal: idle) abnormal: (browse)

         #
         # # 'UNB1_ISTS1': f'{in_dir}/UNB1_ISTS1/Xy-normal-abnormal.dat',
         #
         'CTU1_UNB1': f'{in_dir}/CTU1_UNB1/Xy-normal-abnormal.dat',
         'CTU1_MAWI1': f'{in_dir}/CTU1_MAWI1/Xy-normal-abnormal.dat',
         #
         'UChi_FRIG1': f'{in_dir}/UChi_FRI1/Xy-normal-abnormal.dat',  # Fridge: (normal: idle, and idle1) abnormal: (open_shut, browse)
         'UChi_FRIG2': f'{in_dir}/UChi_FRIG2/Xy-normal-abnormal.dat',

         'MAWI1_UNB1': f'{in_dir}/MAWI1_UNB1/Xy-normal-abnormal.dat',
         'MAWI1_CTU1': f'{in_dir}/MAWI1_CTU1/Xy-normal-abnormal.dat',# works
         'MAWI1_UNB2': f'{in_dir}/MAWI1_UNB2/Xy-normal-abnormal.dat',
         'CTU1_UNB2': f'{in_dir}/CTU1_UNB2/Xy-normal-abnormal.dat',
         #
         'UNB1_FRIG1': f'{in_dir}/UNB1_FRIG1/Xy-normal-abnormal.dat', # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
         'CTU1_FRIG1': f'{in_dir}/CTU1_FRIG1/Xy-normal-abnormal.dat', # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
         'MAWI1_FRIG1': f'{in_dir}/MAWI1_FRIG1/Xy-normal-abnormal.dat', # MAWI1+Fridge: (normal: idle) abnormal: (open_shut)

         'FRIG1_UNB1': f'{in_dir}/FRIG1_UNB1/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (open_shut)
         'FRIG1_CTU1': f'{in_dir}/FRIG1_CTU1/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (open_shut)
         'FRIG1_MAWI1': f'{in_dir}/FRIG1_MAWI1/Xy-normal-abnormal.dat',

         'UNB1_FRIG2': f'{in_dir}/UNB1_FRIG2/Xy-normal-abnormal.dat',    # UNB1+Fridge: (normal: idle) abnormal: (browse)
         'CTU1_FRIG2': f'{in_dir}/CTU1_FRIG2/Xy-normal-abnormal.dat', # CTU1+Fridge: (normal: idle) abnormal: (browse)
         'MAWI1_FRIG2': f'{in_dir}/MAWI1_FRIG2/Xy-normal-abnormal.dat', # MAWI1+Fridge: (normal: idle) abnormal: (browse)

         'FRIG2_UNB1': f'{in_dir}/FRIG2_UNB1/Xy-normal-abnormal.dat',  # UNB1+Fridge: (normal: idle) abnormal: (browse)
         'FRIG2_CTU1': f'{in_dir}/FRIG2_CTU1/Xy-normal-abnormal.dat',  # CTU1+Fridge: (normal: idle) abnormal: (browse)
         'FRIG2_MAWI1': f'{in_dir}/FRIG2_MAWI1/Xy-normal-abnormal.dat', # MAWI1+Fridge: (normal: idle) abnormal: (browse)
         # #
         'UNB1_SCAM1': f'{in_dir}/UNB1_SCAM1/Xy-normal-abnormal.dat', # UNB1+SCam
         'CTU1_SCAM1': f'{in_dir}/CTU1_SCAM1/Xy-normal-abnormal.dat', # UNB1+SCam
         'MAWI1_SCAM1': f'{in_dir}/MAWI1_SCAM1/Xy-normal-abnormal.dat', # UNB1+SCam
         'FRIG1_SCAM1': f'{in_dir}/FRIG1_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam
         'FRIG2_SCAM1': f'{in_dir}/FRIG2_SCAM1/Xy-normal-abnormal.dat',  # UNB1+SCam


        'MACCDC1_UNB1': f'{in_dir}/MACCDC1_UNB1/Xy-normal-abnormal.dat',
        'MACCDC1_CTU1': f'{in_dir}/MACCDC1_CTU1/Xy-normal-abnormal.dat',
        'MACCDC1_MAWI1': f'{in_dir}/MACCDC1_MAWI1/Xy-normal-abnormal.dat',

         # less flows of wshr1
         'UNB1_DRYER1': f'{in_dir}/UNB1_DRYER1/Xy-normal-abnormal.dat',
         'DRYER1_UNB1': f'{in_dir}/DRYER1_UNB1/Xy-normal-abnormal.dat',

         # it works
         'UNB1_DWSHR1': f'{in_dir}/UNB1_DWSHR1/Xy-normal-abnormal.dat',
         'DWSHR1_UNB1': f'{in_dir}/DWSHR1_UNB1/Xy-normal-abnormal.dat',


         'FRIG1_DWSHR1': f'{in_dir}/FRIG1_DWSHR1/Xy-normal-abnormal.dat',
         'FRIG2_DWSHR1': f'{in_dir}/FRIG2_DWSHR1/Xy-normal-abnormal.dat',
         'CTU1_DWSHR1': f'{in_dir}/CTU1_DWSHR1/Xy-normal-abnormal.dat',
         'MAWI1_DWSHR1': f'{in_dir}/MAWI1_DWSHR1/Xy-normal-abnormal.dat',
         'MACCDC1_DWSHR1': f'{in_dir}/MACCDC1_DWSHR1/Xy-normal-abnormal.dat',
         #
         # less flows of wshr1
         'UNB1_WSHR1': f'{in_dir}/UNB1_WSHR1/Xy-normal-abnormal.dat',
         'WSHR1_UNB1': f'{in_dir}/WSHR1_UNB1/Xy-normal-abnormal.dat',

    }

    # img_lst = [f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.5.dat.pdf',
    #            f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.8.dat.pdf',
    #            f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.9.dat.pdf',
    #            f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.95.dat.pdf',
    #            f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_1.0.dat.pdf']
    outfile = f'out/n_init_train_{n_init_train}_imgs.xlsx'
    imgs2xlsx(dataname_file_mappings, outfile)


def display(info, out_file, key=(), is_show=False):

    online_info = info['online']
    batch_info = info['batch']
    dataset_name, data_file, ratio, experiment_case = key

    def get_values_of_key(info, name='train_times'):
        vs = []
        for i_repeat, res in info.items():
            # train_times = res['train_times']
            # test_times = res['test_times']
            # acus = res['aucs']
            # n_components = res['n_components']
            if name == 'train_times':
                v = [_v['train_time'] for _v in res[name]]
            elif name == 'test_times':
                v = [_v['test_time'] for _v in res[name]]
            elif name == 'n_components':
                v = [_v[name] for _v in res['model_params']]
            else:
                v = res[name]
            vs.append(np.asarray(v))

        return np.asarray(vs)

    def _plot(ax, online_train_times, batch_train_times, xlabel, ylabel, title, out_file, ylim=[],
              legend_position='upper right'):

        y = batch_train_times
        x = range(y.shape[1])
        yerr = np.std(y, axis=0)
        y = np.mean(y, axis=0)
        ax.errorbar(x, y, yerr, ecolor='r', capsize=2, linestyle='-', marker='.', color='green',
                    markeredgecolor='green', markerfacecolor='green', label=f'Batch', alpha=0.9)  # marker='*',

        y = online_train_times
        yerr = np.std(y, axis=0)
        y = np.mean(y, axis=0)
        ax.errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                    markeredgecolor='blue', markerfacecolor='blue', label=f'Online', alpha=0.9)  # marker='*',

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            ax.set_ylim(ylim)  # [0.0, 1.05]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        ax.legend(loc=legend_position)
        # ax.set_title(title)
        out_file += '.png'
        if pth.exists(out_file): os.remove(out_file)
        if not pth.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
        print(out_file)
        ax.figure.savefig(out_file, format='png', dpi=300)

    params = online_info[0]['params']
    print(f'\n***{dataset_name}, {experiment_case}')
    if 'online:False' in experiment_case:
        online = False
        params.incorporated_points = 0
        params.fixed_U_size = False
    else:
        online = True

    n_point = params.incorporated_points
    n_point = f'{n_point}' if n_point > 1 else f'{n_point}'

    fixed_kjl = params.fixed_kjl
    fixed_U_size = params.fixed_U_size
    n_components_init = int(
        np.mean([online_info[i]['model_params'][0]['n_components'] for i in range(params.n_repeats)]))
    gs = params.gs
    covariance_type = params.covariance_type
    meanshift = params.meanshift
    q_kjl = params.q_kjl
    n_kjl = params.n_kjl
    d_kjl = params.d_kjl
    c_kjl = params.centering_kjl
    std = params.std
    random_state = params.random_state
    with_means_std = params.with_means_std
    kjl = params.kjl
    n_repeats = params.n_repeats
    # dataset_name, data_file = k_dataset
    dataset_name = f'{dataset_name} (init_set={int(params.percent_first_init * 100)}:{int(round((1 - params.percent_first_init) * 100))}-{params.X_init_train.shape})'
    if gs:
        init_val_set = f'{params.X_init_test.shape[0]}'
    else:
        init_val_set = f'=X_test'
    arrival_set = f'{params.X_arrival.shape[0]}'
    test_set = f'{params.X_test.shape[0]}'
    if kjl:
        # {covariance_type};
        title = f'n_cp={n_components_init}, {covariance_type}; gs={gs}; std={std}_c={with_means_std}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}, c={c_kjl}; ms={meanshift}'
        if fixed_kjl:
            title = f'Fixed KJL on {dataset_name};\ninit_val(test)={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n{title}'
        elif fixed_U_size:
            # (replace {n_point} cols and rows of U)
            title = f'Fixed U size on {dataset_name};\ninit_val(test)={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n{title}'
        else:  # increased_U
            title = f'Increased U size on {dataset_name};\ninit_val(test)={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n{title}'
    else:
        #  {covariance_type};
        title = f'n_cp={n_components_init}, {covariance_type}; gs={gs}; std={std}_ctr={with_means_std}; KJL={kjl},  c={c_kjl}; ms={meanshift}'
        title = f'{dataset_name};\ninit_val={init_val_set},arrival_set={arrival_set},test_set={test_set};n_rp={n_repeats};seed={random_state}\n({title})'

    title = title.replace('False', 'F')
    title = title.replace('True', 'T')
    batch_size = params.batch_size
    xlabel = f'The ith batch: i * batch_size({batch_size}) datapoints'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax = ax.reshape(1, 1)
    # train_times
    online_train_times = get_values_of_key(online_info, name='train_times')
    batch_train_times = get_values_of_key(batch_info, name='train_times')
    _plot(ax, online_train_times, batch_train_times, xlabel=xlabel, ylabel='Training time (s)', title=title,
          out_file= out_file +'/train_times.pdf')
    # test times
    online_test_times = get_values_of_key(online_info, name='test_times')
    batch_test_times = get_values_of_key(batch_info, name='test_times')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax = ax.reshape(1, 1)
    _plot(ax, online_test_times, batch_test_times, xlabel=xlabel, ylabel='Testing time (s)', title=title,
          out_file= out_file + '/test_times.pdf')

    # aucs
    online_aucs = get_values_of_key(online_info, name='aucs')
    batch_aucs = get_values_of_key(batch_info, name='aucs')
    print(f'online_aucs: {online_aucs}')
    print(f'batch_aucs: {batch_aucs}')
    print(f'online_aucs-batch_aucs: {online_aucs - batch_aucs}')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    _plot(ax, online_aucs, batch_aucs, xlabel=xlabel, ylabel='AUCs', title=title,
          legend_position='lower right', ylim=[0.0, 1.05],
          out_file= out_file + '/aucs.pdf')

    # n_components
    online_n_components = get_values_of_key(online_info, name='n_components')
    batch_n_components = get_values_of_key(batch_info, name='n_components')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    _plot(ax, online_n_components, batch_n_components, xlabel=xlabel, ylabel='n_components', title=title,
          legend_position='lower right',
          out_file= out_file + '/n_components.pdf')

    # fig.suptitle(title, fontsize=11)
    #
    # plt.tight_layout()  # rect=[0, 0, 1, 0.95]
    # try:
    #     plt.subplots_adjust(top=0.85, bottom=0.1, right=0.975, left=0.12)
    # except Warning as e:
    #     raise ValueError(e)
    #
    # # fig.text(.5, 15, "total label", ha='center')
    # plt.figtext(0.5, 0.01, f'X-axis:({xlabel})', fontsize=11, va="bottom", ha="center")
    # print(out_file)
    # if not pth.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
    # if pth.exists(out_file): os.remove(out_file)
    # fig.savefig(out_file, format='pdf', dpi=300)
    # out_file += '.png'
    # if pth.exists(out_file): os.remove(out_file)
    # fig.savefig(out_file, format='png', dpi=300)
    # if is_show: plt.show()
    # plt.close(fig)

    return out_file

def individual_img(in_file='out.dat',    root_dir = 'online'):

    with open(in_file, 'rb') as f:
        outs = pickle.load(f)

    for i, v1 in enumerate(outs):
        for (dataset_name, data_file), v2 in v1.items():
            for ratio, v3 in v2.items():
                for experiment_case, v4 in v3.items():
                    # online = v3['online']
                    # batch = v3['batch']
                    key = (dataset_name, data_file, ratio, experiment_case)
                    p = v4['online'][0]['params']
                    out_file = f'{root_dir}/out/{pth.dirname(data_file)}/gs={p.gs}-std={p.std}_center={p.with_means_std}-' \
                               f'kjl={p.kjl}-d_kjl={p.d_kjl}-n_kjl={p.n_kjl}-c_kjl={p.centering_kjl}-' \
                               f'ms={p.meanshift}-before_kjl={p.before_kjl_meanshift}-fixed_kjl={p.fixed_kjl}-seed={p.random_state}/case0-ratio_{ratio}.dat-single'
                    display(v4, out_file, key)



if __name__ == '__main__':
    # main(online=False)
    # main2()
    out_file = 'online/out/online/data/src_dst/iat_size/n_init_train_500/gs=True-std=False_center=False-kjl=True-d_kjl=5-n_kjl=100-c_kjl=False-ms=False-before_kjl=False-fixed_kjl=False-seed=10/res.dat'
    individual_img(out_file, root_dir = 'online')
