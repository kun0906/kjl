import os

from pdf2image import convert_from_bytes

from kjl.utils.tool import load_data
import numpy as np
import matplotlib.pyplot as plt
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
        ax.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-',marker='*', markeredgecolor='b',
                     label=label)   #  marker='*',

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

    def plot_times(ax,y_dict, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):


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
        best_results_arr = [best_res for (best_res, mid_res) in v ]
        middle_results_arr =[mid_res for (best_res, mid_res) in v ]

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

        fig, ax = plt.subplots(nrows=2,ncols=2)

        y = train_times_dict
        plot_times(ax[0,0], y, xlabel=xlabel, ylabel='Training time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-train_times.pdf'))

        # ys = best_results['test_times']
        y= test_times_dict
        plot_times(ax[0,1], y, xlabel=xlabel, ylabel='Testing time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-test_times.pdf'))

        # y = best_results['aucs']
        y= aucs
        y_err = aucs_std
        plot_data(ax[1,0], range(len(y)), y, y_err,  xlabel=xlabel, ylabel='AUC', ylim=[0.0, 1.05], title=title,
                  out_file=out_file.replace('.pdf', '-aucs.pdf'))

        # y = [v['n_components'] for v in best_results['model_params']]
        y = n_components
        y_err = n_components_std
        plot_data(ax[1,1], range(len(y)), y, y_err, xlabel=xlabel, ylabel='n_components', title=title,
                  out_file=out_file.replace('.pdf', '-n_components.pdf'))

        # y = best_results['novelty_threses']
        # plot_data(range(len(y)), y, xlabel=xlabel, ylabel='novelty threshold', title=title,
        #           out_file=out_file.replace('.pdf', '-novelty.pdf'))

        # y = best_results['abnormal_threses']
        y= abnormal_threses
        y_err = abnormal_threses_std
        q_thres = params.q_abnormal_thres
        plot_data(plt, range(len(y)), y, y_err, xlabel=xlabel, ylabel='abnormal threshold', title=title,
                  label = f'{q_thres} quantile of normal scores',
                  out_file=out_file.replace('.pdf', '-abnormal.pdf'))

        # y1 = best_results['novelty_threses']
        # y2 = best_results['abnormal_threses']
        # xs = [range(len(y1)), range(len(y2))]
        # ys = [y1, y2]
        # plot_data2(xs, ys, xlabel=xlabel, ylabel='Threshold', title=title,
        #            out_file=out_file.replace('.pdf', '-threshold.pdf'))

        plt.show()


def _plot_each_result(results, k_dataset=(), experiment_case = '', out_file=''):

    def plot_data(ax, x, y, y_err, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):
        ax.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-',marker='*', markeredgecolor='b',
                     label=label)   #  marker='*',

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            ax.set_ylim(ylim)  # [0.0, 1.05]
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        ax.legend(loc='lower right')
        # ax.set_title(title)


    def plot_times(ax,y_dict, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):
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
    for (k, v) in best_results['train_times'][0].items():   # {}
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

def plot_data(ax, x, y_batch, y_err_batch, y_online, y_err_online, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label='', legend_position = 'upper right'):
        # x = range(y_batch)
        ax.errorbar(x, y_batch, yerr=y_err_batch, ecolor='r', capsize=2, linestyle='-',marker='.', markeredgecolor='g',
                     label=f'Batch', alpha=0.9)   #  marker='*',
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
            params = _result[0][0]['params']    # _result[n_repeats] = {best_results, middle_result}
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
    dataset_name = f'{dataset_name} (init_ratio={int(params.percent_first_init*100)}:{int(round((1-params.percent_first_init)*100))})'
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
        else: # increased_U
            title = f'Batch GMM vs. Online GMM with unfixed KJL on {dataset_name}\n{title}'
    else:
        title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}; n_repeats={n_repeats}'
        title = f'Batch GMM vs. Online GMM on {dataset_name}\n({title})'


    batch_size = params.batch_size
    xlabel = f'The ith batch: i * batch_size({batch_size}) datapoints'

    fig, ax = plt.subplots(nrows=2,ncols=2)
    x = range(len(result_online_GMM['aucs'][0]))
    y_batch, y_err_batch = result_batch_GMM['train_times']
    y_online, y_err_online = result_online_GMM['train_times']
    # plot_times(ax[0,0],x, y_batch_dict, y_online_dict, xlabel=xlabel, ylabel='Training time (s)', title=title,
    #            out_file=out_file.replace('.pdf', '-train_times.pdf'))
    plot_data(ax[0, 0], x, y_batch, y_err_batch, y_online, y_err_online, xlabel=xlabel, ylabel='Training time (s)', title=title,
               label='Train_time (s)',legend_position = 'upper right',
              out_file=out_file.replace('.pdf', '-train_times.pdf'))

    y_batch, y_err_batch = result_batch_GMM['test_times']
    y_online, y_err_online = result_online_GMM['test_times']
    # plot_times(ax[0,1],x, y_batch_dict, y_online_dict, xlabel=xlabel, ylabel='Testing time (s)', title=title,
    #            out_file=out_file.replace('.pdf', '-test_times.pdf'))
    plot_data(ax[0, 1], x, y_batch, y_err_batch, y_online, y_err_online, xlabel=xlabel,
              ylabel='Testing time (s)', title=title, label='Test_time (s)', legend_position = 'upper right',
              out_file=out_file.replace('.pdf', '-test_times.pdf'))

    y_batch, y_err_batch= result_batch_GMM['aucs']
    y_online, y_err_online = result_online_GMM['aucs']
    plot_data(ax[1,0], x, y_batch, y_err_batch, y_online, y_err_online,  xlabel=xlabel, ylabel='AUC', ylim=[0.0, 1.05], title=title,
              label='AUC',legend_position = 'lower right',
              out_file=out_file.replace('.pdf', '-aucs.pdf'))

    y_batch, y_err_batch = result_batch_GMM['n_components']
    y_online, y_err_online = result_online_GMM['n_components']
    plot_data(ax[1,1],x, y_batch, y_err_batch, y_online, y_err_online, xlabel=xlabel, ylabel='n_components', title=title,
              label='n_components',legend_position = 'lower right',
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
        ax.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-',marker='*', markeredgecolor='b',
                     label=label)   #  marker='*',

        # plt.xlim([0.0, 1.0])
        if len(ylim) == 2:
            ax.set_ylim(ylim)  # [0.0, 1.05]
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # plt.xticks(x)
        # plt.yticks(y)
        ax.legend(loc='lower right')
        # ax.set_title(title)


    def plot_times(ax,y_dict, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):
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
        best_results_arr = [best_res for (best_res, mid_res) in v ]
        middle_results_arr =[mid_res for (best_res, mid_res) in v ]

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
        dataset_name = f'{dataset_name} (init_ratio={int(params.percent_first_init*100)}:{int(round((1-params.percent_first_init)*100))})'
        if kjl:
            title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}, n={n_kjl}, d={d_kjl}, q={q_kjl}; n_repeats={n_repeats}'
            if fixed_kjl:
                title = f'online GMM with fixed KJL on {dataset_name}\n{title}' if online else f'Batch GMM with fixed KJL on {dataset_name}\n({title})'
            elif fixed_U_size:
                # (replace {n_point} cols and rows of U)
                title = f'online GMM with fixed U size on {dataset_name}\n{title}' if online else f'Batch GMM on {dataset_name}\n({title})'
            else: # increased_U
                title = f'online GMM with unfixed KJL (incorporated {n_point}) on {dataset_name}\n{title}' if online else f'Batch GMM on {dataset_name}\n({title})'
        else:
            title = f'n_components={n_components_init}, {covariance_type}; KJL={kjl}; n_repeats={n_repeats}'
            title = f'online GMM on {dataset_name}\n({title})' if online else f'Batch GMM on {dataset_name}\n({title})'


        batch_size = params.batch_size
        xlabel = f'The ith batch: i * batch_size({batch_size}) datapoints'

        fig, ax = plt.subplots(nrows=2,ncols=2)

        y = train_times_dict
        plot_times(ax[0,0], y, xlabel=xlabel, ylabel='Training time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-train_times.pdf'))

        # ys = best_results['test_times']
        y= test_times_dict
        plot_times(ax[0,1], y, xlabel=xlabel, ylabel='Testing time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-test_times.pdf'))

        # y = best_results['aucs']
        y= aucs
        y_err = aucs_std
        plot_data(ax[1,0], range(len(y)), y, y_err,  xlabel=xlabel, ylabel='AUC', ylim=[0.0, 1.05], title=title,
                  label='AUC', out_file=out_file.replace('.pdf', '-aucs.pdf'))

        # y = [v['n_components'] for v in best_results['model_params']]
        y = n_components
        y_err = n_components_std
        plot_data(ax[1,1], range(len(y)), y, y_err, xlabel=xlabel, ylabel='n_components', title=title,
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


def imgs2xls(xls_pth, img_lst = []):
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
    df = pd.DataFrame({'Data': [10, 20, 30, 20, 15, 30, 45]})

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(xls_pth, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']



    # Insert an image.
    for i, img in enumerate(img_lst):
        # worksheet.insert_image('D3', 'logo.png')
        print(i, img)

        from PIL import Image

        im = Image.open(img)
        image_width, image_height = im.size

        # image_width = 140.0
        # image_height = 182.0

        cell_width = 200.0
        cell_height = 200.0

        x_scale = cell_width / image_width
        y_scale = cell_height / image_height

        worksheet.insert_image(i+2, 3, img, {'x_scale': x_scale, 'y_scale': y_scale})
        worksheet.set_column(1, 10, width=cell_width/5)
        worksheet.set_row(i+2, height=cell_height)

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
    print(pdf_pth, os.path.exists(pdf_pth))
    """
    install pdf2image, poppler
    https://github.com/Belval/pdf2image
    https://stackoverflow.com/questions/54990306/how-to-write-pandas-dataframe-and-insert-image-or-chart-into-the-same-excel-wor
    """
    # from pdf2image import convert_from_path
    # pages = convert_from_path(pdf_pth, 500)
    pages = convert_from_bytes(open(pdf_pth, 'rb').read())
    for page in pages:
        page.save(img_pth, 'JPEG')
        break
    return img_pth

def main2():
    in_dir = 'out/data/feats/'
    img_lst = [f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.5.dat.pdf',
               f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.8.dat.pdf',
               f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.9.dat.pdf',
               f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_0.95.dat.pdf',
               f'{in_dir}CTU1_FRIG1/Xy-normal-abnormal.dat-case0-ratio_1.0.dat.pdf']


    img_lst = [_pdf2img(v, v+'.jpg') for v in img_lst]
    imgs2xls('out/imgs.xlsx', img_lst)

if __name__ == '__main__':

    # main(online=False)
    main2()