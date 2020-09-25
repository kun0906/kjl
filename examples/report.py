from kjl.utils.tool import load_data
import numpy as np

def plot_result(result, out_file, fixed_U=None, n_point=None):
    import matplotlib.pyplot as plt

    def plot_data(x, y, y_err, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):

        # with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        # ax.plot(x, y, '*-', alpha=0.9, label=label)
        # # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
        plt.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-',marker='*', markeredgecolor='b',
                     label=label)   #  marker='*',

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

    def plot_times(y_dict, xlabel='range', ylabel='auc', ylim=[], title='', out_file='', label=''):

        fig, ax = plt.subplots()
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
            plt.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='m',
                         label=name)

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

    for (in_dir, case_str), (best_results_arr, middle_results_arr) in result.items():

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

        print(f'\n***{in_dir}, {case_str}')
        if 'online_False' in case_str:
            online = False
            best_results['params']['incorporated_points'] = 0
            best_results['params']['fixed_U'] = False
        else:
            online = True

        if n_point is None:
            n_point = best_results['params']['incorporated_points']
            n_point = f'{n_point} datapoints' if n_point > 1 else f'{n_point} datapoint'

        fixed_U = best_results['params']['fixed_U']
        if fixed_U:
            title = f'online GMM with fixed KJL (incorporated {n_point})' if online else f'Batch GMM'
        else:
            title = f'online GMM with unfixed KJL (incorporated {n_point})' if online else f'Batch GMM'

        batch_size = best_results['params']['batch_size']
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
        y = train_times_dict
        plot_times(y, xlabel=xlabel, ylabel='Training time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-train_times.pdf'))

        # ys = best_results['test_times']
        y= test_times_dict
        plot_times(y, xlabel=xlabel, ylabel='Testing time (s)', title=title,
                   out_file=out_file.replace('.pdf', '-test_times.pdf'))

        # y = [v['n_components'] for v in best_results['model_params']]
        y = n_components
        y_err = n_components_std
        plot_data(range(len(y)), y, y_err, xlabel=xlabel, ylabel='n_components', title=title,
                  out_file=out_file.replace('.pdf', '-n_components.pdf'))

        # y = best_results['aucs']
        y= aucs
        y_err = aucs_std
        plot_data(range(len(y)), y, y_err,  xlabel=xlabel, ylabel='AUC', ylim=[0.0, 1.05], title=title,
                  out_file=out_file.replace('.pdf', '-aucs.pdf'))

        # y = best_results['novelty_threses']
        # plot_data(range(len(y)), y, xlabel=xlabel, ylabel='novelty threshold', title=title,
        #           out_file=out_file.replace('.pdf', '-novelty.pdf'))

        # y = best_results['abnormal_threses']
        y= abnormal_threses
        y_err = abnormal_threses_std
        q_thres = best_results['params']['q_abnormal_thres']
        plot_data(range(len(y)), y, y_err, xlabel=xlabel, ylabel='abnormal threshold', title=title,
                  label = f'{q_thres} quantile of normal scores',
                  out_file=out_file.replace('.pdf', '-abnormal.pdf'))

        # y1 = best_results['novelty_threses']
        # y2 = best_results['abnormal_threses']
        # xs = [range(len(y1)), range(len(y2))]
        # ys = [y1, y2]
        # plot_data2(xs, ys, xlabel=xlabel, ylabel='Threshold', title=title,
        #            out_file=out_file.replace('.pdf', '-threshold.pdf'))



def main(online=True):
    out_dir = 'out/data/data_reprst/pcaps/mimic_GMM_dataset'
    results = load_data( f'{out_dir}/all-online_{online}-results.dat')
    ##########################################################################################
    out_file = f'{out_dir}/online_{online}.pdf'
    # # Todo: format the results
    plot_result(results, out_file, fixed_U=None, n_point=None)
    print("\n\n---finish succeeded!")

if __name__ == '__main__':

    main(online=False)
