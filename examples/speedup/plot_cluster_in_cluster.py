import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm

from examples.speedup._speedup_kjl import BASE
#
# # sns.set_style("darkgrid")
# sns.set_style("whitegrid")
# # colorblind for diff
from kjl.utils.tool import check_path
#
# sns.set_palette("bright")  # for feature+header
# sns.palplot(sns.color_palette())


def make_circles(N=5000, r1=1, r2=5, w1=0.8, w2=1 / 3, arms=64):
    """ clusterincluster.m
    function data = clusterincluster(N, r1, r2, w1, w2, arms)
    %% Data is N x 3, where the last column is the label


        if nargin < 1
            N = 1000;
        end
        if nargin < 2
            r1 = 1;
        end
        if nargin < 3
            r2 = 5*r1;
        end
        if nargin < 4
            w1 = 0.8;
        end
        if nargin < 5
            w2 = 1/3;
        end
        if nargin < 6
            arms = 64;
        end

        data = [];

        N1 = floor(N/2);
        N2 = N-N1;

        phi1 = rand(N1,1) * 2 * pi;
        %dist1 = r1 + randint(N1,1,3)/3 * r1 * w1;
        dist1 = r1 + randi([0, 2], [N1 1])/3 * r1 * w1;

        d1 = [dist1 .* cos(phi1) dist1 .* sin(phi1) zeros(N1,1)];
        perarm = round(N2/arms);
        N2 = perarm * arms;
        radperarm = (2*pi)/arms;
        phi2 = ((1:N2) - mod(1:N2, perarm))/perarm * (radperarm);
        phi2 = phi2';
        dist2 = r2 * (1 - w2/2) + r2 * w2 * mod(1:N2, perarm)'/perarm;
        d2 = [dist2 .* cos(phi2) dist2 .* sin(phi2) ones(N2,1)];

        data = [d1;d2];
        %scatter(data(:,1), data(:,2), 20, data(:,3)); axis square;
    end

    """
    N1 = int(np.floor(N / 2))
    N2 = N - N1

    phi1 = np.random.rand(N1, 1) * 2 * np.pi  # return a matrix with shape N1x1, values in [0,1]
    # # % dist1 = r1 + np.random.randint(N1, 1, 3) / 3 * r1 * w1;
    dist1 = r1 + np.random.randint(0, high=2 + 1, size=[N1, 1]) / 3 * r1 * w1

    # d1 = [dist1. * np.cos(phi1) dist1. * np.sin(phi1) zeros(N1, 1)];
    # d1 = [col1, col2, label]
    d1 = np.concatenate([dist1 * np.cos(phi1), dist1 * np.sin(phi1), np.zeros((N1, 1))], axis=1)

    perarm = round(N2 / arms)
    N2 = perarm * arms
    radperarm = (2 * np.pi) / arms
    # phi2 = ((1:N2) - mod(1:N2, perarm)) / perarm * (radperarm)
    vs = np.reshape(range(1, N2 + 1), (N2, 1))
    phi12 = (vs - np.mod(vs, perarm)) / perarm * (radperarm)
    # phi2 = phi2';
    phil2 = phi12
    # dist2 = r2 * (1 - w2 / 2) + r2 * w2 * mod(vs, perarm)'/perarm;
    dist2 = r2 * (1 - w2 / 2) + r2 * w2 * np.mod(vs, perarm) / perarm
    # d2 = [dist2. * cos(phi2) dist2. * sin(phi2) ones(N2, 1)]
    d2 = np.concatenate([dist2 * np.cos(phil2), dist2 * np.sin(phil2), np.ones((N2, 1))], axis=1)
    data = np.concatenate([d1, d2], axis=0)
    # % scatter(data(:, 1), data(:, 2), 20, data(:, 3)); axis square;
    X, y = data[:, :2], data[:, -1]
    return X, y


def get_cluster_data(n_samples=5000, random_state=42):
    # X, y = datasets.make_circles(n_samples=n_samples, factor=.5, random_state=random_state,
    #                                       noise=.00)

    X, y = make_circles()
    return X, y


def plot_data(X, y, title='Data'):
    plt.figure(figsize=(10, 10))
    y_unique = np.unique(y)
    font_size = 20
    colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    colors = ['green', 'red', 'blue']
    for i, (this_y, color) in enumerate(zip(y_unique, colors)):
        this_X = X[y == this_y]
        plt.scatter(this_X[:, 0], this_X[:, 1], s=50,
                    # c=color[np.newaxis, :],
                    c=colors[i],
                    alpha=0.5, edgecolor='k',
                    label=f"{'Normal' if this_y < 1.0 else 'Novelty'}")
    plt.legend(loc="upper right", fontsize=font_size, frameon=False)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.title(title)
    plt.show()


def plot_compare_kjl_nystrom(raw_data, kjl_data, nystrom_data, out_file='.pdf'):
    check_path(out_file)
    print(out_file)

    # sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style('ticks')
    show_detectors = []
    # sns.set_style("darkgrid")
    # create plots
    num_figs = 3
    c = 3  # cols of subplots in each row
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
        # gridkw = dict(width_ratios = [1, 1, 1], height_ratios=[1])
        fig, axes = plt.subplots(r, num_figs, figsize=(14, 5))  # (width, height)
    print(f'subplots: ({r}, {c})')
    # fig, axes = plt.subplots(r, c, figsize=(18, 10))  # (width, height)
    if r == 1:
        axes = axes.reshape(1, -1)
    t = 0

    datasets = {'Data': raw_data, 'Projected data by KJL': kjl_data, 'Projected data by Nystrom': nystrom_data}
    for j, (data_name, (X, y)) in enumerate(datasets.items()):
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        new_colors = []
        if j % c == 0:
            if j == 0:
                t = 0
            else:
                t += 1

        print(f'{data_name}: {data}')
        df = pd.DataFrame(data, columns=['col1', 'col2', 'label'])
        colors = ['green', 'red',  'green', 'orange', 'c', 'm',  'b', 'r','tab:brown', 'tab:green'][:2]
        # g = sns.barplot(y="diff", x='dataset', hue='model_name', data=df, palette=colors, ci=None,
        #                 capsize=.2, ax=axes[t, j % c])
        g = sns.scatterplot(x='col1', y="col2", data=df, hue='label', ci=None, palette=colors, ax=axes[t, j % c])

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
        # g.set_ylabel(data_name, fontsize=font_size + 4)

        #
        # g.set_style("whitegrid", {'axes.grid': False})
        # sns.axes_style("whitegrid", {'axes.grid': False})
        # sns.axes_style('white')
        # g.grid(False)
        # g.set_xticklabels(g.get_xticklabels(), fontsize=font_size + 4, rotation=30, ha="center")
        # yticks(np.arange(0, 1, step=0.2))
        # g.set_yticklabels([f'{x:.2f}' for x in g.get_yticks() if x%0.5==0], fontsize=font_size + 4)

        # y_v = [float(f'{x:.1f}') for x in g.get_yticks() if x%0.5==0]
        # y_v = [float(f'{x:.2f}') for x in g.get_yticks()]
        # v_max = max([ int(np.ceil(v)) for v in g.get_yticks()])
        # step = v_max/5
        # step = int(step) if step > 1 else step
        # y_v = [v for v in range(0, v_max, step)]

        # g.set_yticklabels([v_tmp if v_tmp in [0, 0.5, -0.5, 1, -1] else '' for v_tmp in y_v],
        #                   fontsize=font_size + 6)  # set the number of each value in y axis
        # vs = max([abs(v) for v in g.get_yticks()])
        print(g.get_yticks(), g.get_xticks())
        if data_name == 'Data':
            y_v = [-8, -4, 0, 4, 8]
            x_v =  [-8, -4, 0, 4, 8]
        elif data_name == 'Projected data by KJL':
            y_v = [-100, -50, 0, 50, 100]
            x_v = [-150, -100, -50, 0, 50]
        elif data_name == 'Projected data by Nystrom':
            y_v = [-1, -0.5, 0, 0.5, 1]
            x_v = [-1, -0.75, -0.5, -0.25, 0]

        g.set_yticks(y_v)
        g.set_yticklabels(g.get_yticks(), fontsize=font_size)  # set the number of each value in y axis
        g.set_xticks(x_v)
        g.set_xticklabels(g.get_xticks(), fontsize=font_size)
        # print(g.get_yticks(), y_v)
        # g.set_yticklabels('')
        # if j % c != 0:
        #     # g.get_yaxis().set_visible(False)
        #     g.set_yticklabels(['' for v_tmp in y_v])
        #     g.set_ylabel('')

        g.set_title(data_name, fontsize=font_size , pad=10)
        # print(f'ind...{ind}')
        # if j == 1:
        #     # g.get_legend().set_visible()
        #     handles, labels = g.get_legend_handles_labels()
        #     axes[0, 1].legend(handles, labels, loc=8,fontsize=font_size-4) #bbox_to_anchor=(0.5, 0.5)
        # else:
        #     g.get_legend().set_visible()
        g.get_legend().set_visible(False)
    handles, labels = g.get_legend_handles_labels()
    # axes[t, j % c].legend(handles, labels, loc="upper right", fontsize=font_size - 4)  # bbox_to_anchor=(0.5, 0.5)
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
    # share one legend
    labels  = ['Normal', 'Novelty']
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

    # plt.gca().set_aspect('equal', adjustable='datalim')
    # plt.gca().set_aspect('equal')
    plt.tight_layout()
    #


    try:
        if r == 1:
            plt.subplots_adjust(bottom=0.19, top=0.90)
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


def main():
    random_state = 42
    X, y = get_cluster_data(n_samples=5000, random_state=random_state)
    plot_data(X, y)

    # KJL projection
    kjl_params = {'is_kjl': True, 'kjl_d': 2, 'kjl_n': 100, 'kjl_q': 0.3}
    # X_train, X_test = self.project_kjl(X_train, X_test, kjl_params=self.params)
    X_kjl, _ = BASE(random_state).project_kjl(X, X, kjl_params=kjl_params)
    plot_data(X_kjl, y)

    # Nystrom projection
    nystrom_params = {'is_nystrom': True, 'nystrom_d': 2, 'nystrom_n': 100, 'nystrom_q': 0.3}
    X_nystrom, _ = BASE(random_state).project_nystrom(X, X, nystrom_params=nystrom_params)
    plot_data(X_nystrom, y)

    out_file = 'speedup/out_cluster_in_cluster/raw_kjl_nystrom_comparison.pdf'
    plot_compare_kjl_nystrom((X, y), (X_kjl, y), (X_nystrom, y), out_file=out_file)


if __name__ == '__main__':
    main()
