import matplotlib.pyplot as plt
import numpy as np


def plot_data(x, y, xlabel='range', ylabel='auc', title=''):
    # with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.plot(x, y, '*-', alpha=0.9)
    # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

    # plt.xlim([0.0, 1.0])
    plt.ylim([0., 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xticks(x)
    # plt.yticks(y)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def plot_err_bar(x, y, xlabel='range', ylabel='auc', title=''):
    """
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/errorbar_limits_simple.html#sphx-glr-gallery-lines-bars-and-markers-errorbar-limits-simple-py

    Parameters
    ----------
    x
    y
    xlabel
    ylabel
    title

    Returns
    -------

    """
    fig = plt.figure()
    x = np.arange(10)
    y = 2.5 * np.sin(x / 20 * np.pi)
    yerr = np.linspace(0.05, 0.2, 10)

    plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')

    plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')

    plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
                 label='uplims=True, lolims=True')

    upperlimits = [True, False] * 5
    lowerlimits = [False, True] * 5
    plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
                 label='subsets of uplims and lolims')

    plt.legend(loc='lower right')
    plt.show()
