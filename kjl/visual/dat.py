"""
    open .dat
"""
import matplotlib.pyplot as plt
import numpy as np

from kjl.utils.data import load_data


#
# def get_each(results, params):
#     kjl_n_aucs = {}
#     kjl_d_aucs = {}
#     kjl_q_aucs = {}
#     n_components_aucs= {}
#     for i, each_value in enumerate(results):
#         each_params = each_value['params']
#
#         if each_params['n_components'] == params['n_components'] and each_params['kjl_q' ]== 0.3 and each_params['kjl_d']  == 10:
#             kjl_n_aucs[each_params['kjl_n']] = each_value['aucs'][0]
#
#         if each_params['n_components'] == params['n_components']  and each_params['kjl_q'] == 0.3 and each_params['kjl_n'] == 100:
#             kjl_d_aucs[each_params['kjl_d']] = each_value['aucs'][0]
#
#         if each_params['n_components'] == params['n_components']  and each_params['kjl_d' ]== 10 and each_params['kjl_n']  == 100:
#             kjl_q_aucs[each_params['kjl_q']] = each_value['aucs'][0]
#
#         if each_params['kjl_q'] == 0.3 and each_params['kjl_d'] == 10 and each_params['kjl_n'] == 100:
#             n_components_aucs[each_params['n_components']] = each_value['aucs'][0]
#
#     return kjl_n_aucs, kjl_d_aucs, kjl_q_aucs, n_components_aucs


def plot_err_bar(x, y, y_err, xlabel='range', ylabel='auc', title=''):
    # with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    # ax.plot(x, y, '*-', alpha=0.9)
    # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)
    plt.errorbar(x, y, yerr=y_err, ecolor='r', capsize=2, linestyle='-', marker='*', markeredgecolor='m',
                 label=r'AUC($\mu \pm \sigma$)')

    # plt.xlim([0.0, 1.0])
    plt.ylim([0., 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xticks(x)
    # plt.yticks(y)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def get_each(results, params):
    kjl_n_aucs = {}
    kjl_d_aucs = {}
    kjl_q_aucs = {}
    n_components_aucs = {}
    for i, each_value in enumerate(results):
        each_params = each_value['params']

        if each_params['n_components'] == params['n_components'] and each_params['kjl_q'] == params['q'] and \
                each_params['kjl_d'] == params['d']:
            kjl_n_aucs[each_params['kjl_n']] = each_value['aucs']

        if each_params['n_components'] == params['n_components'] and each_params['kjl_q'] == params['q'] and \
                each_params['kjl_n'] == params['n']:
            kjl_d_aucs[each_params['kjl_d']] = each_value['aucs']

        if each_params['n_components'] == params['n_components'] and each_params['kjl_d'] == params['d'] and \
                each_params['kjl_n'] == params['n']:
            kjl_q_aucs[each_params['kjl_q']] = np.mean(each_value['aucs'])

        if each_params['kjl_q'] == params['q'] and each_params['kjl_d'] == params['d'] and each_params['kjl_n'] == \
                params['n']:
            n_components_aucs[each_params['n_components']] = np.mean(each_value['aucs'])

    return kjl_n_aucs, kjl_d_aucs, kjl_q_aucs, n_components_aucs


def main():
    data_mapping = {
        # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5':'UNB_PC1',
        'DS10_UNB_IDS/DS12-srcIP_192.168.10.8': 'UNB_PC2',
        'DS10_UNB_IDS/DS13-srcIP_192.168.10.9': 'UNB_PC3',
        'DS10_UNB_IDS/DS14-srcIP_192.168.10.14': 'UNB_PC4',
        'DS10_UNB_IDS/DS15-srcIP_192.168.10.15': 'UNB_PC5',

        # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1':'SMTV',
        # # #
        'DS40_CTU_IoT/DS41-srcIP_10.0.2.15': 'CTU',
        'DS40_CTU_IoT/DS42-srcIP_192.168.1.196': 'CTU',
        #
        # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50':'MAWI',
        # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165': 'MAWI_PC2',
        # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32':'MAWI_PC3',
        # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171':'MAWI_PC4',
        # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204':'MAWI_PC5',
        # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109':'MAWI_PC6',

        'ISTS/2015': 'ISTS',
        'MACCDC/2012': 'MACCDC',

        # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20':'GHom',
        'DS60_UChi_IoT/DS62-srcIP_192.168.143.42': 'SCam',
        # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43':'SFrig',
        # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48':'BSTCH',

    }

    datasets = [
        #     # # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',
        'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
        #     # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
        #     # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
        #     # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
        #     # # # # #
        #     # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
        #     # # # # # #
        #     'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
        'DS40_CTU_IoT/DS42-srcIP_192.168.1.196',
        #     # # #
        #     # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        #     # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
        'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
        #     # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
        #     # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
        #     # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
        #     # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
        #     # #
        #     # # #
        #     # 'WRCCDC/2020-03-20',
        #     # 'DEFCON/ctf26',
        'ISTS/2015',
        'MACCDC/2012',

        #     # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
        'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
        #     # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
        #     # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'
        #

    ]
    # dataset_name = datasets[0]

    for dataset_name in datasets:
        print(f'\n\n***dataset_name: {dataset_name}')
        # data_type = 'all_results'
        data_type = 'middle_results'
        if data_type == 'all_results':
            in_file = 'out/data_kjl/all_results.dat'
            results = load_data(in_file)
            print(results)
        elif data_type == 'middle_results':
            if 'DS10_UNB_IDS/DS12-srcIP_192.168.10.8' in dataset_name:
                in_file = f'out/data_kjl/{dataset_name}/iat_size/header:False/' \
                          'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False.csv-middle_results.dat'
                params = {'file_name': dataset_name,
                          'detector': 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False',
                          'n_components': 5,
                          'q': 0.4,
                          'd': 10,
                          'n': 100
                          }

            elif 'DS40_CTU_IoT/DS42-srcIP_192.168.1.196' in dataset_name:
                in_file = f'out/data_kjl/{dataset_name}/iat_size/header:False/' \
                          'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False.csv-middle_results.dat'
                params = {'file_name': dataset_name,
                          'detector': 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False',
                          'n_components': 5,
                          'q': 0.2,
                          'd': 10,
                          'n': 100
                          }
            elif 'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165' in dataset_name:
                in_file = f'out/data_kjl/{dataset_name}/iat_size/header:False/' \
                          'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False.csv-middle_results.dat'
                params = {'file_name': dataset_name,
                          'detector': 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False',
                          'n_components': 45,
                          'q': 0.9,
                          'd': 10,
                          'n': 100
                          }


            elif 'ISTS/2015' == dataset_name:
                in_file = f'out/data_kjl/{dataset_name}/iat_size/header:False/' \
                          'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False.csv-middle_results.dat'
                params = {'file_name': dataset_name,
                          'detector': 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False',
                          'n_components': 45,
                          'q': 0.2,
                          'd': 10,
                          'n': 100
                          }

            elif 'MACCDC/2012' == dataset_name:
                in_file = f'out/data_kjl/{dataset_name}/iat_size/header:False/' \
                          'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False.csv-middle_results.dat'
                params = {'file_name': dataset_name,
                          'detector': 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False',
                          'n_components': 35,
                          'q': 0.4,
                          'd': 10,
                          'n': 100
                          }

            elif 'DS60_UChi_IoT/DS62-srcIP_192.168.143.42' == dataset_name:
                in_file = f'out/data_kjl/{dataset_name}/iat_size/header:False/' \
                          'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False.csv-middle_results.dat'
                params = {'file_name': dataset_name,
                          'detector': 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False',
                          'n_components': 40,
                          'q': 0.7,
                          'd': 10,
                          'n': 100
                          }

            in_file = 'out/data_kjl/all_results-kjl.dat'
            results = load_data(in_file)
            # get middle results
            results = results[(f'data/data_kjl/{dataset_name}/iat_size/header:False',
                               'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False')][
                1]
            # pprint(results)
            kjl_n_aucs, kjl_d_aucs, kjl_q_aucs, n_components_aucs = get_each(results, params)
            title = data_mapping[params['file_name']]
            y = [np.mean(v) for v in list(kjl_n_aucs.values())]
            y_err = [np.std(v) for v in list(kjl_n_aucs.values())]
            print(f'{dataset_name}, n (varies): {y_err}')
            plot_err_bar(x=list(kjl_n_aucs.keys()), y=y, y_err=y_err, xlabel='n',
                         ylabel='auc',
                         title=title + '\n(d={}, q={} (of kjl) and n_components={} (of GMM))'.format(params['d'],
                                                                                                     params['q'],
                                                                                                     params[
                                                                                                         'n_components']))

            y = [np.mean(v) for v in list(kjl_d_aucs.values())]
            y_err = [np.std(v) for v in list(kjl_d_aucs.values())]
            print(f'{dataset_name}, d (varies): {y_err}')
            plot_err_bar(x=list(kjl_d_aucs.keys()), y=y, y_err=y_err, xlabel='d',
                         ylabel='auc',
                         title=title + '\n(n={}, q={} (of kjl) and n_components={} (of GMM))'.format(params['n'],
                                                                                                     params['q'],
                                                                                                     params[
                                                                                                         'n_components']))

            # plot_data(x=list(kjl_q_aucs.keys()), y=list(kjl_q_aucs.values()), xlabel='q',
            #           ylabel='auc',
            #           title=title + '\n(d={}, n={} (of kjl) and n_components={} (of GMM))'.format(params['d'], params['n'],
            #                                                                                       params['n_components']))
            #
            # plot_data(x=list(n_components_aucs.keys()), y=list(n_components_aucs.values()), xlabel='n_components',
            #           ylabel='auc',
            #           title=title + '\n(d={}, n={}, and q={} (of kjl) )'.format(params['d'], params['n'], params['q']))


if __name__ == '__main__':
    main()
