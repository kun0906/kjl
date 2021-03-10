""" improvement: average auc (with KJL) - average auc (without KJL)

"""
import os
import traceback

import numpy as np
import pandas as pd
from pandas import ExcelWriter

from kjl.utils.data import load_data


#
# def _get_all_diffs(idx_full, idx_w_kjl, df, values):
#     # get each row
#     # df.iloc[1].values
#     for i in range(5):  # 5 datasets
#         # GMM: diag
#         _line = []
#         print(i, df.iloc[idx_w_kjl + i].values.tolist())
#         for j in range(len(df.columns)):
#             if j == df.columns.get_loc('feat_set/auc') and j == 7:
#                 wo_kjl = df.iloc[idx_full + i]['feat_set/auc']
#                 w_kjl = df.iloc[idx_w_kjl + i]['feat_set/auc']
#                 diff = get_diff(wo_kjl, w_kjl)
#                 _line.append(diff)
#
#                 wo_kjl = df.iloc[idx_full + i]['train_time']
#                 w_kjl = df.iloc[idx_w_kjl + i]['train_time']
#                 diff = get_diff(wo_kjl, w_kjl)
#                 _line.append(diff)
#
#                 wo_kjl = df.iloc[idx_full + i]['test_time']
#                 w_kjl = df.iloc[idx_w_kjl + i]['test_time']
#                 diff = get_diff(wo_kjl, w_kjl)
#                 _line.append(diff)
#             elif j == 8 or j == 9:
#                 continue
#             else:
#                 _line.append(df.iloc[idx_full + i].values.tolist()[j])
#         values.append(_line)
#
#     return values
#
#
# def improvement_back(resulst_xlsx, out_file='-improvement.xlsx'):
#     xls = pd.ExcelFile(resulst_xlsx)
#     # Now you can list all sheets in the file
#     print(f'xls.sheet_names:', xls.sheet_names)
#     with ExcelWriter(out_file) as writer:
#         for _, sheet_name in enumerate(xls.sheet_names):
#             print(sheet_name)
#             if sheet_name not in ['KJL-GMM']:
#                 continue
#             df = pd.read_excel(resulst_xlsx, sheet_name=sheet_name, header=0,
#                                index_col=None)  # index_col=False: not use the first columns as index
#
#             # get each column
#             # feat_auc=df['feat_set/auc'].values.tolist()
#             # train_time = df['train_time'].values.tolist()
#             # test_time= df['test_time'].values.tolist()
#
#             values=[[str(v).replace('nan', '') for v in df.columns.values.tolist()]]
#             values.append([str(v).replace('diag', 'full') for v in df.iloc[0].values.tolist()])
#             idx_diag = 1
#             idx_w_kjl = 31 - 2  # -2: header not included, python index starts from 0, so -2
#             _get_all_diffs(idx_diag, idx_w_kjl, df, values)
#
#             # for GMM: full
#             values.append(['']* len(df.columns))
#             values.append([str(v).replace('nan', '') for v in df.columns.values.tolist()])
#             values.append([str(v).replace('diag', 'full') for v in df.iloc[0].values.tolist()])
#             idx_full = 10 - 2
#             idx_w_kjl = 38 - 2  # -2: header not included, python index starts from 0, so -2
#             _get_all_diffs(idx_full, idx_w_kjl, df, values)
#
#             # Generate dataframe from list and write to xlsx.
#             pd.DataFrame(values).to_excel(writer, sheet_name=sheet_name, index=False, header=False)
#
#         writer.save()
#
#     return out_file
#


def _get_diff(cell_1, cell_2, same=False):
    """ cell_2 / cell_1

    # combine two random variables=
    # https://stats.stackexchange.com/questions/117741/adding-two-or-more-means-and-calculating-the-new-standard-deviation
    # E(X+Y) = E(X) + E(Y)
    # Var(X+Y) = var(X) + var(Y) + 2cov(X, Y)
    # If X and Y are independent, Var(X+Y) = var(X) + var(Y)

    # divide two independent random normal variables
    # https://en.wikipedia.org/wiki/Ratio_distribution
    #  E(X/Y) = E(X)E(1/Y)
    #  Var(X/Y) =  E([X/Y]**2)-[E([X/Y])]**2 = [E(X**2)] [E(1/Y**2)] - {E(X)]**2 [E(1/Y)]**2
    # {\displaystyle \sigma _{z}^{2}={\frac {\mu _{x}^{2}}{\mu _{y}^{2}}}\left({\frac {\sigma _{x}^{2}}{\mu _{x}^{2}}}+{\frac {\sigma _{y}^{2}}{\mu _{y}^{2}}}\right)}

    Parameters
    ----------
    cell_1: "train_time: mean+/std)"
    cell_2

    Returns
    -------

    """
    try:
        _arr_1 = cell_1.split(':')[1]
        _arr_1 = [float(_v.split(')')[0]) for _v in _arr_1.split('+/-')]

        if same:
            mu = _arr_1[0]
            sigma = _arr_1[1]
        else:
            _arr_2 = cell_2.split(':')[1]
            _arr_2 = [float(_v.split(')')[0]) for _v in _arr_2.split('+/-')]

            # cell2 - cell1
            # mu = _arr_2[0] - _arr_1[0]      #
            # sigma = np.sqrt(_arr_2[1] ** 2 + _arr_1[1] ** 2)

            # X~ N(u_x, s_x**2), Y~ N(u_y, s_y**2), Z = X/Y
            # X and Y are independent,
            # E(X/Y) = E(X)E(1/Y) = E(X)/E(Y)
            # cell2 / cell1
            u_x = _arr_1[0]  # OCSVM
            u_y = _arr_2[0]  # other method
            mu = u_x / u_y
            s_x = _arr_1[1]
            s_y = _arr_2[1]
            # one approximation for Var(X/Y)
            sigma = np.sqrt(((u_x / u_y) ** 2) * ((s_x / u_x) ** 2 + (s_y / u_y) ** 2))

        diff = f"{mu:.3f} +/- {sigma:.3f}"

    except Exception as e:
        traceback.print_exc()
        diff = f"{-1:.3f} +/- {-1:.3f}"
    return diff


def get_all_aucs(raw_values):
    values = raw_values.split(':')[1].split('-')
    values = [float(v) for v in values]
    # for i, v in enumerate(raw_values):
    #     if i == 0:
    #         values.append(float(v.split(':')[1]))
    #     elif i == len(values)-1:
    #         values.append(float(v.split(']')[0]))
    #     else:
    #         values.append(float(v))

    return values


def get_auc_ratio(ocsvm_aucs, gmm_aucs, same=True):
    if same:
        mean_ratio = np.mean(ocsvm_aucs)
        std_ratio = np.std(ocsvm_aucs)
    else:
        mean_ocsvm = np.mean(ocsvm_aucs)
        mean_gmm = np.mean(gmm_aucs)

        mean_ratio = mean_gmm / mean_ocsvm

        ratios = [v / mean_ocsvm for v in gmm_aucs]
        std_ratio = np.std(ratios)

    diff = f"{mean_ratio:.2f} +/- {std_ratio:.2f}"

    return diff


def get_diff(df1, df2, feat_set='iat', same=False):
    """ d2 - df1

    Parameters
    ----------
    df2
    df1

    Returns
    -------

    """
    values = []
    # get each row
    # df.iloc[1].values
    for i in range(df1.shape[0]):  # 5 datasets
        # GMM: diag
        _line = []
        _line2 = []
        _line3 = []
        print(i, df2.iloc[i].values.tolist())
        for j in range(len(df2.columns)):
            # if j == df2.columns.get_loc('feat_set'):
            start_col = 7
            if j == start_col:
                _line_str = ''
                ocsvm_aucs = get_all_aucs(df1.iloc[i][j])
                gmm_aucs = get_all_aucs(df2.iloc[i][j])
                diff = get_auc_ratio(ocsvm_aucs, gmm_aucs, same)  # auc ratio should be GMM/OCSVM
                _line.append(diff)
                # _line_str +=diff

                ocsvm_train_times = get_all_aucs(df1.iloc[i][j + 1])
                gmm_train_times = get_all_aucs(df2.iloc[i][j + 1])
                # get 'number of points of the train set': df1.iloc[i][2] = ' X_train_shape: (8000-33)'
                n_train = int(df1.iloc[i][2].split('(')[1].split('-')[0])
                # get the test time for each point
                scale = 300
                ocsvm_train_times = [v / n_train * scale for v in ocsvm_train_times]
                gmm_train_times = [v / n_train * scale for v in gmm_train_times]

                diff = get_auc_ratio(gmm_train_times, ocsvm_train_times,
                                     same)  # training time ratio should be OCSVM/GMM
                print(i, j, list(df2.iloc[i].values))
                idxs = [_i for _i, v in enumerate(list(df2.iloc[i].values)) if 'n_components' in str(v)]
                if len(idxs) > 0:
                    comp_str = df2.iloc[i][idxs[0]].replace('n_components', 'n_comp').replace('}:', '')
                    diff = diff + f" ({comp_str})"
                _line2.append(diff)
                # _line_str += '\n' + diff

                ocsvm_test_times = get_all_aucs(df1.iloc[i][j + 2])
                gmm_test_times = get_all_aucs(df2.iloc[i][j + 2])
                # get 'number of points of the test set': df1.iloc[i][3] = ' X_test_shape: (514-33)'
                n_test = int(df1.iloc[i][3].split('(')[1].split('-')[0])
                # get the test time for each point
                ocsvm_test_times = [v / n_test * scale for v in ocsvm_test_times]
                gmm_test_times = [v / n_test * scale for v in gmm_test_times]
                diff = get_auc_ratio(gmm_test_times, ocsvm_test_times, same)  # testing time ratio should be OCSVM/GMM
                _line3.append(diff)
                # _line_str += '\n' + diff
                # _line.append(_line_str)

            elif j == start_col + 1 or j == start_col + 2:
                continue
            else:
                _line.append(df2.iloc[i].values.tolist()[j])
                _line2.append('')
                _line3.append('')
        values.append(_line)
        values.append(_line2)
        values.append(_line3)

        values.append([''] * len(_line))
    return values


def improvement(resulst_xlsx, feat_set='iat', out_file='-improvement.xlsx'):
    xls = pd.ExcelFile(resulst_xlsx)
    # Now you can list all sheets in the file
    print(f'xls.sheet_names:', xls.sheet_names)
    with ExcelWriter(out_file + '-tmp.xlsx') as writer:
        # for _, sheet_name in enumerate(x):
        #     print(sheet_name)
        #     if sheet_name not in ['KJL-GMM']:
        #         continue
        results = []
        for i, sheet_name in enumerate(xls.sheet_names):

            print(f'\n\n----{i}, {sheet_name}')
            if i == 0:  # baseline
                header = 5
                baseline_sheet_name = sheet_name
                df1 = pd.read_excel(resulst_xlsx, sheet_name=baseline_sheet_name, header=header,
                                    index_col=None)  # index_col=False: not use the first columns as index

                values = get_diff(df1, df1, feat_set=feat_set, same=True)
                values.insert(0, [sheet_name if _i == 0 else '' for _i in range(len(values[0]))])
                pd.DataFrame(values).to_excel(writer, sheet_name=baseline_sheet_name, index=False, header=False)

                startrow = len(values) + 5
                _each_total_rows = len(values) + 5

                results.append(values)
                continue
            try:
                # header = 5: it will skip the first 5 row
                df2 = pd.read_excel(resulst_xlsx, sheet_name=sheet_name, header=header,
                                    index_col=None)  # index_col=False: not use the first columns as index
                values = get_diff(df1, df2, feat_set=feat_set, same=False)
                values.insert(0, [sheet_name if _i == 0 else '' for _i in range(len(values[0]))])
                # break
            except Exception as e:
                traceback.print_exc()
                values = [sheet_name]
                values.extend([] * _each_total_rows)

            # Generate dataframe from list and write to xlsx.
            pd.DataFrame(values).to_excel(writer, sheet_name=baseline_sheet_name, index=False,
                                          startrow=startrow)
            startrow += _each_total_rows
            results.append(values)

        writer.save()

    with ExcelWriter(out_file) as writer:

        v_tmp = []
        for i, values in enumerate(results):
            if i == 0:
                header = ['Dataset', 'X_train(size-dim)', 'X_test(size-dim)', values[0][0]]
                # 'Best_auc', 'Ratio(OCSVM/method)'
                for row_vs in values:
                    vs = []
                    for v in [row_vs[0], row_vs[2], row_vs[3], row_vs[7]]:

                        v = find_data_mapping(v)

                        if 'shape: (' in v:
                            v = v.split('(')[1].split(')')[0]
                        vs.append(v)
                    v_tmp.append(vs)
            else:
                header.append(values[0][0])
                for row, _ in enumerate(v_tmp):
                    v_tmp[row].append(values[row][7])

            # for row, _ in enumerate(v_tmp):
            #     # for each algorithm, it has len(v_tmp) rows
            #     if i == 0:
            #         # header.append('dataset', 'X_train(size-dime)', 'X_test(size-dime)', values[row][0])
            #         header.append(values[row][1, 2, 3])
            #         v_tmp[row].append(values[row][4])
            #     else:
            #         if row ==0:
            #             header.append(values[row][0])
            #         v_tmp[row].append(values[row][4])

        # Generate dataframe from list and write to xlsx.
        v_tmp.insert(0, header)
        pd.DataFrame(v_tmp).to_excel(writer, sheet_name=baseline_sheet_name, index=False,
                                     startrow=2)
        writer.save()

    print('finished')
    return out_file


datasets = [
    # # # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5',
    'DS10_UNB_IDS/DS12-srcIP_192.168.10.8',
    # 'DS10_UNB_IDS/DS13-srcIP_192.168.10.9',
    # 'DS10_UNB_IDS/DS14-srcIP_192.168.10.14',
    # 'DS10_UNB_IDS/DS15-srcIP_192.168.10.15',
    # # # # #
    # # # # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1',
    # # # # # #
    'DS40_CTU_IoT/DS41-srcIP_10.0.2.15',
    # # #
    # # # # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
    # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50',
    'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165',
    # 'DS50_MAWI_WIDE/DS53-srcIP_203.78.4.32',
    # 'DS50_MAWI_WIDE/DS54-srcIP_222.117.214.171',
    # 'DS50_MAWI_WIDE/DS55-srcIP_101.27.14.204',
    # 'DS50_MAWI_WIDE/DS56-srcIP_18.178.219.109',
    # #
    # # #
    # # # # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20',
    'DS60_UChi_IoT/DS62-srcIP_192.168.143.42',
    # # # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43',
    # # # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48'

    # 'WRCCDC/2020-03-20',
    # 'DEFCON/ctf26',
    'ISTS/2015',
    # 'MACCDC/2012',

]

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


def find_data_mapping(value):
    for k, v in data_mapping.items():
        if k in value:
            value = v
            break

    return value


def dat2xlxs(in_file, out_file):
    results = load_data(in_file)

    new_results = {}
    # methods_mapping = {
    #     'detector_name_OCSVM-gs_True-kjl_False-quickshift_False-meanshift_False-nystrom_False': 'OCSVM',
    #     # 'detector_name_GMM-covariance_type_full-gs_True-kjl_False-meanshift_False-quickshift_False-nystrom_False':'GMM-full',
    #     # 'detector_name_GMM-covariance_type_diag-gs_True-kjl_False-meanshift_False-quickshift_False-nystrom_False':'GMM-diag',
    #     # 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-meanshift_False-quickshift_False-nystrom_False':'GMM-full-kjl',
    #     # 'detector_name_GMM-covariance_type_diag-gs_True-kjl_True-meanshift_False-quickshift_False-nystrom_False':'GMM-diag-kjl',
    #     # 'detector_name_GMM-covariance_type_full-gs_True-kjl_False-meanshift_False-quickshift_False-nystrom_True': 'GMM-full-nystrom',
    #     # 'detector_name_GMM-covariance_type_diag-gs_True-kjl_False-meanshift_False-quickshift_False-nystrom_True': 'GMM-diag-nystrom',
    #     # 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-meanshift_False-quickshift_True-nystrom_False': 'GMM-full-kjl-quickshift',
    #     # 'detector_name_GMM-covariance_type_diag-gs_True-kjl_True-meanshift_False-quickshift_True-nystrom_False': 'GMM-diag-kjl-quickshift',
    #     # 'detector_name_GMM-covariance_type_full-gs_True-kjl_True-meanshift_True-quickshift_False-nystrom_False': 'GMM-full-kjl-meanshift',
    #     # 'detector_name_GMM-covariance_type_diag-gs_True-kjl_True-meanshift_True-quickshift_False-nystrom_False': 'GMM-diag-kjl-meanshift',
    #
    #     'detector_name_GMM-covariance_type_full-gs_True-kjl_False-nystrom_False-quickshift_False-meanshift_False': 'GMM-full',
    #     'detector_name_GMM-covariance_type_diag-gs_True-kjl_False-nystrom_False-quickshift_False-meanshift_False': 'GMM-diag',
    #     'detector_name_GMM-covariance_type_full-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False': 'GMM-full-kjl',
    #     'detector_name_GMM-covariance_type_diag-gs_True-kjl_True-quickshift_False-meanshift_False-nystrom_False': 'GMM-diag-kjl',
    #     'detector_name_GMM-covariance_type_full-gs_True-kjl_False-nystrom_True-quickshift_False-meanshift_False': 'GMM-full-nystrom',
    #     'detector_name_GMM-covariance_type_diag-gs_True-kjl_False-nystrom_True-quickshift_False-meanshift_False': 'GMM-diag-nystrom',
    #     'detector_name_GMM-covariance_type_full-gs_True-kjl_True-meanshift_False-quickshift_True-nystrom_False': 'GMM-full-kjl-quickshift',
    #     'detector_name_GMM-covariance_type_diag-gs_True-kjl_True-meanshift_False-quickshift_True-nystrom_False': 'GMM-diag-kjl-quickshift',
    #     'detector_name_GMM-covariance_type_full-gs_True-kjl_True-meanshift_True-quickshift_False-nystrom_False': 'GMM-full-kjl-meanshift',
    #     'detector_name_GMM-covariance_type_diag-gs_True-kjl_True-meanshift_True-quickshift_False-nystrom_False': 'GMM-diag-kjl-meanshift',
    #
    # }

    GMM_covariance_type = 'diag'
    methods_mapping = {
        'case1': 'OCSVM',
        'case2': f'GMM-{GMM_covariance_type}',
        'case3': f'GMM-{GMM_covariance_type}-kjl',
        'case4': f'GMM-{GMM_covariance_type}-nystrom',
        'case5': f'GMM-{GMM_covariance_type}-kjl-quickshift',
        'case6': f'GMM-{GMM_covariance_type}-kjl-meanshift',
    }

    with ExcelWriter(out_file) as writer:

        for sheet_name in ['OCSVM']:
            values = []
            for i, ((data_key, method_key), (best_values, mid_values)) in enumerate(results.items()):

                if methods_mapping[method_key] != sheet_name:
                    continue

                print(f'\n\n----{i}, {sheet_name}')
                try:
                    header = 5
                    X_train_shape = best_values['X_train_shape']
                    X_test_shape = best_values['X_test_shape']
                    aucs = best_values['aucs']
                    train_times = best_values['train_times']
                    test_times = best_values['test_times']
                    aucs_str = "-".join([str(v) for v in aucs])
                    train_times_str = "-".join([str(v) for v in train_times])
                    test_times_str = "-".join([str(v) for v in test_times])
                    params = best_values['params']
                    _suffex = ''

                    line = f'{data_key}, {X_train_shape}, {X_test_shape}, => aucs:{aucs_str}, train_times:{train_times_str}, test_times:{test_times_str}, with params: {params}: {_suffex}'
                    values.append(line)
                    pd.DataFrame(values).to_excel(writer, sheet_name, index=False, header=False)

                    startrow = len(values) + 5
                    _each_total_rows = len(values) + 5


                except Exception as e:
                    traceback.print_exc()
                    values = [sheet_name]
                    values.extend([] * _each_total_rows)

            writer.save()

    return out_file


#
# def results2xlsx(in_file, out_file = ''):
#     if out_file == '':
#         out_file = in_file + '.xlsx'
#     out_file = dat2xlxs(in_file, out_file)
#
#     out_file = improvement(out_file, feat_set='iat_size', out_file=os.path.splitext(out_file)[0] + '-ratio.xlsx')
#     return out_file


def dat2xlsx2(in_file=''):
    file_type = 'ratio_variance'

    if file_type == '.dat':
        out_file = in_file + '.xlsx'
        dat2xlxs(in_file, out_file)

    elif file_type == 'ratio_variance':
        # result_xlsl = 'out/data_kll/iat_size/header:False/KJL-OCSVM-GMM-iat_size-header_False-gs_True_grid_search-20200523-all.xlsx'
        # result_xlsl = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200630.xlsx'
        # result_xlsl = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200702.xlsx'
        result_xlsl = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200729.xlsx'
        # result_xlsl = 'out/data_kjl/report/all_results-20200829.csv.xlsx'
        out_file = os.path.splitext(result_xlsl)[0] + '-ratio.xlsx'
        improvement(result_xlsl, feat_set='iat_size', out_file=out_file)


if __name__ == '__main__':
    in_file = 'out/data_kjl/report/all_results.dat'
    dat2xlsx2(in_file)
