""" improvement: average auc (with KJL) - average auc (without KJL)

"""
import os
import traceback

import numpy as np
import pandas as pd
from pandas import ExcelWriter


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

    # divide two independent random variables
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
            start_col = 4
            if j == start_col:
                _line_str = ''
                wo_kjl = df1.iloc[i][j]
                w_kjl = df2.iloc[i][j]
                # diff = _get_diff(wo_kjl, w_kjl, same=same)  # auc diff
                diff = _get_diff(w_kjl, wo_kjl, same=same)  # auc diff
                _line.append(diff)
                # _line_str +=diff

                wo_kjl = df1.iloc[i][j + 1]
                w_kjl = df2.iloc[i][j + 1]
                diff = _get_diff(wo_kjl, w_kjl, same=same)  # training time
                print(i, j, list(df2.iloc[i].values))
                idxs = [_i for _i, v in enumerate(list(df2.iloc[i].values)) if 'n_components' in str(v)]
                if len(idxs) > 0:
                    comp_str = df2.iloc[i][idxs[0]].replace('n_components', 'n_comp').replace('}:','')
                    diff = diff + f" ({comp_str})"
                _line2.append(diff)
                # _line_str += '\n' + diff

                wo_kjl = df1.iloc[i][j + 2]
                w_kjl = df2.iloc[i][j + 2]
                diff = _get_diff(wo_kjl, w_kjl, same=same)  # testing time
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

        v_tmp=[]
        for i, values in enumerate(results):
            if i == 0:
                header=['Dataset', 'X_train(size-dim)', 'X_test(size-dim)', values[0][0]]
                # 'Best_auc', 'Ratio(OCSVM/method)'
                for row_vs in values:
                    vs = []
                    for v in [row_vs[0], row_vs[2], row_vs[3], row_vs[4]]:

                        v = find_data_mapping(v)

                        if 'shape: (' in v:
                            v= v.split('(')[1].split(')')[0]
                        vs.append(v)
                    v_tmp.append(vs)
            else:
                header.append(values[0][0])
                for row, _ in enumerate(v_tmp):
                    v_tmp[row].append(values[row][4])

            # for row, _ in enumerate(v_tmp):
            #     # for each algorithm, it has len(v_tmp) rows
            #     if i == 0:
            #         # header.append('datasets', 'X_train(size-dime)', 'X_test(size-dime)', values[row][0])
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
   'DS50_MAWI_WIDE/DS52-srcIP_203.78.7.165':'MAWI_PC2',
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


if __name__ == '__main__':
    # result_xlsl = 'out/data_kll/iat_size/header:False/KJL-OCSVM-GMM-iat_size-header_False-gs_True_grid_search-20200523-all.xlsx'
    result_xlsl = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200630.xlsx'
    result_xlsl = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200702.xlsx'
    result_xlsl = 'out/data_kjl/report/KJL-OCSVM-GMM-gs_True-20200725.xlsx'
    out_file = os.path.splitext(result_xlsl)[0] + '-ratio.xlsx'
    improvement(result_xlsl, feat_set='iat_size', out_file=out_file)
