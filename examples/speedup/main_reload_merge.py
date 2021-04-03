""" Main function: load built models from disk and evaluate them on test data

"""

import itertools
# Authors: kun.bj@outlook.com
# License: XXX
import os
import os.path as pth
import shutil
import traceback

import numpy as np
import numpy as np
import pandas as pd
from pandas import ExcelWriter

# from kjl import pstats
from kjl.log import get_log
from kjl.utils.data import _get_line, dump_data
from kjl.utils.tool import load_data, check_path
# create a customized log instance that can print the information.
from speedup.ratio_variance import dat2xlxs_new, improvement, dat2latex

# from scalene.scalene_profiler import Scalene

lg = get_log(level='info')

DATASETS = [
    ### Final datasets for the paper
    'UNB345_3',  # Combine UNB3, UNB3 and UNB5 attack data as attack data and only use UNB3's normal as normal data
    'CTU1',  # Two different abnormal data
    'MAWI1_2020',  # Two different normal data
    'MACCDC1',  # Two different normal data
    'SFRIG1_2020',  # Two different normal data
    'AECHO1_2020',  # Two different normal data
    'DWSHR_WSHR_2020',  # only use Dwshr normal as normal data, and combine Dwshr and wshr novelties as novelty
]

MODELS = [
    ### Algorithm name
    "OCSVM(rbf)",
    "KJL-OCSVM(linear)",
    "Nystrom-OCSVM(linear)",

    # "GMM(full)", "GMM(diag)",

    "KJL-GMM(full)",  "KJL-GMM(diag)",
    "Nystrom-GMM(full)",   "Nystrom-GMM(diag)",
    #
    # ### quickshift(QS)/meanshift(MS) are used before KJL/Nystrom projection
    # # "QS-KJL-GMM(full)", "QS-KJL-GMM(diag)",
    # # "MS-KJL-GMM(full)", "MS-KJL-GMM(diag)",
    #
    # # "QS-Nystrom-GMM(full)", "QS-Nystrom-GMM(diag)",
    # # "MS-Nystrom-GMM(full)", "MS-Nystrom-GMM(diag)",
    #
    ################################################################################################################
    # 3. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection
    "KJL-QS-GMM(full)", "KJL-QS-GMM(diag)",
    # "KJL-MS-GMM(full)", "KJL-MS-GMM(diag)"

    "Nystrom-QS-GMM(full)", "Nystrom-QS-GMM(diag)",
    # # "Nystrom-MS-GMM(full)", "Nystrom-MS-GMM(diag)"
    #
    # ################################################################################################################
    # # 4. quickshift(QS)/meanshift(MS) are used after KJL/Nystrom projection and initialize GMM (set 'GMM_is_init_all'=True)
    "KJL-QS-init_GMM(full)",  "KJL-QS-init_GMM(diag)",
    "Nystrom-QS-init_GMM(full)", "Nystrom-QS-init_GMM(diag)",

]


def _res2csv(f, data, feat_set='iat_size', data_name='', model_name=''):
    """ Format the data in one line and save it to the csv file (f)

    Parameters
    ----------
    f: file object
    data: dict
    feat_set
    data_name
    model_name

    Returns
    -------

    """
    try:
        # best_auc = data['best_auc']
        aucs = data['aucs']
        params = data['params'][-1]
        train_times = data['train_times']
        test_times = data['test_times']
        space_sizes = data['space_sizes']
        model_spaces = data['model_spaces']

        _prefix, _line, _suffex = _get_line(data, feat_set=feat_set)
        # line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs: {aucs} with best_params: {params}: {_suffex}'

        aucs_str = "-".join([str(v) for v in aucs])
        train_times_str = "-".join([str(v) for v in train_times])
        test_times_str = "-".join([str(v) for v in test_times])
        space_size_str = "-".join([str(v) for v in space_sizes])
        model_spaces_str = "-".join([str(v) for v in model_spaces])

        try:
            n_comps = [int(v['GMM_n_components']) for v in data['params']]
            mu_n_comp = np.mean(n_comps)
            std_n_comp = np.std(n_comps)
            n_comp_str = f'{mu_n_comp:.2f}+/-{std_n_comp:.2f}'
            n_comp_str2 = "-".join([str(v) for v in n_comps])
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
                params['qs_res']['tot_clusters'] = '-'.join(
                    [f'{k}:{v}' for k, v in params['qs_res']['tot_clusters'].items()])
        line = f'{data_name}|, {model_name}, {_prefix}, {_line}, => aucs:{aucs_str}, ' \
               f'train_times:{train_times_str}, test_times:{test_times_str}, n_comp: {n_comp_str}, ' \
               f'{n_comp_str2}, space_sizes: {space_size_str}|model_spaces: {model_spaces_str},' \
               f'tot_clusters:{tot_clusters_str}, {tot_clusters_str2},' \
               f'n_clusters: {n_clusters_str}, {n_clusters_str2}, with params: {params}: {_suffex}'

    except Exception as e:
        traceback.print_exc()
        line = f'{data_name}|, {model_name}, X_train, X_test, auc, train, test, => aucs:, ' \
               f'train_times:, test_times:, n_comp:, ' \
               f', space_sizes: |model_spaces: ,' \
               f'tot_clusters:, ,' \
               f'n_clusters: , , with params: : '
    f.write(line + '\n')
    return line


def res2csv(result, out_file, feat_set='iat_size'):
    """ save results to a csv file

    Parameters
    ----------
    result: dict
        all results
    out_file: csv

    feat_set: str
        "iat_size"

    Returns
    -------
    out_file: csv

    """
    outs = {}
    with open(out_file, 'w') as f:
        for data_name, values in result.items():
            for model_name, vs in values.items():
                line = _res2csv(f, vs, feat_set, data_name, model_name)

                line = np.asarray(line.split(','))
                if model_name not in outs.keys():
                    outs[model_name] = [line]
                else:
                    outs[model_name].append(line)
    lg.info(out_file)
    return out_file, outs


def _main(in_dir, data_name, model_name, feat_set='iat_size', is_gs=True, is_header=False, GMM_covariance_type='full'):
    if 'OCSVM' in model_name:
        GMM_covariance_type = 'None'
    # elif 'diag' in model_name:
    #     GMM_covariance_type = 'diag'
    # else:
    #     GMM_covariance_type = 'full'

    in_dir = pth.join(in_dir,
                      'src_dst',
                      feat_set + f"-header_{is_header}",
                      data_name,
                      "before_proj_False" + \
                      f"-gs_{is_gs}",
                      model_name + "-std_False"
                      + "_center_False" + "-d_5" \
                      + f"-{GMM_covariance_type}")

    in_file = f'{in_dir}/{data_name}-{model_name}.dat'
    try:
        out = load_data(in_file)
        out['train_times'] = [0] * len(out['test_times'])
        out['X_train_shape'] = 'X_train_shape:(5000-)'
        out['params'] = [{}]
    except Exception as e:
        traceback.print_exc()
        out = {}
        out['train_times'] = [0]
        out['test_spaces'] = [0]
        out['model_spaces'] = [0]
        out['X_train_shape'] = 'X_train_shape:(5000-)'
        out['params'] = [{}]
    out['out_dir'] = in_dir
    return out

def _main0(in_dir, feat_set, is_gs, is_header, GMM_covariance_type='full'):
    res = {}
    # feat_set = 'iat_size'
    # in_dir = 'speedup/out/kjl_serial_ind_32_threads-cProfile_perf_counter'
    # in_dir = 'speedup/out/pi_out/models_res'
    # in_dir = 'speedup/out/pi_out31/models_res'
    # # in_dir = 'speedup/out/neon_out/models_res'
    # in_dir = 'speedup/out/neon_out31/models_res'
    # # in_dir = 'speedup/out/calumet_out/models_res'
    # # in_dir = 'speedup/out/calumet_out2/models_res'
    # in_dir = 'speedup/out/nano_out/models_res'
    # in_dir = 'speedup/out/nano_out31/models_res'
    # # in_dir = 'speedup/out/kjl_joblib_parallel_30'

    # in_dir = 'speedup/paper_data/pi_out31/models_res'
    # in_dir = 'speedup/paper_data/nano_out31/models_res'
    # in_dir = 'speedup/paper_data/neon_out31/models_res'
    # in_dir = 'speedup/paper_data/calumet_out31/models_res'
    # Load models and evaluate them on test data
    if 'neon_train_out' not in in_dir:
        tot_spaces = {}
        datasets = []
        for i, (data_name, model_name) in enumerate(itertools.product(DATASETS, MODELS)):
            if 'GMM' in model_name and GMM_covariance_type not in model_name: continue
            out = _main(in_dir, data_name, model_name, feat_set, is_gs, is_header, GMM_covariance_type)
            if data_name not in res.keys():
                res[data_name] = {model_name: out}
                tot_spaces[data_name] =  [[out['test_spaces'][0]], out['model_spaces']]
            else:
                res[data_name][model_name] = out
                tot_spaces[data_name].append(out['model_spaces'])

        lg.debug(res)
        lg.info(tot_spaces)
        tot_space = 0
        for k, vs in tot_spaces.items():
            tot_space += sum([sum(v) for v in vs])
        lg.info(f'*** tot_space: {tot_space} KB, {tot_space/1e+3} MB')

        # Save results
        out_dir = pth.join(in_dir,
                          'src_dst',
                          feat_set + f"-header_{is_header}",
                          "before_proj_False" + \
                          f"-gs_{is_gs}", f'std_False_center_False-d_5-{GMM_covariance_type}')


        out_file = f'{out_dir}/res.csv'
        check_path(out_file)
        # save as csv
        _, outs = res2csv(res, out_file, feat_set=feat_set)
        out_file_dat = out_file + '.dat'
        dump_data(outs, out_file=out_file_dat)
        lg.info(out_file)
        # save as xlsx
        out_xlsx = dat2xlxs_new(out_file_dat, out_file=out_file_dat + '.xlsx', models=MODELS)
        # compute ratio OCSVM/GMM
        out_xlsx_ratio = improvement(out_xlsx, feat_set=feat_set,
                                     out_file=os.path.splitext(out_file_dat)[0] + '-ratio.xlsx')
        print(out_xlsx_ratio)
        # for paper
        out_latex = dat2latex(out_xlsx_ratio, out_file=os.path.splitext(out_file_dat)[0] + '-latex.xlsx')
        print(out_latex)

        lg.info('finish!')
        return os.path.dirname(out_latex)
    else:

        # Save results
        out_dir = pth.join(in_dir,
                           'src_dst',
                           feat_set + f"-header_{is_header}",
                           "before_proj_False" + \
                           f"-gs_{is_gs}", f'std_False_center_False-d_5-{GMM_covariance_type}')
        return out_dir

def _space_format(v='u +/- std'):
    v = v.split('+/-')
    return '\\Cell{' + f'{v[0]}' + '$\\pm$\\\\' + f'{v[1]}' + '}'

def xlsx2latex(res, out_file='res.txt'):

    for in_dir in res.keys():
        in_file = res[in_dir] + '/res.csv-latex.xlsx'
        xls = pd.ExcelFile(in_file)
        break
    DATASETS = ['UNB', 'CTU', 'MAWI', '\\Cell{MAC-\\\\CDC}', 'SFRIG', '\\Cell{AEC-\\\\HO}', '\\Cell{DWS-\\\\HR}']
    with open(out_file, 'w') as f:
        for _, sheet_name in enumerate(xls.sheet_names):
            try:
                if sheet_name == 'OCSVM':
                    outs = []
                    for in_dir in res.keys():
                        in_file = res[in_dir] + '/res.csv-latex.xlsx'
                        df = pd.read_excel(in_file, header=None, sheet_name=sheet_name,
                                           index_col=None)  # index_col=False: not use the first columns as index
                        # df.dropna(axis='rows', how='all', inplace=True)
                        # df = df.fillna(value='-')
                        outs.append(df.values.tolist())

                    line = "Dataset   & UNB      & CTU  & MAWI         & MACCDC       & SFRIG  & AECHO   &DWSHR " \
                           + ' \\\\' + '\n\midrule'
                    line += '\nCommon AUC &' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[0][1][1:]])  \
                           + ' \\\\' + '\n\midrule' \
                                       '\nServer train time (ms) &' + " & ".join(
                        [v.replace('+/-', '$\pm$').split('(')[0] for i_, v in enumerate(outs[3][2][1:])]) \
                            + ' \\\\' + '\n\midrule'\
                        '\n\Cell{Test time (ms):\\\\RSPI} &' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[0][3][1:]]) \
                           + ' \\\\' + '\n\cmidrule{2-8}' \
                      '\nNANO&' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[1][3][1:]]) \
                           + ' \\\\' + '\n\cmidrule{2-8}' \
                      '\nServer&' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[2][3][1:]]) \
                           + ' \\\\'+ '\n\midrule'\
                      '\n\\Cell{Space (kB)} &' + " & ".join([_space_format(v) for v in outs[0][4][1:]]) \
                           + ' \\\\'+ '\n'
                elif sheet_name == 'GMMs':
                    outs = []
                    for in_dir in res.keys():
                        in_file = res[in_dir] + '/res.csv-latex.xlsx'
                        df = pd.read_excel(in_file, header=None, sheet_name=sheet_name,
                                           index_col=None)  # index_col=False: not use the first columns as index
                        # df.dropna(axis='rows', how='all', inplace=True)
                        # df = df.fillna(value='-')
                        outs.append(df.values.tolist())
                    i = 0
                    line  = '\n\n'
                    j = 0
                    while i < len(outs[0]):
                        line += '\n\multirow{4}{*}{'+f'{DATASETS[j]}' +'}   & AUC speedup &' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[0][i][1:]])\
                                + ' \\\\' + '\n\cmidrule{2-6}' \
                            '\n& Server train speedup &' + " & ".join([v.replace('+/-', '$\pm$') for i_, v in enumerate(outs[3][i + 1]) if i_ not in [0, 1, 2, 5, 6]]) \
                                + ' \\\\' + \
                            '\n& \\Cell{Test speedup:\\\\RSPI} &' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[0][i+2][1:]]) \
                                + ' \\\\' + \
                            '\n& NANO &' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[1][i+2][1:]]) \
                                + ' \\\\' + \
                            '\n& Server  &' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[2][i+2][1:]]) \
                                + ' \\\\'   + '\n\cmidrule{2-6}' \
                            '\n& Space saving &' + " & ".join([v.replace('+/-', '$\pm$') for v in outs[0][i+3][1:]]) \
                                + ' \\\\'   + '\n\cmidrule{2-6}'
                        line +='\n\midrule'
                        # print(i)
                        i += 4
                        j +=1
                elif sheet_name == 'n_components':
                    outs = []
                    for in_dir in res.keys():
                        in_file = res[in_dir] + '/res.csv-latex.xlsx'
                        df = pd.read_excel(in_file, header=None, sheet_name=sheet_name,
                                           index_col=None)  # index_col=False: not use the first columns as index
                        # df.dropna(axis='rows', how='all', inplace=True)
                        # df = df.fillna(value='-')
                        outs.append(df.values.tolist())
                    line = '\n\n'
                    for i, vs in enumerate(outs[-1]):
                        line +=f"{DATASETS[i]} & " + " & ".join([v.replace('+/-', '$\pm$') for i_, v in enumerate(vs)  if i_ not in [0, 1,2,3]]) + '\\\\ \n'
                else:
                    outs = []
                    for in_dir in res.keys():
                        in_file = res[in_dir] + '/res.csv-latex.xlsx'
                        df = pd.read_excel(in_file, header=None, sheet_name=sheet_name,
                                           index_col=None)  # index_col=False: not use the first columns as index
                        # df.dropna(axis='rows', how='all', inplace=True)
                        # df = df.fillna(value='-')
                        outs.append(df.values.tolist())
                    line = '\n\n'
                    for vs in outs[-1]:
                        line += " & ".join([str(v) for i_, v in enumerate(vs)]) + '\\\\ \n'
            except Exception as e:
                traceback.print_exc()
                line = ''
            f.write(line+'\n')

    print(out_file)
    return out_file



def main(feat_set = 'iat_size', is_gs = True, is_header=False, GMM_covariance_type='full'):
    # in_dir = 'speedup/out/kjl_serial_ind_32_threads-cProfile_perf_counter'
    # in_dir = 'speedup/out/pi_out/models_res'
    # in_dir = 'speedup/out/pi_out31/models_res'
    # # in_dir = 'speedup/out/neon_out/models_res'
    # in_dir = 'speedup/out/neon_out31/models_res'
    # # in_dir = 'speedup/out/calumet_out/models_res'
    # # in_dir = 'speedup/out/calumet_out2/models_res'
    # in_dir = 'speedup/out/nano_out/models_res'
    # in_dir = 'speedup/out/nano_out31/models_res'
    # # in_dir = 'speedup/out/kjl_joblib_parallel_30'

    # in_dir = 'speedup/paper_data/pi_out31/models_res'
    # in_dir = 'speedup/paper_data/nano_out31/models_res'
    # in_dir = 'speedup/paper_data/neon_out31/models_res'
    # in_dir = 'speedup/paper_data/calumet_out31/models_res'

    devices = ['speedup/paper_data/rspi_out/out/models_res',
               'speedup/paper_data/nano_out/out/models_res',
               'speedup/paper_data/neon_out/out/models_res',
               'speedup/paper_data/neon_train_out/out'
               ]
    res = {}
    for in_dir in devices:
        # in_dir = f'speedup/paper_data/{device}/models_res'
        out_dir = _main0(in_dir, feat_set, is_gs, is_header, GMM_covariance_type)
        # out_dir = in_dir
        res[in_dir] = out_dir

    print(res)
    # merge 3 xlsx to latex
    xlsx2latex(res, out_file = os.path.join('speedup/paper_data', f'{feat_set}-header_{is_header}-gs_{is_gs}-{GMM_covariance_type}-out.txt'))

if __name__ == '__main__':
    import subprocess
    print(os.getcwd())  # it must be ".kjl/examples/
    root_dir = 'speedup/paper_data'
    for tmp_dir in ['rspi_out', 'nano_out', 'neon_out', 'neon_train_out']:
        tmp_dir = os.path.join(root_dir, tmp_dir)
        cmd = f'unzip -q {tmp_dir}.zip -d {tmp_dir}'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # subprocess.run(["unzip", "-q", "" f"{tmp_dir}"])
        print(cmd)
        os.system(cmd)

    demo = False
    if demo:
        for feat_set in ['iat_size']:  # 'iat_size',
            for is_gs in [False]:
                for is_header in [False]:
                    for GMM_covariance_type in ['full']:
                        if feat_set == 'iat_size' and is_header: continue
                        if feat_set == 'stats' and not is_header: continue
                        main(feat_set, is_gs, is_header, GMM_covariance_type)
    else:
        for feat_set in ['iat_size', 'stats']:  # 'iat_size',
            for is_gs in [True, False]:
                for is_header in [True, False]:
                    for GMM_covariance_type in ['full', 'diag']:
                        if feat_set == 'iat_size' and is_header: continue
                        if feat_set == 'stats' and not is_header: continue
                        main(feat_set, is_gs, is_header, GMM_covariance_type)
