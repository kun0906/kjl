import os
import os.path as pt
import pickle
import traceback
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os.path as pth

def split_train_test(normal_data, abnormal_data, train_size=0.8, test_size=150 * 2, random_state=42, debug=False):
    """Split train and test set

    Parameters
    ----------
    normal_data
    abnormal_data
    show

    Returns
    -------

    """
    # split train/test
    n_abnormal = abnormal_data.shape[0] if test_size == -1 else int(test_size / 2)

    # avoid using choice, which doesn't has random seed
    # normal_test_idx = np.random.choice(normal_data.shape[0], size=n_abnormal, replace=False)
    normal_test_idx = resample(range(normal_data.shape[0]), n_samples=n_abnormal, random_state=random_state,
                               replace=False)
    normal_test_idx = np.in1d(range(normal_data.shape[0]), normal_test_idx)  # return boolean idxes
    #
    train_normal, test_normal = normal_data[~normal_test_idx], normal_data[normal_test_idx]

    # for abnormal
    if n_abnormal < abnormal_data.shape[0]:
        abnormal_test_idx = resample(range(abnormal_data.shape[0]), n_samples=n_abnormal, random_state=random_state,
                                     replace=False)
        abnormal_test_idx = np.in1d(range(abnormal_data.shape[0]), abnormal_test_idx)  # return boolean idxes
        #
        _, test_abnormal = abnormal_data[~abnormal_test_idx], abnormal_data[
            abnormal_test_idx]  # return boolean idxes
    else:
        test_abnormal = abnormal_data

    if train_size == -1:
        n_normal = train_normal.shape[0]
    elif 0 < train_size <= 1:
        n_normal = int(train_normal.shape[0] * train_size)
    elif train_normal.shape[0] > train_size:
        n_normal = train_size
    else:  # train_normal.shape[0] < train_size:
        n_normal = train_normal.shape[0]

    train_normal = train_normal[: n_normal, :]

    X_train_normal = train_normal[:, :-1]
    X_train = X_train_normal
    y_train = train_normal[:, -1]

    X_test_normal = test_normal[:, :-1]
    y_test_normal = test_normal[:, -1].reshape(-1, 1)
    X_test_abnormal = test_abnormal[:, :-1]
    y_test_abnormal = test_abnormal[:, -1].reshape(-1, 1)
    X_test = np.vstack((X_test_normal, X_test_abnormal))
    # normal and abnormal have the same size in the test set
    y_test = np.vstack((y_test_normal, y_test_abnormal)).flatten()

    print("train.shape: {}, test.shape: {}".format(X_train.shape, X_test.shape))
    if debug:  # DEBUG=10
        data_info(X_train, name='X_train')
        data_info(X_test, name='X_test')

    return X_train, y_train, X_test, y_test


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    return data


def data_info(data, name=''):
    """Show data basic information

    Parameters
    ----------
    data
    name

    Returns
    -------

    """

    columns = ['col_' + str(i) for i in range(data.shape[1])]
    dataset = pd.DataFrame(data=data, index=range(data.shape[0]), columns=columns)
    print(f'\n{name}.shape: {data.shape}')
    print(f'\n{dataset.describe()}')
    print(dataset.info(verbose=True))


def extract_data(normal_pth, abnormal_pth, meta_data={}):
    """Get normal and abnormal data from csv
    # NORMAL(inliers): 0, ABNORMAL(outliers): 1
    Returns
    -------
        normal_data
        abnormal_data

    """
    NORMAL = 0  # Use 0 to label normal data
    ABNORMAL = 1  # Use 1 to label abnormal data

    # Normal and abnormal are the same size in the test set
    if meta_data['train_size'] <= 0:
        n_normal = -1
        n_abnormal = -1
    else:
        n_abnormal = int(meta_data['test_size'] / 2)
        n_normal = meta_data['train_size'] + n_abnormal
    start = meta_data['idxs_feat'][0]
    end = meta_data['idxs_feat'][1]

    def _label_n_combine_data(X, size=-1, data_type='normal'):
        if size == -1:
            size = X.shape[0]
        # idx = np.random.randint(0, high=X.shape[0], size=size)    # random choose sample
        # X = X[idx, :]
        if data_type.upper() == 'normal'.upper():
            y = np.ones((X.shape[0], 1)) * NORMAL
        elif data_type.upper() == 'abnormal'.upper():
            y = np.ones((X.shape[0], 1)) * ABNORMAL
        else:
            # todo
            print(f"KeyError: {data_type}")
            raise KeyError(f'{data_type}')
        _data = np.hstack((X, y))
        nans = np.isnan(_data).any(axis=1)  # remove NaNs
        _data = _data[~nans]
        return _data

    # Get normal data
    try:
        if end == -1:
            X_normal = genfromtxt(normal_pth, delimiter=',', skip_header=0)[:, start:]  # skip_header=1
            X_abnormal = genfromtxt(abnormal_pth, delimiter=',', skip_header=0)[:, start:]
        else:
            X_normal = genfromtxt(normal_pth, delimiter=',', skip_header=0)[:, start:end]  # skip_header=1
            X_abnormal = genfromtxt(abnormal_pth, delimiter=',', skip_header=0)[:, start:end]
    except FileNotFoundError as e:
        print(f'FileNotFoundError: {e}')
        raise FileNotFoundError(e)

    normal_data = _label_n_combine_data(X_normal, size=n_normal, data_type='normal')  # (X, y)
    abnormal_data = _label_n_combine_data(X_abnormal, size=n_abnormal, data_type='abnormal')  # (X, y)

    # data={'X_train':'', 'y_train':'', 'X_test':'', 'y_test':''}
    # data = {'normal_data': normal_data, 'abnormal_data': abnormal_data,
    #         'label': {'NORMAL': NORMAL, 'ABNORMAL': ABNORMAL}}

    return normal_data, abnormal_data


def dump_data(data, out_file='data.dat'):
    """

    Parameters
    ----------
    data
    out_file

    Returns
    -------

    """
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def _get_line(result_each, feat_set=''):
    """Get each feat_set result and format it

    Parameters
    ----------
    result_each: dict
         result_each=(_best, '')
    feat_set

    Returns
    -------

    """

    try:
        value = result_each
        X_train_shape = str(value['X_train_shape']).replace(', ', '-')
        X_test_shape = str(value['X_test_shape']).replace(', ', '-')

        mu_auc = np.mean(value['aucs'])
        std_auc = np.std(value['aucs'])

        mu_train_time = np.mean(value['train_times'])
        std_train_time = np.std(value['train_times'])

        mu_test_time = np.mean(value['test_times'])
        std_test_time = np.std(value['test_times'])

        line = f'{feat_set}(auc: ' + f'{mu_auc:0.4f}' + '+/-' + f'{std_auc:0.4f}' + \
               ',train_time: ' + f'{mu_train_time:0.5f}' + '+/-' + f'{std_train_time:0.5f}' + \
               ',test_time: ' + f'{mu_test_time:0.5f}' + '+/-' + f'{std_test_time:0.5f})'
        suffex = ''
    except (Exception, KeyError, ValueError) as e:
        X_train_shape = '(0-0)'
        X_test_shape = '(0-0)'
        line = f',{feat_set}(-)'
        suffex = ', '
        msg = f'{_get_line.__name__}, error:{e}, result_each[{feat_set}]: {result_each}'
        print(msg)

    prefix = f'X_train_shape: {X_train_shape}, X_test_shape: {X_test_shape}'

    return prefix, line, suffex


def save_each_result(data, case_str, out_file=None):
    if not pt.exists(pt.dirname(out_file)): os.makedirs(pt.dirname(out_file))
    # dump first
    dump_data(data, pt.splitext(out_file)[0] + '.dat')

    with open(out_file, 'w') as f:
        aucs = data['aucs']
        train_times = data['train_times']
        test_times = data['test_times']
        params = data['params']

        _prefix, _line, _suffex = _get_line(data, feat_set='iat_size')
        aucs_str = "-".join([str(v) for v in aucs])
        train_times_str = "-".join([str(v) for v in train_times])
        test_times_str = "-".join([str(v) for v in test_times])

        line = f'{case_str}, {_prefix}, {_line}, => aucs:{aucs_str}, train_times:{train_times_str}, test_times:{test_times_str}, with params: {params}: {_suffex}'

        f.write(line + '\n')


data_mapping = {
    # 'DS10_UNB_IDS/DS11-srcIP_192.168.10.5':'UNB_PC1',
    'DS10_UNB_IDS/DS12-srcIP_192.168.10.8': 'UNB_PC2',
    'DS10_UNB_IDS/DS13-srcIP_192.168.10.9': 'UNB_PC3',
    'DS10_UNB_IDS/DS14-srcIP_192.168.10.14': 'UNB_PC4',
    'DS10_UNB_IDS/DS15-srcIP_192.168.10.15': 'UNB_PC5',

    # 'DS20_PU_SMTV/DS21-srcIP_10.42.0.1':'SMTV',
    # # #
    'DS40_CTU_IoT/DS41-srcIP_10.0.2.15': 'CTU',
    #
    # 'DS50_MAWI_WIDE/DS51-srcIP_202.171.168.50':'MAWI',

    # 'DS60_UChi_IoT/DS61-srcIP_192.168.143.20':'GHom',
    'DS60_UChi_IoT/DS62-srcIP_192.168.143.42': 'SCam',
    # 'DS60_UChi_IoT/DS63-srcIP_192.168.143.43':'SFrig',
    # 'DS60_UChi_IoT/DS64-srcIP_192.168.143.48':'BSTCH'
}


def save_result(result, out_file):
    dump_data(result, pt.splitext(out_file)[0] + '.dat')

    with open(out_file, 'w') as f:
        keys = []
        for (in_dir, case_str), (best_results, middle_results) in result.items():
            if case_str not in keys:
                keys.append(case_str)
        print(keys)

        for key in keys:
            print('\n\n')
            for (in_dir, case_str), (best_results, middle_results) in result.items():
                # print(case_str, key)
                if case_str != key:
                    continue
                data = best_results
                try:
                    # best_auc = data['best_auc']
                    aucs = data['aucs']
                    params = data['params']
                    train_times = data['train_times']
                    test_times = data['test_times']
                    # params = data['params']

                    _prefix, _line, _suffex = _get_line(data, feat_set='iat_size')
                    # line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs: {aucs} with best_params: {params}: {_suffex}'

                    aucs_str = "-".join([str(v) for v in aucs])
                    train_times_str = "-".join([str(v) for v in train_times])
                    test_times_str = "-".join([str(v) for v in test_times])

                    line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs:{aucs_str}, train_times:{train_times_str}, test_times:{test_times_str}, with params: {params}: {_suffex}'

                except Exception as e:
                    traceback.print_exc()
                    line = ''
                f.write(line + '\n')
                print(line)
            f.write('\n')


def batch(X, y, *, step=1, stratify=True):
    if stratify:
        # keep that each batch has the same ratio of different labels
        # https://stackoverflow.com/questions/36997619/sklearn-stratified-sampling-based-on-a-column
        while len(y) > 0:
            if len(y) <= step:
                yield X, y
                X=[]
                y = []
            else:
                X, X_test, y, y_test = train_test_split(X, y, test_size=step, shuffle=True,
                                                             random_state=42, stratify=y)
                yield X_test, y_test

    else:
        # cannot make sure that each batch has the same ratio of different labels
        size = len(X)
        for i in range(0, size, step):
            if step == 1:
                yield X[i:min(i + step, size)].reshape(1, -1), y[i:min(i + step, size)]
            else:
                yield X[i:min(i + step, size)], y[i:min(i + step, size)]

#
# def get_batch_mean_varaince():
#     pass
#
#
# def get_batch_mean_covariance(x, n_samples, mean, M2):
#     """
#     https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
#     Parameters
#     ----------
#     x
#     n_samples
#     mean
#     M2: it's not variance or covariance
#         M2= np.sum(np.subtract(x, [mean] * count)**2)
#
#     Returns
#     -------
#
#     """
#
#     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
#     def update(existingAggregate, newValues):
#         if isinstance(newValues, (int, float, complex)):
#             # Handle single digits.
#             newValues = [newValues]
#
#         (count, mean, M2) = existingAggregate
#         count += len(newValues)
#         # newvalues - oldMean
#         delta = np.subtract(newValues, [mean] * len(newValues))
#         mean += np.sum(delta / count)
#         # newvalues - newMeant
#         delta2 = np.subtract(newValues, [mean] * len(newValues))
#         M2 += np.sum(delta * delta2)
#
#         return (count, mean, M2)
#
#     def finalize(existingAggregate):
#         (count, mean, M2) = existingAggregate
#         (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
#         if count < 2:
#             return float('nan')
#         else:
#             return (mean, variance, sampleVariance)
#
#     a = (n_samples, mean, M2)
#
#     mean, variance, sampleVariance = finalize(update(a, x))
#
#     return mean, sampleVariance
#



def save_result(result, out_file):
    dump_data(result, pth.splitext(out_file)[0] + '.dat')

    # with open(out_file, 'w') as f:
    #     keys = []
    #     for (in_dir, case_str), (best_results, middle_results) in result.items():
    #         if case_str not in keys:
    #             keys.append(case_str)
    #     mprint(keys)
    #
    #     for key in keys:
    #         mprint('\n\n')
    #         for (in_dir, case_str), (best_results, middle_results) in result.items():
    #             # mprint(case_str, key)
    #             if case_str != key:
    #                 continue
    #             data = best_results
    #             try:
    #                 aucs = data['aucs']
    #                 # params = data['params']
    #                 train_times = data['train_times']
    #                 test_times = data['test_times']
    #
    #                 # _prefix, _line, _suffex = _get_line(data, feat_set='iat_size')
    #                 # line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs: {aucs} with best_params: {params}: {_suffex}'
    #                 _prefix = ''
    #                 _line = ''
    #                 params = ''
    #                 _suffex = ''
    #
    #                 aucs_str = "-".join([str(v) for v in aucs])
    #                 train_times_str = "-".join([str(v) for v in train_times])
    #                 test_times_str = "-".join([str(v) for v in test_times])
    #
    #                 line = f'{in_dir}, {case_str}, {_prefix}, {_line}, => aucs:{aucs_str}, train_times:' \
    #                        f'{train_times_str}, test_times:{test_times_str}, with params: {params}: {_suffex}'
    #
    #             except Exception as e:
    #                 traceback.mprint_exc()
    #                 line = ''
    #             f.write(line + '\n')
    #             mprint(line)
    #         f.write('\n')
