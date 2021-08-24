
"""Useful tools includes 'data_info', 'dump_data', etc.

"""
# Authors: kun.bj@outlook.com
#
# License: XXX

import inspect
import os
import pickle
import subprocess
from time import time
from datetime import datetime
from functools import wraps

import pandas as pd


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func

def load(in_file):
    """load data from file

    Parameters
    ----------
    in_file: str
        input file path

    Returns
    -------
    data:
        loaded data
    """
    with open(in_file, 'rb') as f:
        data = pickle.load(f)

    return data

def dump(data, out_file=''):
    """Save data to file

    Parameters
    ----------
    data: any data

    out_file: str
        out file path
    Returns
    -------

    """

    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

    return out_file


def data_info(data=None, name='data'):
    """Print data basic information

    Parameters
    ----------
    data: array

    name: str
        data name

    Returns
    -------

    """

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 100)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)  # without scientific notation

    columns = ['col_' + str(i) for i in range(data.shape[1])]
    dataset = pd.DataFrame(data=data, index=range(data.shape[0]), columns=columns)
    print(f'{name}.shape: {data.shape}')
    print(dataset.describe())
    print(dataset.info(verbose=True))


def check_path(dir_path):
    """Check if a path is existed or not.
     If the path doesn't exist, then create it.

    Parameters
    ----------
    file_path: str

    overwrite: boolean (default is True)
        if the path exists, delete all data in it and create a new one

    Returns
    -------

    """
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def timing(func):
    """Calculate the execute time of the given func"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        st = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print(f'\'{func.__name__}()\' starts at {st}')
        result = func(*args, **kwargs)
        end = time.time()
        ed = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
        tot_time = (end - start) / 60
        tot_time = float(f'{tot_time:.4f}')
        print(f'\'{func.__name__}()\' ends at {ed} and takes {tot_time} mins.')
        func.tot_time = tot_time  # add new variable to func
        return result, tot_time

    return wrapper


def merge_dicts(dict_1, dict_2):
    """

    Parameters
    ----------
    dict_1
    dict_2

    Returns
    -------

    """
    for key in dict_2.keys():
        # if key not in dict_1.keys():
        dict_1[key] = dict_2[key]

    return dict_1


def func_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _args = inspect.getfullargspec(func)  # get all paramters and the corresponding default values
        # # might not work for kwargs
        # _keys_values = dict(list(zip(_args.args, args)))  # it will fail when misses pass some (k,v) pairs
        _keys_values = list(args)  # args and kwargs only get passed values to the current func
        [_keys_values.append(f"{k}={v}") for k, v in kwargs.items() if k not in _keys_values]
        print(f"\'{func.__name__}'s arguments ( {_keys_values} )")
        result = func(*args, **kwargs)
        return result

    return wrapper


def execute_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        st = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print(f'\'{func.__name__}()\' starts at {st}')
        result = func(*args, **kwargs)
        end = time.time()
        ed = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
        tot_time = (end - start) / 60
        tot_time = float(f'{tot_time:.4f}')
        print(f'\'{func.__name__}()\' ends at {ed}. It takes {tot_time} mins')
        return result

    return wrapper


def time_func(func, *args, **kwargs):
    start = datetime.now()

    result = func(*args, **kwargs)

    end = datetime.now()
    total_time = (end - start).total_seconds()
    # print(f'{func} running time: {total_time}, and result: {result}')

    return result, total_time


def mprint(msg, verbose=10, level=1):
    if verbose >= level:
        print(msg)


def run(in_file, out_file):
    """Save output to file by subprocess

        https://janakiev.com/blog/python-shell-commands/
        https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
        https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running

    Parameters
    ----------
    in_file
    out_file

    Returns
    -------

    """
    # cmd = f"python3.7 {in_file} > {out_file} 2>&1"
    cmd = f"python3.7 {in_file}"
    print(cmd)
    with open(out_file, 'wb') as f:
        # buffsize: 0 = unbuffered (default value); 1 = line buffered; N = approximate buffer size, ...
        # p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        #                      bufsize=0, universal_newlines=True)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             bufsize=1, universal_newlines=True)
        for line in p.stdout:
            # sys.stdout.write(line) # output to console
            f.write(line.encode())  # save to file
            print(line, end='')  # output to console

    return 0
