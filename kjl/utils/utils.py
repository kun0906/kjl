"""Useful tools includes 'data_info', 'dump_data', etc.

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import inspect
import pickle
import time
from functools import wraps
from datetime import datetime
import pandas as pd


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


def func_running_time(func, *args, **kwargs):
    start = datetime.now()
    result = func(*args, **kwargs)

    end = datetime.now()
    total_time = (end - start).total_seconds()

    # print(f'{func} running time: {total_time}, and result: {result}')

    return result, total_time
