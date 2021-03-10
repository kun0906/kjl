

import random
import timeit
import numpy as np
from datetime import datetime
from joblib import delayed, Parallel
import copy

timeit.timeit()
def _my_random(X=None, i=0):

    n_repeats = 5
    diffs = []
    for c in range(n_repeats):
        start = datetime.now()
        j = 0
        # for timing purpose
        while j < 1000000:
            j +=1
        X = [v + i for v in X]
        end = datetime.now()
        diff_time = (end - start).total_seconds()
        diffs.append(diff_time)
    return f'{np.mean(diffs):.2f}+/-{np.std(diffs):.2f}'


def find_best_param(X = None, i = 0):
    # find the best params in parallel
    parallel = Parallel(n_jobs=5, verbose=30,backend='threading', pre_dispatch=1)
    with parallel:
        outs = parallel(delayed(_my_random)(X=copy.deepcopy(X), i=i+j) for j in range(10))

    print('with best_params', _my_random(X=copy.deepcopy(X), i=0))
    print(f'X={X}, {outs}')
    return outs



if __name__ == '__main__':
    # print('stdlib')
    # print(Parallel(n_jobs=2)(delayed(find_best_param)() for _ in range(6)))
    # print('stdlib setting seed to None')
    # print(Parallel(n_jobs=2)(delayed(find_best_param)(set_seed_to_none=True)
    #                          for _ in range(6)))
    # print('numpy')
    # print(Parallel(n_jobs=2)(delayed(find_best_param)(use_numpy=True)
    #                          for _ in range(6)))
    # print('numpy setting seed to None')
    # print(Parallel(n_jobs=2)(delayed(find_best_param)(
    #     set_seed_to_none=True, use_numpy=True)
    #                          for _ in range(6)))

    # X_ = np.zeros((3,2))
    # X_ = [ 1, 2, 3]
    data1 = [1, 2, 3]
    data2=[2, 3, 4]
    data3 = [4, 5, 6]
    #
    # print('dataset with parallel, job=3')
    # parallel = Parallel(n_jobs=3, verbose=30)
    # with parallel:
    #     outs = parallel(delayed(find_best_param)(X=copy.deepcopy(X_), i=1 ) for X_ in [data1, data2, data3])

    # print('dataset with parallel: job=1')
    # parallel = Parallel(n_jobs=2, verbose=100, backend='loky')
    # with parallel:
    #     outs = parallel(delayed(find_best_param)(X=copy.deepcopy(X_), i=1) for X_ in [data1, data2, data3])

    # print('dataset without parallel')
    # for X_ in [data1, data2, data3]:
    #     find_best_param(X= copy.deepcopy(X_), i =1)

    # print(outs)
