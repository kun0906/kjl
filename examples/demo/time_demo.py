import gc
import os
import time
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

# from kjl.model.kjl import getGaussianGram

# from scalene.scalene_profiler import *

# print(os.environ)
os.environ['JOBLIB_TEMP_FOLDER'] = '~/kjl/examples/demo'
print(os.environ)


# @scalene_redirect_profile
# @profile
def foo():
    is_numpy_arr = 2
    if is_numpy_arr == 1:
        Xrow = np.ones((100, 30))
        A = getGaussianGram(Xrow, Xrow, sigma=1e-3)  # 100x100
        np.random.seed(42)
        d = 5
        m = 100
        random_matrix = np.random.multivariate_normal([0] * d, np.diag([1] * d), m)  # mxd
        # # random_matrix = Xrow
        U = np.matmul(A, random_matrix)  # preferred for matrix multiplication
        #
        X = np.ones((600, 30))
        K = getGaussianGram(X, Xrow, sigma=1e-3)  # 100x100
        s = np.matmul(K, U)  # K@U
    elif is_numpy_arr == 2:
        Xrow = np.ones((100, 5))
        X = np.ones((600, 5))
        s = np.matmul(X, Xrow.T)
    else:
        # don't use numpy array
        # start = time.time()
        s = 0
        # time.sleep(0.001)
        for i in range(100000):
            s += i
        del s
        # end = time.time()
        # print(end-start)

    return ''


def without_parallel(n_repeats=5):
    dt_times = []
    times = []
    time_nss = []
    monotonics = []
    monotonic_nss = []
    perf_counters = []
    perf_counter_nss = []
    process_times = []
    process_time_nss = []

    # start = time.time()
    # end = time.time()
    # diff = end - start
    # time.sleep(0.1)

    for i in range(n_repeats):
        # print(f'*********{i}')
        ###########
        start = time.time()
        foo()
        end = time.time()
        diff = end - start
        # print(f'time(): {diff}')
        times.append(diff)
        # del locals()['foo']
        # gc.collect()


        # start = time.time_ns()
        # foo()
        # end = time.time_ns()
        # diff = (end - start) / (10 ** 9)
        # # print(f'time_ns(): {diff}')
        # time_nss.append(diff)
        # # gc.collect()


    #     ###########
    #     start = datetime.now()
    #     foo()
    #     end = datetime.now()
    #     diff = (end - start).total_seconds()
    #     # print(f'time(): {diff}')
    #     dt_times.append(diff)
    #
    # for i in range(n_repeats):
    #     ###########
    #     start = time.monotonic()
    #     foo()
    #     end = time.monotonic()
    #     diff = end - start
    #     # print(f'monotonic(): {diff}')
    #     monotonics.append(diff)
    #
    # for i in range(n_repeats):
    #     start = time.monotonic_ns()
    #     foo()
    #     end = time.monotonic_ns()
    #     diff = (end - start) / (10 ** 9)
    #     # print(f'monotonic_ns(): {diff}')
    #     monotonic_nss.append(diff)
    #
    # for i in range(n_repeats):
    #     ###########
    #     start = time.perf_counter()
    #     foo()
    #     end = time.perf_counter()
    #     diff = end - start
    #     # print(f'perf_counter(): {end - start}')
    #     perf_counters.append(diff)

    # for i in range(n_repeats):
    #     start = time.perf_counter_ns()
    #     foo()
    #     end = time.perf_counter_ns()
    #     diff = (end - start) / (10 ** 9)
    #     # print(f'perf_counter_ns(): {diff}')
    #     perf_counter_nss.append(diff)

    # for i in range(n_repeats):
        ###########
        start = time.process_time()
        foo()
        end = time.process_time()
        diff = end - start
        # print(f'process_time(): {end - start}')
        process_times.append(diff)

    # for i in range(n_repeats):
    #     start = time.process_time_ns()
    #     foo()
    #     end = time.process_time_ns()
    #     diff = (end - start) / (10 ** 9)
    #     # print(f'process_time_ns(): {diff}')
    #     process_time_nss.append(diff)

    # print('\n\n')
    res = {'times': times, 'time_nss': time_nss, 'dt_times': dt_times,
           'monotonics': monotonics, 'monotonic_nss': monotonic_nss,
           'perf_counters': perf_counters, 'perf_counter_nss': perf_counter_nss,
           'process_times': process_times, 'process_time_nss': process_time_nss}
    return res


def with_parallel(n_repeats=5, flg=False, n_jobs=10):
    def _timing(name):
        # print(f'*********{i}')
        if name == 'times':
            ###########
            start = time.time()
            foo()
            end = time.time()
            diff = end - start

        elif name == 'time_nss':
            start = time.time_ns()
            foo()
            end = time.time_ns()
            diff = (end - start) / (10 ** 9)

        elif name == 'dt_times':
            ###########
            start = datetime.now()
            foo()
            end = datetime.now()
            diff = (end - start).total_seconds()

        elif name == 'monotonics':
            ###########
            start = time.monotonic()
            foo()
            end = time.monotonic()
            diff = end - start

        elif name == 'monotonic_nss':
            start = time.monotonic_ns()
            foo()
            end = time.monotonic_ns()
            diff = (end - start) / (10 ** 9)

        elif name == 'perf_counters':
            ###########
            start = time.perf_counter()
            foo()
            end = time.perf_counter()
            diff = end - start

        elif name == 'perf_counter_nss':
            start = time.perf_counter_ns()
            foo()
            end = time.perf_counter_ns()
            diff = (end - start) / (10 ** 9)

        elif name == 'process_times':
            ###########
            start = time.process_time()
            foo()
            end = time.process_time()
            diff = end - start

        elif name == 'process_time_nss':
            start = time.process_time_ns()
            foo()
            end = time.process_time_ns()
            diff = (end - start) / (10 ** 9)

        return diff

    def timing(name):
        if not flg:
            res = []
            for i in range(n_repeats):
                diff = _timing(name)
                # vs = [_timing(name) for _ in range(20)]
                # print(vs)
                # diff = np.mean(vs)
                res.append(diff)
        else:
            with Parallel(n_jobs=3, verbose=0, backend='loky', pre_dispatch=1, batch_size=1, max_nbytes=None,
                          mmap_mode=None) as parallel:
                res = parallel(delayed(_timing)(name) for i in range(n_repeats))

        return name, res

    names = ['times', 'time_nss', 'dt_times', 'monotonics', 'monotonic_nss', 'perf_counters', 'perf_counter_nss',
             'process_times',
             'process_time_nss']
    with Parallel(n_jobs=n_jobs, verbose=0, backend='loky', pre_dispatch=1, batch_size=1, temp_folder=None,
                  max_nbytes='100M', mmap_mode=None) as parallel:
        res = parallel(delayed(timing)(name_) for name_ in names)

    return res


def main():
    n_repeats = 10
    # foo()

    print('\nwithout_parallel')
    res = without_parallel(n_repeats)
    for name_, vs_ in res.items():
        print(f'{name_:20}: {np.mean(vs_):.5f}+/-{np.std(vs_):.5f}', [f'{v:.5f}' for v in vs_])

    print('\nwith_parallel')
    res = with_parallel(n_repeats, flg=False, n_jobs=10)
    for (name_, vs_) in res:
        print(f'{name_:20}: {np.mean(vs_):.5f}+/-{np.std(vs_):.5f}', [f'{v:.5f}' for v in vs_])

    # print('\nwith_parallel2')
    # res = with_parallel(n_repeats, flg=False, n_jobs=1)
    # for (name_, vs_) in res:
    #     print(f'{name_:20}: {np.mean(vs_):.5f}+/-{np.std(vs_):.5f}', [f'{v:.5f}' for v in vs_])


if __name__ == '__main__':
    main()
