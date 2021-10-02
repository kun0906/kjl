#
# import numpy as np
# import scipy
# from sklearn.utils import resample
#
# d = 5
# m = 10
# n = m
# random_state = 42
# np.random.seed(random_state)
# for i in range(2):
#     np.random.seed(42)
#     indices = resample(range(5000), n_samples=max(n, m), random_state=random_state,
#                        replace=False)
#     v = np.random.multivariate_normal([0] * d, np.diag([1] * d), size=m)    # mxd
#     # v = scipy.stats.multivariate_normal([0] * d, np.diag([1] * d), m, seed=random_state).pdf()
#     print(v)



from joblib import Parallel, delayed

import multiprocessing


def func():
    return multiprocessing.current_process().pid


def parallel_func():
    return Parallel(n_jobs=1)(delayed(func)() for _ in range(2))

if __name__ == '__main__':
    print(Parallel(n_jobs=2)(delayed(parallel_func)() for _ in range(3)))
    # print(Parallel(n_jobs=1)(delayed(parallel_func)() for _ in range(3)))
    # print(Parallel(n_jobs=1)(delayed(parallel_func)() for _ in range(3)))