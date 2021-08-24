import copy
import datetime

import matplotlib
import numpy as np
from odet.pparser.parser import PCAP, _get_flow_duration, _get_frame_time
from sklearn.base import BaseEstimator

from kjl.models.kjl import getGaussianGram
from kjl.utils.data import dump_data, load_data, data_info


def op_demo():
    a = np.array([1, 2, 3, 4])
    # b = np.array([[1, 2, 3],
    #    [3, 4, 5],
    #    [3, 6, 9]])
    print(a.T, a.T.shape, a, a.shape)
    b = np.array([[1, 2, 3, 5],
                  [3, 4, 5, 5],
                  [3, 6, 9, 5]])
    a1 = a[:, np.newaxis]
    print(f'a1:', a1, )
    print(f'b:', b)
    c = a * b
    print(f'c:', c)
    d = a * b.T
    print(f'd:', d)

import logging
from logging import getLogger

lg = getLogger(name='mylog')
lg.setLevel(level=logging.INFO)

lg.info('afadfasf')
lg.debug('afadfasf')
lg.warning('afadfasf')
lg.error('afadfasf')

class MYCLASSIFIER(BaseEstimator):
    print(type(BaseEstimator))
    a = 10

    def __init__(self, a=1, b=2, c = ''):
        print(type(self))
        self.a = a
        self.b = b
        self.c1 =c

    def fit(self):
        pass


def estimator_demo():
    cf = MYCLASSIFIER()
    cf.fit()


def pcap2time():

    pcap_file = '../../IoT_feature_sets_comparison_20190822/applications/data/data_reprst/pcaps/DEMO_IDS' \
                '/DS-srcIP_192.168.10.5/AGMT-WorkingHours/srcIP_192.168.10.5_AGMT.pcap'
    pp=PCAP(pcap_file)
    pp.pcap2flows()
    out_file = pp.pcap_file + '.dat'
    data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in pp.flows]).reshape(-1, 1), name=f'durations before')
    dump_data(pp.flows, out_file)

    flows = load_data(out_file)
    for fid, pkts in flows:
        print([str(datetime.datetime.fromtimestamp(_get_frame_time(pkt))) for pkt in pkts])
    data_info(np.asarray([_get_flow_duration(pkts) for fid, pkts in flows]).reshape(-1, 1), name=f'durations')


def sciview():
    print(matplotlib.get_backend())
    # plt.imshow(img.reshape((28, 28)))
    # plt.show()


def arr():
    X = [[i] * 3 for i in range(10)]
    X = np.asarray(X)

    print('X', X)
    idx = [5, 3, 7, 4]

    X1 = [[i] * 3 for i in range(len(idx))]
    X1 = np.asarray(X1)
    print('X1', X1)
    X[idx] = X1

    print('X', X)

def kjl_demo():
    Xrow = [[i] * 3 for i in range(5)]
    Xrow = np.asarray(Xrow)
    print('Xrow', Xrow)

    K = getGaussianGram(Xrow, Xrow, sigma=1, goFast=1)
    print('K:', K)

    K1 = getGaussianGram(Xrow, Xrow, sigma=1, goFast=0)
    print('K1:', K1)
    print('diff:', K1-K)



def exp1():
    s = 1/100
    for i in range(30):
        s += s**2


    print(s)

#
# import pandas as pd
#
# def solution(input_file=''):
#     df = pd.read_csv(input_file)
#     n_customers = df.shape[0]
#     customers_per_city = df.groupby(by=['CITY'])
#     custmoers_per_country = df.groupby(by=['COUNTRY'])
#     idx = df.groupby(by=['COUNTRY']).agg({'CONTRCNT':'sum'}).idxmax()
#     # name_max_contracts = df[id]['COUNTRY']
#     # n_contracts = pd.sum(df[id])
#     # n_unique_cities = pd.groupby(df['CITY'])
#
#     print('Total customers:')
#     print(n_customers)
#     print('Customers by city:')
#     for tup in customers_per_city:
#         print(tup[0], ':', tup[1].shape[0])
#     print('Customers by country:')
#     for tup in custmoers_per_country:
#         print(tup[0], ':', tup[1].shape[0])
#     # print('Country with the largest number of customers\' contracts: ')
#     # print(f'{name_max_contracts} ({n_contracts}) contracts')
#     # print('Unique cities with at least one customer:')
#     # print(n_unique_cities)


import copy


def find_next(T, p, lth=0, sub=[]):
    sub.append(p)
    s = [v for v in sub if v % 2 != 0]
    if len(s) > 1:
        return lth, sub[:-1]
    lth += 1
    next_cities = []
    for i, v in enumerate(T):
        if v == p or i == p:
            if v not in sub:
                next_cities.append(v)
            if i not in sub:
                next_cities.append(i)
            else:
                continue
    # print(next_cities, p, lth, sub, s)
    max_lth = lth
    max_sub = sub
    if len(next_cities) == 0 or next_cities == []:
        return max_lth, max_sub

    for p_ in next_cities:
        sub_new = copy.deepcopy(sub)
        lth_, sub_new = find_next(T, p_, lth, sub_new)
        if lth_ > max_lth:
            max_sub = sub_new
            max_lth = lth_
    return max_lth, max_sub

def solution(T):
    # write your code in Python 3.6

    next_cities = [i for i, v in enumerate(T) if v == 0 and i > 0]

    max_lth = 0
    for p in next_cities:
        lth, sub = find_next(T, p, lth=1, sub=[0])
        if lth > max_lth:
            max_sub = sub
            max_lth = lth

    print(max_lth, max_sub)
    return max_lth

#
# T = [0, 9, 0, 2, 6, 8, 0, 8, 3, 0]
# # T=[0,0,0,1,6,1,0,0]
# solution(T)

#
# def split_sum(v):
#     return sum([int(_v) for _v in str(v)])
#
# def solution2(N):
#     # write your code in Python 3.6
#     # i_0 = 0
#     # i_1 = 1
#     #
#     # i = 0
#     # s = 0
#     # N = N-1
#     # while i < N:
#     #     s = i_0 + i_1
#     #     i_0 = i_1
#     #     i_1 = s
#     #     i += 1
#
#     arr = [0, 1, 1, 2, 3, 5, 8, 13, 12, 7, 10, 8, 9]
#     s = split_sum(arr[N-1-1]) + split_sum(arr[N - 1])
#
#     return s
#
# N = 10
# print(solution(N))
#
# N = 6
# solution(N)
#
# N = 10
# solution(N)


def de2():
    import copy


    def find_next(T, p, lth=0, sub=[]):
        sub.append(p)
        s = [v for v in sub if v % 2 != 0]
        if len(s) > 1:
            return lth, sub[:-1]
        next_cities = []
        for i, v in enumerate(T):
            if v == p or i == p:
                if v not in sub:
                    next_cities.append(v)
                if i not in sub:
                    next_cities.append(i)
                else:
                    continue
        # next_cities = [i for i, v in enumerate(T) if v == p or i ==p]

        # print(next_cities, p, lth, sub, s)
        max_lth = lth
        max_sub = sub
        if len(next_cities) == 0 or next_cities == [] or len(s) > 1:
            return max_lth, max_sub

        for p_ in next_cities:
            sub_new = copy.deepcopy(sub)
            lth_, sub_new = find_next(T, p_, lth + 1, sub_new)
            if lth_ > max_lth:
                max_sub = sub_new
                max_lth = lth_
        return max_lth, max_sub


    #
    # def find_next1(T, p, lth=0):
    #     while

    def solution(T):
        # write your code in Python 3.6

        next_cities = [i for i, v in enumerate(T) if v == 0 and i > 0]
        # print(next_cities)

        max_lth = 0
        for p in next_cities:

            # T_sub = [i for i, v in enumerate(T) if v == 0]
            lth, sub = find_next(T, p, lth=1, sub=[0])
            # print(f'p: {p}, lth: {lth}, sub: {sub}')
            if len(sub) > max_lth:
                max_sub = sub
                max_lth = len(sub)

        # print(max_lth, max_sub)
        # print(len(max_sub))
        return len(max_sub)


    T = [0, 9, 0, 2, 6, 8, 0, 8, 3, 0]
    # T=[0,0,0,1,6,1,0,0]
    solution(T)


def positive(n=0):
    arr = []
    nums = [i for i in range(n)] + [n]
    print(nums)
    for i in range(n+1):
        if i == 0: continue
        if n/i in nums:
            arr.append(i)
    print(arr)
    print(len(set(arr)))

def intersect(a, b):
    a = [1,2,3]
    b=[2,3]
    print(set(a)-set(b))
    print(a-b)
    print(set(a).intersection(set(b)))

if __name__ == '__main__':
    # op_demo()
    # estimator_demo()
    # pcap2time()
    # sciview()
    # arr()
    # kjl_demo()
    # exp1()

    # solution('data/data.csv')

    # T = [0, 9, 0, 2, 6, 8, 0, 8, 3, 0]
    # # T = [0, 0, 0, 1, 6, 1, 0, 0]
    # solution(T)


    positive(n=2160)

