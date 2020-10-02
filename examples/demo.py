import numpy as np
from sklearn.base import BaseEstimator


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


if __name__ == '__main__':
    # op_demo()
    estimator_demo()
