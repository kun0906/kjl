import numpy as np


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


if __name__ == '__main__':
    op_demo()
