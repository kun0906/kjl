import time
import gc
import numpy as np

n_repeats = 100





def demo1():
    def foo():
        # don't use numpy array
        start = time.time()
        # s = 0
        # # time.sleep(0.001)
        # for i in range(100000):
        #     s += i

        Xrow = np.ones((100, 5))
        X = np.ones((600, 5))
        s = np.matmul(X, Xrow.T)
        del Xrow
        del X
        gc.collect()

        end = time.time()
        # print(end-start)
        return end - start

    vs = []
    for _ in range(n_repeats):
        vs.append(foo())
    res = [('time()', vs)]
    for (name_, vs_) in res:
        print(f'{name_:20}: {np.mean(vs_):.5f}+/-{np.std(vs_):.5f}', [f'{v:.5f}' for v in vs_])



def deom2():
    def foo():
        # don't use numpy array
        # start = time.time()
        # s = 0
        # # time.sleep(0.001)
        # for i in range(100000):
        #     s += i

        Xrow = np.ones((100, 5))
        X = np.ones((600, 5))
        s = np.matmul(X, Xrow.T)
        del Xrow
        del X
        gc.collect()

        # end = time.time()
        # print(end-start)
        return ''

    vs = []
    for _ in range(n_repeats):
        start = time.time()
        foo()
        end = time.time()
        vs.append(end - start)

        # start = time.time_ns()
        # foo()
        # end = time.time_ns()
        # # vs.append(end - start)
        #
        # start = time.time_ns()
        # foo()
        # end = time.time_ns()
        # # vs.append(end - start)
        #
        # start = time.time_ns()
        # foo()
        # end = time.time_ns()
        # # vs.append(end - start)

    res = [('time()', vs)]
    for (name_, vs_) in res:
        print(f'{name_:20}: {np.mean(vs_):.5f}+/-{np.std(vs_):.5f}', [f'{v:.5f}' for v in vs_])


def main():
    # demo1()
    deom2()


if __name__ == '__main__':
    main()
