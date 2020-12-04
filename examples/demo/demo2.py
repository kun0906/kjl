
def f1(data):
    vs = [ v.split(',') for v in data]
    vs = sorted(vs, key=lambda v:float(v[1]), reverse=True)
    return vs[0][0]


def removeKElems(a, k):
    n = len(a)
    result = lambda arr, idx: arr[0:(idx)] + arr[(idx + k):n]
    return [ result(a, i) for i in range(n - k + 1) ]


# code


def _sum(S, A, i):
    s = 0
    for idx, v in enumerate(A):
        idx = i+ idx
        s += A[idx]
        if s == S:
            return idx, True
        if s > S:
            return 0, False

    return 0, False


def subarray_sum(N, S, A):
    is_find = False
    acc_sum = 0
    st =  0
    for i, v in enumerate(A):
        # print(i, end_idx, is_find)
        if acc_sum == S:
            return st + 1, i + 1
        elif acc_sum > S:
            while acc_sum > S:
                acc_sum = acc_sum - A[st]
                st += 1
            if acc_sum == S:
                return st + 1, i
        else:
            acc_sum += v

    return - 1



if __name__ == '__main__':
    # data = ["iqttt,0077",
    #         "obvhd,0093",
    #         "flohd,0075"]
    #
    # # print(f1(data))
    # a = [2, 4, 6, 8, 10]
    # k =3
    # print(removeKElems(a, k))

    N = 5
    S = 12
    A = [1, 2, 3, 7, 5]
    print(subarray_sum(N, S, A))
