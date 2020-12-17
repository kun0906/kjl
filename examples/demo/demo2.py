
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

def kill(arr, k ):
    if len(arr) == 1:
        return arr[0]

    print(arr, k)
    arr.pop(k-1)

    k +=k
    n = len(arr)
    k = k % n
    kill(arr, k)


import copy




class Solution:
    def subsets(self, nums):
        def sub(nums, sub_res):
            # if len(nums) == 1:
            #     return
            # sub_res.append(nums)
            res.append(sub_res)
            for i in range(len(nums) ):
                # new_nums = copy.deepcopy(nums)
                # new_nums.pop(i)
                # if not new_nums in sub_res:
                #     sub_res.append(new_nums)
                # print(new_nums, i, sub_res)
                # sub(new_nums, sub_res)
                # print(nums, sub_res)
                sub(nums[i + 1:], sub_res + [nums[i]])

        res = [nums,[]]
        sub(nums, [])
        print(res)
        return res


# Python3 program to count total number
# of non-leaf nodes in a binary tree

# class that allocates a new node with the
# given data and None left and right pointers.
class newNode:
    def __init__(self, data):
        self.data = data
        self.left = self.right = None


# Computes the number of non-leaf
# nodes in a tree.
def countNonleaf(root):
    # Base cases.
    if (root == None or (root.left == None and
                         root.right == None)):
        return 0

    # If root is Not None and its one of
    # its child is also not None
    return (1 + countNonleaf(root.left) +
            countNonleaf(root.right))


# Driver Code
if __name__ == '__main__':
    root = newNode(1)
    root.left = newNode(2)
    root.right = newNode(3)
    root.left.left = newNode(4)
    root.left.right = newNode(5)
    print(countNonleaf(root))

# This code is contributed by PranchalK

if __name__ == '__main__':
    # data = ["iqttt,0077",
    #         "obvhd,0093",
    #         "flohd,0075"]
    #
    # # print(f1(data))
    # a = [2, 4, 6, 8, 10]
    # k =3
    # print(removeKElems(a, k))

    # N = 5
    # S = 12
    # A = [1, 2, 3, 7, 5]
    # print(subarray_sum(N, S, A))
    # arr = [i for i in range(8)]
    # kill(arr, 2)
    Solution().subsets([1,2,3])
