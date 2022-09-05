# class MovingTotal:
#     def __init__(self):
#         self.remain = []
#         self.res = {}
#
#     def append(self, numbers):
#         """
#         :param numbers: (list) The list of numbers.
#         """
#
#         self.remain +=numbers
#
#         for i in range(len(self.remain)):
#             if i+2 >= len(self.remain):
#                 self.remain = self.remain[i:]
#                 break
#             self.res[self.remain[i] + self.remain[i + 1] + self.remain[i + 2]] = 1
#
#     def contains(self, total):
#         """
#         :param total: (int) The total to check for.
#         :returns: (bool) If MovingTotal contains the total.
#         """
#         if total in self.res.keys():
#             return True
#         return False
#
#
# if __name__ == "__main__":
#     movingtotal = MovingTotal()
#
#     movingtotal.append([1, 2, 3, 4])
#     print(movingtotal.contains(6))
#     print(movingtotal.contains(9))
#     print(movingtotal.contains(12))
#     print(movingtotal.contains(7))
#
#     movingtotal.append([5])
#     print(movingtotal.contains(6))
#     print(movingtotal.contains(9))
#     print(movingtotal.contains(12))
#     print(movingtotal.contains(7))



#
# a=[[10,50,10,20,10,0],
# [30,60,30,60,10,20],
# [20,50,30,80,20,15],
# [60,10,60,30,30,50],
# [50,40,120,100,70,80],
# [90,80,80,10,50,50],
# [20,90,100,110,50,60],
# [20,70,130,30,80,120],
# [60,50,90,60,60,80]]
#
# import numpy as np
# b = np.asarray(a).transpose()
# b= np.corrcoef(b)
# print(b, b.shape)
#

# a = 0.3**3 + 0.7*(0.3**2) + (0.7**2) * 0.3 + 0.7**3
# print(a)
#
# def first_unique_product(products):
#     res = {}
#     for v in products:
#         if v in res.keys():
#             res[v] +=1
#         else:
#             res[v] = 1
#
#     for k, v in res.items():
#         if v == 1:
#             return k
#
#     return False
#
# if __name__ == "__main__":
#     print(first_unique_product(["Apple", "Computer", "Apple", "Bag"])) #should print "Computer"

#
# def find_max_sum(numbers):
#     if numbers == []:
#         return 0
#     if len(numbers) == 1:
#         return numbers[0]
#
#     if len(numbers) == 2:
#         return numbers[0] + numbers[1]
#
#     if numbers[0] > numbers[1]:
#         first = numbers[0]
#         second = numbers[1]
#
#     else:
#         first = numbers[1]
#         second = numbers[0]
#
#     for i, v in enumerate(numbers[2:]):
#         if v > first:
#             second = first
#             first = v
#         elif v > second and v != first:
#             second = v
#     return first + second
#
#
# if __name__ == "__main__":
#     print(find_max_sum([5, 9, 7, 11]))
#
# def find_target(nums, l, h, target):
#
#     while l < h:
#         m = l + (h - l) // 2
#         if nums[m] == target:
#             return m
#         elif nums[m] < target:
#             l = m + 1
#         else:
#             h = m - 1
#
#     return -1
#
# class Solution:
#     def searchRange(self, nums, target):
#
#         l = 0
#         h = len(nums) - 1
#         m = find_target(nums, l, h, target)
#         if m ==-1:
#             return [-1, -1]
#         else:
#             s = m
#             while s > 0:
#                 if nums[s] == target:
#                     s -= 1
#                 else:
#                     break
#             s = s + 1 if nums[s] != target else s
#             e = m
#             while e < h:
#                 if nums[e] == target:
#                     e += 1
#                 else:
#                     break
#             e = e -1 if nums[e] !=target else e
#         return [s, e]
#
# nums = [5,7,7,8,8,10]
# target = 8
# res = Solution().searchRange(nums, target)
# print(res)

#
# class Solution:
#     def myAtoi(self, s: str) -> int:
#         min_int = -2 ** 31
#         max_int = 2 ** 31 - 1
#
#         res = s.strip().split(' ')[0]
#         print(res)
#         value = None
#
#         is_int = False
#         for v in res:
#             print(v)
#             if v == '-':
#                 value = -1
#             elif v == '+':
#                 value = 1
#             elif v.isdigit():
#                 is_int = True
#                 break
#             else:
#                 is_int = False
#                 break
#         if is_int:
#             value = int(res)
#         else:
#             pass
#         if value is None:
#             return 0
#         if value > max_int:
#             value = max_int
#         if value < min_int:
#             value = min_int
#
#         return value
#
# print(Solution().myAtoi("   -42"))
# import heapq
#
#
# def get_max(nums=[1,2,3,2, 4, 1]):
#
#     heapq.heapify(nums) # in_place
#     print(nums)
#     v = ''
#     for i in range(len(nums)):
#         v = heapq.heappop(nums)
#         print(v)
#         if i == 2:
#             break
#
#     return v
#
# import numpy as np
# np.maximum()
# a = []
# a.pop()
# print(get_max())

#
# def move_to_left(a):
#
#     l = len(a)
#     nonzero_idx = l-1
#     zero_idx = l-1
#     for i in range(l-1, 0, -1):
#         print(i, a[i])
#         if a[i] != 0:
#             a[zero_idx] = a[i]
#             zero_idx -= 1
#     print(zero_idx)
#
#     for i in range(zero_idx):
#         a[i] = 0
#
# a = [1, 10, 20, 0, 59, 63, 0, 88, 0]
# move_to_left(a)
# print(a)
#
# arr = [[1,5], [3,7], [4,6], [6,8]]
# def merge_interval(arr):
#     cur = arr[0]
#     for v in arr[1:]:
#         if v[0] < cur[1]:
#             cur = [cur[0], max(v[1], cur[1])]
#
#         # if v[0] < cur[1]
#     return cur
#
# print(merge_interval(arr))
#
#
# def lever_traverse_tree(T):
#     queue = []
#
#     while T:
#         if not T.left is None:
#             queue.append(T.left)
#         if not T.right is None:
#             queue.append(T.right)
#
#         T = queue.pop(0)

#
# def find(s, d):
#     l = len(s)
#     res = []
#     for i in range(l):
#         sub = s[:i]
#         if sub in d.keys():
#             res.append(sub)
#         if s[i:] in d.keys():
#             res.append(s[i:])
#
# def str_segematation(s:str, d:dict)->list:
#
#     l= len(str)
#     res = []
#     for i in range(l):
#         sub = s[:i]
#         find(s[:i], d)
#         find(s[i:], d)
#
#     return res
#
#
import numpy as np
from sklearn.decomposition import PCA

#
# def main_pac():
#     X = np.array([[-1, -1, 1], [-2, -1, 2], [-3, -2, 3], [1, 1, 4], [2, 1, 5], [3, 2, 6]])
#     print(X.shape)
#     print(X[:, :, np.newaxis].shape)
#     print(X[:, :, np.newaxis, np.newaxis].shape)
#     pca = PCA(n_components=2)
#     pca.fit(X)
#     X_reduced = pca.transform(X)
#
# main_pac()
#
# def find_missing(arr):
#
#     s = arr[0]
#     for v in arr[1:]:
#         s ^=v
#
#     for i in range(1, len(arr)+1):
#         s ^=1
#
#     n = len(arr)
#     d = sum(arr) - n*(n+1)//2
#     if d > 0:
#         return s, s-1
#
#     return s+1, s
#

#
# def heap_demo(arr):
#
#     import heapq as hp
#     # smallest element is always the root
#     hp.heapify(arr)
#     res = []
#     while arr:
#         v = hp.heappop(arr)
#         res.append(v)
#
#     print(res)
#     return res
#
# heap_demo(arr=[12, 5, 787, 1, 23])

#
# # Python3 program to remove invalid parenthesis
#
# # Method checks if character is parenthesis(open
# # or closed)
# def isParenthesis(c):
#     return ((c == '(') or (c == ')'))
#
#
# # method returns true if contains valid
# # parenthesis
# def isValidString(str):
#     cnt = 0
#     for i in range(len(str)):
#         if (str[i] == '('):
#             cnt += 1
#         elif (str[i] == ')'):
#             cnt -= 1
#         # print(cnt)
#         if (cnt < 0):
#             return False
#
#     return (cnt == 0)
#
#
# # method to remove invalid parenthesis
# def removeInvalidParenthesis(str):
#     if (len(str) == 0):
#         return
#
#     # visit set to ignore already visited
#     visit = set()
#
#     # queue to maintain BFS
#     q = []
#     temp = 0
#     level = 0
#
#     # pushing given as starting node into queu
#     q.append(str)
#     visit.add(str)
#     while (len(q)):
#         str = q[0]
#         q.pop(0)
#         # print(str)
#         if (isValidString(str)):
#             print(str)
#             # If answer is found, make level true
#             # so that valid of only that level
#             # are processed.
#             level = True
#         if (level):
#             continue
#         for i in range(len(str)):
#             if (not isParenthesis(str[i])):
#                 continue
#
#             # Removing parenthesis from str and
#             # pushing into queue,if not visited already
#             temp = str[0:i] + str[i + 1:]
#             if temp not in visit:
#                 q.append(temp)
#                 visit.add(temp)
#
#
# str = "()())()"
# removeInvalidParenthesis(str)

import torch
from torch import nn

"""
Input: (N, C, H, W)(N,C,H,W)
Output: (N, C, H, W)(N,C,H,W) (same shape as input)
"""
# import torch
# torch.manual_seed(0)
#
# input = torch.randn(3, 2, 2, 2)
# print(input)
# c1 = input[:, 0, :, :].flatten().data.numpy()
# print( np.mean(c1) , np.std(c1))
# m = nn.BatchNorm2d(2, eps=0)
# output = m(input)
#
# print(output)




import torch
torch.manual_seed(0)

input = torch.randn(3, 2, 2, 2)
print(input)
c1 = input[0, :, :, :].flatten().data.numpy()
print( np.mean(c1) , np.std(c1))
m = nn.LayerNorm(input.size()[1:], eps=0)
output = m(input)

print(output)