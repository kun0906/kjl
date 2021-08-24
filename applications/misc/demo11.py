

import heapq

def f():

	lst = [10 , 5, 2, 3, 6, 7, 9]
	heapq.heapify(lst)  # trasnform a list to heap (in-place)
	print(lst)

	# print(lst[-1])
	print(lst.pop())
	print(lst.pop(0))

f()