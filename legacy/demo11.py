

def get_triplet(arr, target):
	arr.sort()
	print(arr)

	n = len(arr)
	for i in range(n):
		l = i + 1
		r = n - 1
		while l < n:
			s = arr[i] + arr[l] + arr[r]
			if s == target:
				return [arr[i], arr[l], arr[r]]
			elif s > target:
				r -= 1
			else:
				l += 1

	return []

# arr = [12, 3, 4, 1, 6, 9]
# target = 24
arr = [1 ,2, 3, 4, 5]
target = 9
res = get_triplet(arr, target)
print(res)
