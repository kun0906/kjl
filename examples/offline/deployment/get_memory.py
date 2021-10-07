
#
#
# #!/usr/bin/env python
# import psutil
# # gives a single float value
# print(psutil.cpu_percent())
# # gives an object with many fields
# print(psutil.virtual_memory())
# # you can convert that object to a dictionary
# dict(psutil.virtual_memory()._asdict())
# # you can have the percentage of used RAM
# print(psutil.virtual_memory().percent)
# # 79.2
# # you can calculate percentage of available memory
# print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
# # 20.8
#


"mprof run examples / offline / report / paper_speedup.py"
"mprof list"
"mprof plot mprofile_20211007163839.dat"


from memory_profiler import profile

@profile
def func():
	res = 0
	for i in range(100000):
		res +=1
	print(res)

func()

