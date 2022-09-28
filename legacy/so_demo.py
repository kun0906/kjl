


from ctypes import cdll
# # either
# libc = cdll.LoadLibrary("libc.so.6")
# # or
# libc = CDLL("libc.so.6")

libc = cdll.LoadLibrary('/Users/kunyang/opt/miniconda3/envs/py379/lib/python3.7/lib-dynload/_json.cpython-37m-darwin.so')
print(libc)


