"""

/home/ky2440/.local/bin/scalene mem_demo.py >a.txt

"""
import gc
from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    gc.collect()
    return a

if __name__ == '__main__':
    my_func()
