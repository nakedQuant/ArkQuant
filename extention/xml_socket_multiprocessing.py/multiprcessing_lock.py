# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()


# from multiprocessing import Process, Value,Array
# # 使用 Value 或 Array 将数据存储在共享内存映射中
# def f(n, a):
#     n.value = 3.1415927
#     for i in range(len(a)):
#         a[i] = -a[i]
#
# if __name__ == '__main__':
#     # 创建 num 和 arr 时使用的 'd' 和 'i' 参数是 array 模块使用的类型的 typecode ： 'd' 表示双精度浮点数，
#     # 'i' 表示有符号整数。这些共享对象将是进程和线程安全的。
#     num = Value('d', 0.0)
#     arr = Array('i', range(10))
#
#     p = Process(target=f, args=(num, arr))
#     p.start()
#     p.join()
#
#     print(num.value)
#     print(arr[:])