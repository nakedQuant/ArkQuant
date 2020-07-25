#!/usr/bin/env python3
# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

#进程通讯
import multiprocessing

def consumer(input_q):
    while True:
        item = input_q.get()
        print(item)
        # task_done发出信号，表示get返回的项已经被处理 ,防止返回queue.empty错误
        input_q.task_done()


def producer(sequence, out_put_q):
    for item in sequence:
        # 将item放入队列，如果队列已经满，阻塞至有空间用为止
        out_put_q.put(item)


if __name__ == '__main__':
    # 创建共享进程列队，queue对象
    q = multiprocessing.JoinableQueue()
    cons_p = multiprocessing.Process(target=consumer, args=(q,))
    cons_p.daemon = True
    cons_p.start()

    sequence = [1, 2, 3, 4]
    producer(sequence, q)
    # 生产进行阻塞，直到队列所有项均被处理
    q.join()


# lock
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