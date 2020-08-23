# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import multiprocessing
from multiprocessing.connection import Listener, Client
from multiprocessing.managers import BaseManager
from multiprocessing import Queue
from array import array
from multiprocessing import Process, Pipe, current_process, Lock
from multiprocessing.connection import wait

address = ('localhost', 6000)     # family is deduced to be 'AF_INET'

with Listener(address, authkey=b'secret password') as listener:
    with listener.accept() as conn:
        print('connection accepted from', listener.last_accepted)
        conn.send([2.25, None, 'junk', float])
        conn.send_bytes(b'hello')
        conn.send_bytes(array('i', [42, 1729]))


# client

with Client(address, authkey=b'secret password') as conn:
    print(conn.recv())                  # => [2.25, None, 'junk', float]
    print(conn.recv_bytes())            # => 'hello'
    arr = array('i', [2, 4, 7, 8, 0, 0, 0, 0])
    print(conn.recv_bytes_into(arr))    # => 8
    print(arr)


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


# lock
def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

# lock = Lock()
#
# for num in range(10):
#     Process(target=f, args=(lock, num)).start()

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


queue = Queue()


class QManager(BaseManager):
    """
        抽象方法register
    """
    pass


def func(x, y):
    z = x + y
    return z


QManager.register('add', func)
QManager.register('fifo', callable=lambda:  queue)

m = QManager(address=('192.168.0.103', 10000), authkey=b'test')
test = m.get_server()
test.serve_forever()


# client
class CManager(BaseManager):
    pass


CManager.register('add')


# if __name__ == '__main__':
#     m = CManager(address=('192.168.0.103', 50000), authkey=b'test')
#     m.connect()
#     res = m.add(4, 5)
#     q = m.fifo()
#     x = 5
#     while x < 10:
#         res = m.add(x, 3)
#         print(res)
#         q.put(x)
#         x = x + 1
"""
multiprocessing.Pipe([duplex])
返回一对 Connection`对象  ``(conn1, conn2)` ， 分别表示管道的两端。

如果 duplex 被置为 True (默认值)，那么该管道是双向的。如果 duplex 被置为 False ，那么该管道是单向的，即 conn1 只能用于接收消息，
而 conn2 仅能用于发送消息

"""


def foo(w):
    for i in range(10):
        w.send((i, current_process().name))
    w.close()


readers = []

for i in range(4):
    r, w = Pipe(duplex=False)
    readers.append(r)
    p = Process(target=foo, args=(w,))
    p.start()
    # We close the writable end of the pipe now to be sure that
    # p is the only process which owns a handle for it.  This
    # ensures that when p closes its handle for the writable end,
    # wait() will promptly report the readable end as being ready.
    w.close()

while readers:
    for r in wait(readers):
        try:
            msg = r.recv()
        except EOFError:
            readers.remove(r)
        else:
            print(msg)

# from multiprocessing import Process, Pipe
# """
# 返回的两个连接对象 Pipe() 表示管道的两端。每个连接对象都有 send() 和 recv() 方法（相互之间的）。
# 如果两个进程（或线程）同时尝试读取或写入管道的 同一 端，则管道中的数据可能会损坏。当然，同时使用管道的不同端的进程不存在损坏的风险。
# """
#
# def f(conn):
#     conn.send([42, None, 'hello'])
#     conn.close()
#
# if __name__ == '__main__':
#     parent_conn, child_conn = Pipe(duulex = False)
#     p = Process(target=f, args=(child_conn,))
#     p.start()
#     print(parent_conn.recv())   # prints "[42, None, 'hello']"
#     p.join()

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
