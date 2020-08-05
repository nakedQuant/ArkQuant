# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

# import multiprocessing, time, signal
# p = multiprocessing.Process(target=time.sleep, args=(1000,))
# print(p, p.is_alive())
# #<Process(Process-1, initial)> False
# p.start()
# print(p, p.is_alive())
# #<Process(Process-1, started)> True
# p.terminate()
# time.sleep(0.1)
# print(p, p.is_alive())
# #<Process(Process-1, stopped[SIGTERM])> False
# p.exitcode == -signal.SIGTERM
# #True

"""
multiprocessing.Pipe([duplex])
返回一对 Connection`对象  ``(conn1, conn2)` ， 分别表示管道的两端。

如果 duplex 被置为 True (默认值)，那么该管道是双向的。如果 duplex 被置为 False ，那么该管道是单向的，即 conn1 只能用于接收消息，而 conn2 仅能用于发送消息
"""

from multiprocessing import Process, Pipe, current_process
from multiprocessing.connection import wait

def foo(w):
    for i in range(10):
        w.send((i, current_process().name))
    w.close()


if __name__ == '__main__':

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