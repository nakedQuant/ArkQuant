#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from multiprocessing.managers import BaseManager
from multiprocessing import Queue

queue = Queue()


class QManager(BaseManager):
    """
        抽象方法register
    """
    pass


def func(x,y):
    z = x + y
    return z


QManager.register('add',func)
QManager.register('fifo',callable = lambda :  queue)

m = QManager(address = ('192.168.0.103',10000),authkey = b'test')
test = m.get_server()
test.serve_forever()



# client

class CManager(BaseManager):

    pass


CManager.register('add')


if __name__ == '__main__':
    m = CManager(address=('192.168.0.103', 50000), authkey=b'test')
    m.connect()
    res = m.add(4,5)
    q = m.fifo()
    x = 5
    while x <10:
        res = m.add(x,3)
        print(res)
        q.put(x)
        x = x +1

