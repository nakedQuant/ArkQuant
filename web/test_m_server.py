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

