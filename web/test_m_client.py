# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

from multiprocessing.managers import BaseManager

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




