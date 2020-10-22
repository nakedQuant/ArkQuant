# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from functools import wraps
import threading

context = threading.local()


def get_algo_instance():
    return getattr(context, 'algorithm', None)


def set_algo_instance(algo):
    context.algorithm = algo


class ZiplineAPI(object):
    """
    Context manager for making an algorithm instance available to zipline API
    functions within a scoped block.
    多线程编程中的对同一变量的访问冲突的一种技术，TLS会为每一个线程维护一个和该线程绑定的变量的副本。而不是无止尽的传递局部参数的方式编程
    每一个线程都拥有自己的变量副本，并不意味着就一定不会对TLS变量中某些操作枷锁了。
    Java平台的java.lang.ThreadLocal和Python 中的threading.local()都是TLS技术的一种实现
    TLS使用的缺陷是，如果你的线程都不退出，那么副本数据可能一直不被GC回收，会消耗很多资源，比如线程池中，线程都不退出，使用TLS需要非常小心
    """

    def __init__(self, algo_instance):
        self.algo_instance = algo_instance

    def __enter__(self):
        """
        Set the given algo instance, storing any previously-existing instance.
        """
        self.old_algo_instance = get_algo_instance()
        set_algo_instance(self.algo_instance)

    def __exit__(self, _type, _value, _tb):
        """
        Restore the algo instance stored in __enter__.
        """
        set_algo_instance(self.old_algo_instance)
