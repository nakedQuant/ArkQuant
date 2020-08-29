# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from gateway.spider.asset_router import AssetRouterWriter
from gateway.spider.bundles import BundlesWriter
from gateway.spider.divdend_rights import AdjustmentsWriter
from gateway.spider.events import EventWriter


class Parallel(object):
    """
    from joblib import Memory,Parallel,delayed
    from math import sqrt

    cachedir = 'your_cache_dir_goes_here'
    mem = Memory(cachedir)
    a = np.vander(np.arange(3)).astype(np.float)
    square = mem.cache(np.square)
    b = square(a)
    Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))

    涉及return value --- concurrent | Thread Process
    """

    def __init__(self, n_jobs=2):

        self.n_jobs = n_jobs

    def __call__(self, iterable):

        result = []

        def when_done(r):
            '''每一个进程结束后结果append到result中'''
            result.append(r.result())

        if self.n_jobs <= 0:
            self.n_jobs = multiprocessing.cpu_count()

        if self.n_jobs == 1:

            for jb in iterable:
                result.append(jb[0](*jb[1], **jb[2]))
        else:
            with ProcessPoolExecutor(max_worker=self.n_jobs) as pool:
                for jb in iterable:
                    future_result = pool.submit(jb[0], *jb[1], **jb[2])
                    future_result.add_done_callback(when_done)
        return result

    def run_in_thread(func, *args, **kwargs):
        """
            多线程工具函数，不涉及返回值
        """
        from threading import Thread
        thread = Thread(target=func, args=args, kwargs=kwargs)
        # 随着主线程一块结束
        thread.daemon = True
        thread.start()
        return thread



