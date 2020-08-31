# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import multiprocessing, datetime
from concurrent.futures import ProcessPoolExecutor
from gateway.spider.asset_router import AssetRouterWriter
from gateway.spider.bundles import BundlesWriter
from gateway.spider.divdend_rights import AdjustmentsWriter
from gateway.spider.events import EventWriter

# 初始化各个spider module
router_writer = AssetRouterWriter()
adjust_writer = AdjustmentsWriter()
event_writer = EventWriter()


class SyncSource(object):

    def __init__(self, init=True):
        # router_writer = AssetRouterWriter()
        # adjust_writer = AdjustmentsWriter()
        # event_writer = EventWriter()
        self.bundel_writer = BundlesWriter(None if init else 1)
        # initialize or daily
        self.pattern = 'initialize' if init else 'daily'
        self._init()

    @classmethod
    def _init(cls):
        router_writer.writer()

    def __call__(self):
        iterable = [self.bundel_writer, adjust_writer]

        def when_done(r):
            # '''每一个进程结束后结果append到result中'''
            # result.append(r.result())
            print('future : %r finished' % r)

        if self.n_jobs <= 0:
            self.n_jobs = multiprocessing.cpu_count()

        if self.n_jobs == 1:

            for jb in iterable:
                # result.append(jb[0](*jb[1], **jb[2]))
                getattr(jb, 'writer')()

        else:
            with ProcessPoolExecutor(max_worker=self.n_jobs) as pool:
                for jb in iterable:
                    method = getattr(jb, 'writer')
                    # future_result = pool.submit(jb[0], *jb[1], **jb[2])
                    future_result = pool.submit(method)
                    future_result.add_done_callback(when_done)
        # execute event_writer
        edate = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        sdate = '1990-01-01' if self.pattern == 'initialize' else  edate
        event_writer.writer(sdate, edate)
        # return result


# if __name__ == '__main__':
#
#     # initialize
#     SyncSource()
#     # daily
#     SyncSource(init=False)
