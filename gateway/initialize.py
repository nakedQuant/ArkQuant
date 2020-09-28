# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from gateway.spider.asset_router import AssetRouterWriter
from gateway.spider.bundle import BundlesWriter
from gateway.spider.divdend_rights import AdjustmentsWriter
from gateway.spider.ownership import OwnershipWriter
from gateway.spider.events import EventWriter
from gateway.driver._ext_mkt import MarketValue

__all__ = ['SyncSpider']

# 初始化各个spider module
router_writer = AssetRouterWriter()
adjust_writer = AdjustmentsWriter()
event_writer = EventWriter()
ownership_writer = OwnershipWriter()


class SyncSpider(object):

    def __init__(self, initialization=True, default=1):
        # initialize or daily
        self.n_jobs = multiprocessing.cpu_count()
        bundle_writer = BundlesWriter(None if initialization else default)
        self._iterable = [adjust_writer, bundle_writer, ownership_writer]
        self._init_date = '2000-01-01' if initialization else None
        self._mcap_writer = MarketValue(initialization)

    def __call__(self):
        # sync asset_router first
        router_writer.writer()

        def when_done(r):
            # 每一个进程结束后结果append到result中
            # result.append(r.result())
            print('future : %r finished' % r)

        if self.n_jobs == 1:
            for jb in self._iterable:
                # result.append(jb[0](*jb[1], **jb[2]))
                getattr(jb, 'writer')()
        else:
            with ThreadPoolExecutor(self.n_jobs) as pool:
                to_do = []
                for jb in self._iterable:
                    method = getattr(jb, 'writer')
                    # future_result = pool.submit(jb[0], *jb[1], **jb[2])
                    future = pool.submit(method)
                    future.add_done_callback(when_done)
                    to_do.append(future)
                    # 线程处理
                for f in as_completed(to_do):
                    f.result()
        # # update events
        event_writer.writer(self._init_date)
        # update m_cap
        self._mcap_writer.calculate_mcap()


if __name__ == '__main__':

    m = SyncSpider(initialization=False)
    m()
