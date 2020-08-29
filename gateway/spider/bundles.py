# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, pandas as pd
from threading import Thread
from functools import partial
from gateway.database.db_writer import db
from gateway.spider import Crawler
from gateway.spider.url import ASSETS_BUNDLES_URL
from gateway.driver.tools import _parse_url

__all__ = ['BundlesWriter']

Method = frozenset(['equity', 'fund', 'convertible'])


class BundlesWriter(Crawler):
    """
        a. obtain asset from mysql
        b. request kline from dfcf
        c. update to mysql
    """

    def __init__(self, lmt=100000):
        self.lmt = lmt

    @property
    def default(self):
        return ['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'amount']

    def _crawler(self, mapping, tbl, pct=False):
        url = ASSETS_BUNDLES_URL[tbl].format(mapping['request_sid'], self.lmt)
        obj = _parse_url(url, bs=False)
        kline = json.loads(obj)['data']
        print('kline', kline)
        cols = self.default + ['pct'] if pct else self.default
        if kline and len(kline['klines']):
            frame = pd.DataFrame([item.split(',') for item in kline['klines']],
                                 columns=cols)
            frame.loc[:, 'sid'] = mapping['sid']
            db.writer(tbl, frame)

    def request_equity_kline(self, sid):
        sid_id = '1.' + sid if sid.startswith('6') else '0.' + sid
        self._crawler({'request_sid': sid_id, 'sid': sid}, 'equity_price', pct=True)

    def request_fund_kline(self, sid):
        # fund 以1或者5开头 --- 5（SH） 1（SZ）
        fund_id = '1.' + sid if sid.startswith('5') else '0.' + sid
        self._crawler({'request_sid': fund_id, 'sid': sid}, 'fund_price')

    def request_convertible_kline(self, sid):
        # 11开头 --- 6 ； 12开头 --- 0或者3
        bond_id = '1.' + sid if sid.startswith('11') else '0.' + sid
        self._crawler({'request_sid': bond_id, 'sid': sid}, 'convertible_price')

    # def request_dual_kline(self, h_symbol):
    #     sid = '.'.join(['116', h_symbol])
    #     self._crawler({'request_sid': sid, 'sid': h_symbol}, 'dual_price')

    # def _implement(self, method_name, q):
    #     method = getattr(self, 'request_{}_kline'.format(method_name))
    #     for sid in q[method_name]:
    #         method(sid)

    # def writer(self):
    #     # q = self._retrieve_assets_from_sqlite()
    #     q = {'equity': ['300357'], 'fund': ['512760'], 'convertible': ['113581']}
    #     _main_func = partial(self._implement, q=q)
    #     # multi thread
    #     threads = []
    #     for method_name in Method:
    #         thread = Thread(target=_main_func, kwargs={'method_name': method_name})
    #         thread.start()
    #         threads.append(thread)
    #
    #     for t in threads:
    #         print(thread.is_alive())
    #         t.join()

    def _implement(self, method_name, q):
        method = getattr(self, 'request_{}_kline'.format(method_name))
        threads = []
        for sid in q[method_name]:
            print('sid', sid)
            # 事件对象（线程之间通信）可以控制线程数量通过event set() clear() wait()
            # Semaphore(value=1) 代表 release() 方法的调用次数减去 acquire() 的调用次数再加上一个初始值(Value)
            # threading.BoundedSemaphore(value) 有界信号量通过检查以确保它当前的值不会超过初始值。如果超过了初始值，将会引发 ValueError 异常(上下文）
            # Semaphore --- return threading instance as contextmanager
            thread = Thread(target=method, kwargs={'sid': sid}, name=sid)
            thread.start()
            threads.append(thread)

        for t in threads:
            print(t.is_alive())
            t.join()

    def writer(self):
        # q = self._retrieve_assets_from_sqlite()
        q = {'equity': ['300357', '600636'], 'fund': ['512760', '512880'], 'convertible': ['113581']}
        _main_func = partial(self._implement, q=q)
        # multi thread
        threads = []
        for method_name in Method:
            thread = Thread(target=_main_func, kwargs={'method_name': method_name})
            thread.start()
            threads.append(thread)

        for t in threads:
            print(thread.is_alive())
            t.join()


if __name__ == '__main__':

    bundle = BundlesWriter()
    bundle.writer()
    # bundle.request_equity_kline('300357')
    # bundle.request_convertible_kline('113581')
    # bundle.request_fund_kline('512760')
