# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, pandas as pd, datetime, threading
from toolz import valmap
from collections import defaultdict
from threading import Thread, Lock
from functools import partial
from gateway.database.db_writer import db
from gateway.spider import Crawler
from gateway.spider.url import ASSETS_BUNDLES_URL
from gateway.driver.tools import _parse_url

__all__ = ['BundlesWriter']

sema = threading.Semaphore(100)


class BundlesWriter(Crawler):
    """
        a. obtain asset from mysql
        b. request kline from dfcf
        c. update to mysql
    """
    def __init__(self, lmt):
        self.lmt = lmt if lmt else (datetime.datetime.now() - datetime.datetime(1990, 1, 1)).days
        self._cache_deadlines = {}
        self.missed = defaultdict(set)

    @property
    def default(self):
        return ['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'amount']

    def retrieve_asset_deadlines(self):
        for tbl in ['equity_price', 'fund_price', 'convertible_price']:
            self._cache_deadlines[tbl] = self._retrieve_latest_from_sqlite(tbl)

    def _crawler(self, mapping, tbl, pct=False):
        sid = mapping['sid']
        url = ASSETS_BUNDLES_URL[tbl].format(mapping['request_sid'], self.lmt)
        obj = _parse_url(url, bs=False)
        kline = json.loads(obj)['data']
        cols = self.default + ['pct'] if pct else self.default
        if kline and len(kline['klines']):
            frame = pd.DataFrame([item.split(',') for item in kline['klines']],
                                 columns=cols)
            frame.loc[:, 'sid'] = sid
            # 过滤
            try:
                deadline = self._cache_deadlines[tbl][sid]
            except Exception as e:
                print('error :%s raise from sid come to market today' % e)
                deadline = None
            # frame = frame[frame['trade_dt'] > self._cache_deadlines[tbl][sid]]
            frame = frame[frame['trade_dt'] > deadline] if deadline else frame
            db.writer(tbl, frame)

    def request_equity_kline(self, sid):
        with sema:
            sid_id = '1.' + sid if sid.startswith('6') else '0.' + sid
            try:
                self._crawler({'request_sid': sid_id, 'sid': sid}, 'equity_price', pct=True)
            except Exception as e:
                print('spider %s  equity kline failure due to %r' % (sid, e))
                self.missed['equity'].add(sid)
            else:
                self.missed['equity'].discard(sid)

    def request_fund_kline(self, sid):
        # fund 以1或者5开头 --- 5（SH） 1（SZ）
        with sema:
            fund_id = '1.' + sid if sid.startswith('5') else '0.' + sid
            try:
                self._crawler({'request_sid': fund_id, 'sid': sid}, 'fund_price')
            except Exception as e:
                print('spider %s  fund kline failure due to %r' % (sid, e))
                self.missed['fund'].add(sid)
            else:
                self.missed['fund'].discard(sid)

    def request_convertible_kline(self, sid):
        # 11开头 --- 6 ； 12开头 --- 0或者3
        with sema:
            bond_id = '1.' + sid if sid.startswith('11') else '0.' + sid
            try:
                self._crawler({'request_sid': bond_id, 'sid': sid}, 'convertible_price')
            except Exception as e:
                print('spider %s  convertible kline failure due to %r' % (sid, e))
                self.missed['convertible'].add(sid)
            else:
                self.missed['convertible'].discard(sid)

    def _implement(self, method_name, q):
        method = getattr(self, 'request_{}_kline'.format(method_name))
        threads = []
        lock = Lock()
        if len(q[method_name]):
            for sid in q[method_name]:
                try:
                    print('sid', sid)
                    # 事件对象（线程之间通信）可以控制线程数量通过event set() clear() wait()
                    # Semaphore(value=1) 代表 release() 方法的调用次数减去 acquire() 的调用次数再加上一个初始值(Value)
                    # threading.BoundedSemaphore(value) 有界信号量通过检查以确保它当前的值不会超过初始值。如果超过了初始值，将会引发 ValueError 异常(上下文）
                    # Semaphore --- return threading instance as contextmanager
                    # if len(threads) <= MaxThreads:
                    thread = Thread(target=method, kwargs={'sid': sid}, name=sid)
                    thread.start()
                except RuntimeError:
                    lock.acquire()
                    for t in threads:
                        print(t.is_alive())
                        t.join()
                    lock.release()
                    thread.start()
                    threads = []
                finally:
                    threads.append(thread)

            for t in threads:
                print(t.is_alive())
                t.join()

    def rerun(self):
        dct = valmap(lambda x: len(x), self.missed)
        if sum(dct.values()) != 0:
            missed = self.missed.copy()
            # RuntimeError: Set changed size during iteration
            self._writer_internal(missed)
            self.rerun()
        self.missed = defaultdict(list)
        self._cache_deadlines = {}

    def _writer_internal(self, q):
        _main_func = partial(self._implement, q=q)
        # multi thread
        threads = []
        for method_name in ['equity', 'convertible', 'fund']:
            thread = Thread(target=_main_func, kwargs={'method_name': method_name})
            thread.start()
            threads.append(thread)

        for t in threads:
            print(thread.is_alive())
            t.join()

    def writer(self):
        self.retrieve_asset_deadlines()
        print('_cache_deadlines', self._cache_deadlines)
        q = self._retrieve_assets_from_sqlite()
        self._writer_internal(q)
        print('failure bundles asset: %r' % self.missed)
        self.rerun()


# if __name__ == '__main__':
#
#     bundle = BundlesWriter(lmt=1)
#     bundle.writer()
#     # 琼民源A
#     bundle.request_equity_kline('688526')
#     bundle.request_convertible_kline('113602')
#     bundle.request_fund_kline('512760')
