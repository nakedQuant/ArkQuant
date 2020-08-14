# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, pandas as pd
from multiprocessing import Pool
from sqlalchemy import select
from gateway.driver.db_writer import db
from gateway.spider.base import Crawler
from.xml import ASSETS_BUNDLES_URL


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

    def _retrieve_assets_from_sqlite(self):
        table = self.metadata.tables['asset_router']
        ins = select([table.c.sid,table.c.asset_type])
        rp = self.engine.execute(ins)
        assets = pd.DataFrame(rp.fetchall(), columns=['sid', 'asset_type'])
        assets.set_index('sid', inplace=True)
        mappings = assets.groupby('asset_type').groups
        return mappings

    def _crawler(self, mapping, tbl, pct=False):
        url = ASSETS_BUNDLES_URL[tbl].format(mapping['request_sid'], self.lmt)
        obj = self.tool(url, bs=False)
        kline = json.loads(obj)['data']
        cols = self.default + ['pct'] if pct else self.default
        if kline and len(kline['klines']):
            frame = pd.DataFrame([item.split(',') for item in kline['klines']],
                                 columns=cols)
            frame.loc[:, 'sid'] = mapping['sid']
            if hasattr(mapping, 'swap_code'):
                frame.loc[:, 'swap_code'] = mapping['swap_code']
            db.writer(tbl, frame)

    def request_equity_kline(self, sid):
        sid_id = '1.' + sid if sid.startswith('6') else '0.' + sid
        self._crawler({'request_sid': sid_id, 'sid': sid}, 'equity_price', pct=True)

    def request_fund_kline(self, fund):
        fund_id = '1.' + fund[2:] if fund.startswith('sh') else '0.' + fund[2:]
        self._crawler({'request_sid': fund_id, 'sid': fund}, 'fund_price')

    def request_convertible_kline(self, bond):
        symbol = bond['cell']['stock_id']
        bond_id = '0.' + bond['id'] if symbol.startswith('sz') else '1.' + bond['id']
        self._crawler({'request_sid': bond_id, 'sid': bond['id'], 'swap_code': symbol}, 'bond_price')

    # def request_dual_kline(self, h_symbol):
    #     sid = '.'.join(['116', h_symbol])
    #     self._crawler({'request_sid': sid, 'sid': h_symbol}, 'dual_price')

    def writer(self):
        q = self._retrieve_assets_from_sqlite()
        pool = Pool(4)
        for method_name in ['equity', 'fund', 'convertible', 'dual']:
            method = getattr(self, 'request_%s_kline' % method_name)
            pool.apply_async(method, q[method_name])
