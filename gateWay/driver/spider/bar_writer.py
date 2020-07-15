#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""
import json,pandas as pd
from sqlalchemy import select,MetaData

from gateWay.driver.reconstruct import  _parse_url
from gateWay.driver.db_writer import  DBWriter

#kline
equity_bundles = 'http://64.push2his.eastmoney.com/api/qt/stock/kline/get?&secid={}&fields1=f1&' \
                'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&end=30000101&lmt={}'

bond_bundles = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5' \
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101'

fund_bundles = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1&' \
               'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101',

dual_bundles = 'http://94.push2his.eastmoney.com/api/qt/stock/kline/get?secid=116.08231&fields1=f1' \
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&klt=101&fqt=1&end=20500101&lmt=2'

ASSETS_BUNDLES_URL = {'equity_bundles':equity_bundles,'bond_bundles':bond_bundles,
                      'fund_bundles':fund_bundles,'dual_bundles':dual_bundles}

dBwriter = DBWriter()


class BundlesWriter(object):

    _default_cols = ['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'amount']

    def __init__(self,engine,lmt = 100000):
        self.engine = engine
        self.lmt = lmt

    @property
    def metadata(self):
        metadata = MetaData(bind = self.engine)
        return metadata

    @property
    def default(self):
        return self._default_cols

    def _retrieve_assets_from_sqlite(self):
        table = self.metadata.tables['asset_router']
        ins = select([table.c.sid,table.c.asset_type])
        rp = self.engine.execute(ins)
        assets = pd.DataFrame(rp.fetchall(),columns = ['sid','asset_type'])
        assets.set_index('sid',inplace = True)
        mappings = assets.groupby('asset_type').groups
        return mappings

    def _request_kline_for_dfcf(self,sid,tbl,extend = False):
        url = ASSETS_BUNDLES_URL[tbl].format(sid['request_sid'],self.lmt)
        obj = _parse_url(url,bs =False)
        raw = json.loads(obj)
        kline = raw['data']
        cols = self.default + ['pct'] if extend else self.default
        if kline and len(kline['klines']):
            kl_pd = pd.DataFrame([item.split(',') for item in kline['klines']],
                                 columns=cols)
            kl_pd.loc[:,'sid'] = sid['sid']
            if hasattr(sid,'bond_id'):
                kl_pd.loc[:,'bond_id'] = sid['bond_id']
            dBwriter.writer(tbl,kl_pd)

    def download_equity_kline(self,sid):
        sid_id = '1.' + sid if sid.startswith('6') else '0.' + sid
        self._request_kline_for_asset({'request_sid':sid_id,'sid':sid},'equity_price',extend = True)

    def download_fund_kline(self,fund):
        fund_id = '1.'+ fund[2:] if fund.startswith('sh') else '0.' + fund[2:]
        self._request_kline_for_asset({'request_sid':fund_id,'sid':fund},'fund_price')

    def download_convertible_kline(self,bond):
        sid = bond['cell']['stock_id']
        bond_id = '0.' + bond['id'] if sid.startswith('sz') else '1.' + bond['id']
        self._request_kline_for_asset({'request_sid':bond_id,'sid':sid,'bond_id':bond['id']},'bond_price')

    def download_dual_kline(self,h_symbol):
        sid = ('.').join(['116',h_symbol])
        self._request_kline_for_asset({'request_sid':sid,'sid':h_symbol},'dual_price')

    def writer(self):
        asset_mappings = self._retrieve_assets_from_sqlite()
        from multiprocessing import Pool
        pool = Pool(4)
        for method_name in ['equity','fund','convertible','dual']:
            method = getattr(self,'download_%s_kline'%method_name)
            pool.apply_async(method,asset_mappings[method_name])