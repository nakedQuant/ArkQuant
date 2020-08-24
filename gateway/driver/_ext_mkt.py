#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019
@author: python
"""
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# market value and vwap


class MarketValue:

    def __init__(self, mode):
        self.frequency = mode

    def enroll_market_value(self,asset,sdate):
        conn = self.db.db_init()
        raw = self.bar.load_equity_info(asset)
        # 将日期转为交易日
        raw.loc[:, 'trade_dt'] = [t if self.bar.is_market_caledar(t) else self.bar.load_calendar_offset(t, 1) for t in
                                  raw.loc[:, 'change_dt'].values]
        """由于存在一个变动时点出现多条记录，保留最大total_assets的记录,先按照最大股本降序，保留第一个记录"""
        raw.sort_values(by='total_assets', ascending=False, inplace=True)
        raw.drop_duplicates(subset='trade_dt', keep='first', inplace=True)
        raw.index = raw['trade_dt']
        close = self.bar.load_stock_kline(sdate,self.edate, ['close'], asset)
        if len(close) == 0:
            print('code:%s has not kline' % asset)
        else:
            # 数据对齐
            if self.frequency:
                raw.sort_index(ascending=False, inplace=True)
                raw = pd.DataFrame(raw.iloc[0, :]).T
                raw.index = close.index
            close.loc[:, 'total'] = raw.loc[:, 'total_assets']
            close.loc[:, 'float'] = raw.loc[:, 'float_assets']
            close.loc[:, 'strict'] = raw.loc[:, 'strict_assets']
            close.loc[:, 'b_assets'] = raw.loc[:, 'b_assets']
            close.loc[:, 'h_assets'] = raw.loc[:, 'h_assets']
            close.fillna(method='ffill', inplace=True)
            close.fillna(method='bfill', inplace=True)
            # 计算不同类型市值
            mkt = close.loc[:, 'total'] * close.loc[:, 'close']
            cap = close.loc[:, 'float'] * close.loc[:, 'close']
            strict = close.loc[:, 'strict'] * close.loc[:, 'close']
            b = close.loc[:, 'b_assets'] * close.loc[:, 'close']
            h = close.loc[:, 'h_assets'] * close.loc[:, 'close']
            # 调整格式并入库
            data = pd.DataFrame([mkt, cap, strict, b, h]).T
            data.columns = ['mkt', 'cap', 'strict', 'foreign', 'hk']
            data.loc[:, 'trade_dt'] = data.index
            data.loc[:, 'code'] = asset
            self.db.enroll('mkt_value', data, conn)
            conn.close()

    def parallel(self):
        basics = self.bar.load_ashare_basics()
        assets = [item[0] for item in basics]
        if self.frequency:
            sdate = self.edate
        else:
            sdate = '1990-01-01'
        with ThreadPoolExecutor(max_workers = 10) as executor:
            f = []
            for asset in assets:
                print('asset',asset)
                future = executor.submit(self.enroll_market_value, asset, sdate)
                f.append(future)
            for job in as_completed(f):
                job.result()
