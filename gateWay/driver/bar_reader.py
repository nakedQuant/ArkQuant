# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
from sqlalchemy import create_engine,MetaData,select,and_,cast ,Numeric
import pandas as pd

class BarReader(object):

    def __init__(self,engine_path,
                 level = 'READ UNCOMMITTED'):
        self._init_db(engine_path,level)

    def _init_db(self,engine_path,level):
        engine = create_engine(engine_path)
        self.tabls = MetaData(bind=engine).tables
        self.conn = engine.connect(_execution_options = {'isolation_level':level})

    def _load_raw_arrays(self,start_date,end_date,asset,tbl):
        tbl = self.tables[tbl]
        rp = select([tbl.c.trade_dt,tbl.c.code,
                      cast(tbl.c.open,Numeric(10,2)).label('open'),
                      cast(tbl.c.close,Numeric(12,2)).label('close'),
                      cast(tbl.c.high,Numeric(10,2)).label('high'),
                      cast(tbl.c.low,Numeric(10,3)).label('low'),
                      cast(tbl.c.volume,Numeric(15,0)).label('volume'),
                      cast(tbl.c.volume,Numeric(15,2)).label('amount')]). \
             where(and_(tbl.c.code == asset,
                        tbl.c.trade_dt.between(start_date, end_date))).execute()
        arrays = [[r.trade_dt,r.code,r.open,r.close,r.high,r.low,r.volume] for r in rp.fetchall()]
        kline = pd.DataFrame(arrays,columns = ['trade_dt','code','open','close','high','low','volume','amount'])
        return kline

    def load_daily_symbol(self,sdate, edate,fields,asset):
        """
            返回特定时间区间日股票K线
        """
        kline = self._load_raw_arrays(sdate,edate,asset,'symbol_price')
        kl_pd = kline.loc[:,fields]
        return kl_pd

    def load_daily_fund(self, sdate, edate,fields,asset):
        kline = self._load_raw_arrays(sdate,edate,asset,'fund_price')
        fund_kline = kline.loc[:,fields]
        etf_pd = fund_kline.loc[:,fields]
        return etf_pd

    def load_daily_bond(self, sdate, edate,fields,asset):
        """
            返回特定时间区间日可转债K线
        """
        kline = self._load_raw_arrays(sdate,edate,asset,'bond_price')
        bond_kline = kline.loc[:,fields]
        bond_pd = bond_kline.loc[:,fields]
        return bond_pd

    def load_daily_dual(self, sdate, edate,fields, asset):
        """
            A H 同时上市
        """
        kline = self._load_raw_arrays(sdate, edate, asset, 'dual_symbol_price')
        hk_kline = kline.loc[:, fields]
        return hk_kline

    def load_minute_symbol(self, sid):
        """
            最近5个交易日的日数据
        """