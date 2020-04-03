# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
from sqlalchemy import select,and_,cast ,Numeric,desc,Integer
from GateWay.Driver import DataLayer

class Core:
    """
        sqlalchemy core 操作
    """
    def __init__(self):

        self.db = DataLayer()
        self.tables = self.db.metadata.tables

    def _proc(self,ins):
        rp = self.db.engine.execute(ins)
        return rp

    def load_calendar(self,sdate,edate):
        """获取交易日"""
        sdate = sdate.replace('-','')
        edate = edate.replace('-','')
        table = self.tables['ashareCalendar']
        ins = select([table.c.trade_dt]).where(table.c.trade_dt.between(sdate,edate))
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        return trade_dt

    def is_calendar(self,dt):
        """判断是否为交易日"""
        dt = dt.replace('-','')
        table = self.tables['ashareCalendar']
        ins = select([table.c.trade_dt])
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        flag = dt in trade_dt
        return flag

    def load_calendar_offset(self,date,sid):
        date = date.replace('-','')
        table = self.tables['ashareCalendar']
        if sid > 0 :
            ins = select([table.c.trade_dt]).where(table.c.trade_dt > date)
            ins = ins.order_by(table.c.trade_dt)
        else :
            ins = select([table.c.trade_dt]).where(table.c.trade_dt < date)
            ins = ins.order_by(desc(table.c.trade_dt))
        ins = ins.limit(abs(sid))
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        return trade_dt

    def load_kline(self,start_date,end_date,asset,tbl):
        """获取Stock 、 ETF 、Index 、Convertible Hk"""
        tbl = self.tables[tbl]
        if asset:
            ins = select([tbl.c.trade_dt,tbl.c.code,cast(tbl.c.open,Numeric(10,2)).label('open'),cast(tbl.c.close,Numeric(12,2)).label('close'),
                        cast(tbl.c.high,Numeric(10,2)).label('high'),cast(tbl.c.low,Numeric(10,3)).label('low'),
                         cast(tbl.c.volume,Numeric(15,0)).label('volume')]). \
                where(and_(tbl.c.code ==asset,tbl.c.trade_dt.between(start_date, end_date)))
        else:
            ins = select([tbl.c.trade_dt,tbl.c.code,cast(tbl.c.open,Numeric(10,2)).label('open'),cast(tbl.c.close,Numeric(12,2)).label('close'),
                        cast(tbl.c.high,Numeric(10,2)).label('high'),cast(tbl.c.low,Numeric(10,2)).label('low'),
                         cast(tbl.c.volume,Numeric(15,0)).label('volume')]).\
                        where(tbl.c.trade_dt.between(start_date,end_date))
        rp = self._proc(ins)
        kline = [[r.trade_dt,r.code,r.open,r.close,r.high,r.low,r.volume] for r in rp.fetchall()]
        return kline

    def load_ashare_kline_offset(self,dt,window,asset):
        tbl = self.tables['asharePrice']
        ins = select([tbl.c.trade_dt,cast(tbl.c.open,Numeric(10,2)).label('open'),cast(tbl.c.close,Numeric(12,2)).label('close'),
                    cast(tbl.c.high,Numeric(10,2)).label('high'),cast(tbl.c.low,Numeric(10,3)).label('low'),
                     cast(tbl.c.volume,Numeric(15,0)).label('volume')]).\
                    where(and_(tbl.c.code == asset,tbl.c.trade_dt <= dt))
        ins = ins.order_by(desc(tbl.c.trade_dt))
        ins = ins.limit(window)
        rp = self._proc(ins)
        kline = [[r.trade_dt,r.open,r.close,r.high,r.low,r.volume] for r in rp.fetchall()]
        return kline

    def load_stock_basics(self,asset):
        """ 股票基础信息"""
        table = self.tables['ashareInfo']
        if asset:
            ins = select([table]).where(table.c.代码 == asset)
        else:
            ins = select([table])
        rp = self._proc(ins)
        basics = rp.fetchall()
        return basics

    def load_convertible_basics(self,asset):
        """ 可转债基础信息"""
        table = self.tables['convertibleDesc']
        ins = select([table]).where(table.c.bond_id == asset)
        rp = self._proc(ins)
        basics = rp.fetchall()
        return basics[0]

    def load_equity_structure(self,asset):
        """
            股票的总股本、流通股本，公告日期,变动日期结构
            Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
            Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
            --- 将 -- 变为0
        """
        table = self.tables['ashareEquity']
        ins = select([table.c.代码,table.c.变动日期,table.c.公告日期,cast(table.c.总股本,Numeric(20,3)).label('总股本'),
                     cast(table.c.流通A股,Numeric(20,3)),cast(table.c.限售A股,Numeric(20,3)),cast(table.c.流通B股,Numeric(20,3)),cast(table.c.流通H股,Numeric(20,3))]).where(table.c.代码 == asset)
        rp = self._proc(ins)
        equtiy = rp.fetchall()
        return equtiy

    def load_splits_divdend(self,asset):
        """股票分红配股"""
        table = self.tables['splitsDivdend']
        ins = select([table.c.除权除息日,cast(table.c.送股,Numeric(5,2)),cast(table.c.转增,Numeric(5,2)),cast(table.c.派息,Numeric(5,2))]).where\
            (and_(table.c.代码 == asset,table.c.进度.like('实施')))
        rp = self._proc(ins)
        splits_divdend = rp.fetchall()
        return splits_divdend

    def load_ashare_mkv(self,sdate,edate,asset):
        """股票流通市值、总市值、B股市值，H股市值"""
        table = self.tables['ashareValue']
        if asset:
            ins = select([table.c.trade_dt,table.c.code,cast(table.c.mkt,Numeric(20,5)),cast(table.c.cap,Numeric(20,5)),cast(table.c.strict,Numeric(20,5)),cast(table.c.hk,Numeric(20,5))]).where\
                (and_(table.c.code == asset,table.c.trade_dt.between(sdate,edate)))
        else:
            ins = select([table.c.trade_dt,table.c.code,cast(table.c.mkt,Numeric(20,5)),cast(table.c.cap,Numeric(20,5)),cast(table.c.strict,Numeric(20,5)),cast(table.c.hk,Numeric(20,5))]).where\
                (table.c.trade_dt.between(sdate,edate))
        rp = self._proc(ins)
        market_value = rp.fetchall()
        return market_value

    def load_ashare_holdings(self,sdate,edate,asset):
        """股东持仓变动"""
        table = self.tables['ashareHolding']
        if asset:
            ins = select(
                [table.c.变动截止日,table.c.代码, cast(table.c.变动股本, Numeric(10,2)), cast(table.c.占总流通比例, Numeric(10,5)), cast(table.c.总持仓, Numeric(10,2)),cast(table.c.占总股本比例,Numeric(10,5)),\
                 cast(table.c.总流通股, Numeric(10,2))]).where \
                (and_(table.c.代码 == asset,table.c.变动截止日.between(sdate,edate)))
        else:
            ins = select(
                [table.c.变动截止日,table.c.代码, cast(table.c.变动股本, Numeric(10,2)), cast(table.c.占总流通比例, Numeric(10,5)), cast(table.c.总持仓, Numeric(10,2)),cast(table.c.占总股本比例,Numeric(10,5)),\
                 cast(table.c.总流通股, Numeric(10,2))]).where \
                (table.c.变动截止日.between(sdate,edate))
        rp = self._proc(ins)
        margin = rp.fetchall()
        return margin

    def load_market_margin(self,start,end):
        """A股市场的融资融券情况"""
        table = self.tables['marketMargin']
        ins = select([table.c.交易日期,cast(table.c.融资余额,Integer),cast(table.c.融券余额,Integer),cast(table.c.融资融券总额,Integer),cast(table.c.融资融券差额,Integer)]).where\
            (table.c.交易日期.between(start,end))
        rp = self._proc(ins)
        margin = rp.fetchall()
        return margin

    def load_stock_status(self,asset):
        """获取股票状态 : 退市、暂时上市"""
        table = self.tables['ashareStatus']
        if asset:
            ins = select([table.c.delist_date,table.c.status]).where(table.c.code == asset)
        else:
            ins = select([table.c.delist_date,table.c.status])
        rp = self._proc(ins)
        res = rp.fetchall()
        return res


if __name__ == '__main__':

    core= Core()

    # flag = core.is_calendar('2010-02-10')
    # print(flag)
    #
    # kline = core.load_kline('2009-02-01','2010-02-01','000001','asharePrice')
    # print('kline',kline)
    #
    # basics = core.load_stock_basics('000001')
    # print('basics',basics)
    #
    # basics_convertible = core.load_convertible_basics('123022')
    # print('basics_convertible',basics_convertible)
    #
    # equity = core.load_equity_structure('002570')
    # print('equity',equity)
    #
    # splits = core.load_splits_divdend('000001')
    # print('splits',splits)
    #
    # status = core.load_stock_status('000001')
    # print('status',status)
    #
    # trade_dt = core.load_calendar('20100201','20101002')
    # print('trade_dt',trade_dt)
    #
    # offset = core.load_calendar_offset('20100201',-10)
    # print('offset',offset)

    # margin = core.load_market_margin('2020-02-13')
    # print('margin',margin)

    # mkv = core.load_ashare_mkt('2020-01-01','2020-01-20',None)
    # print('mkv',mkv)
    #
    # mkt = core.load_ashare_holdings('2020-01-01','2020-02-28',None)
    # print('market value',mkt)