# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from functools import wraps
import pandas as pd

from GateWay.Driver.Engine import BarReader

class Event:
    """
        dt : str or list
        asset --- None specify all stocks
    """
    def __init__(self, dt, asset=None):
        self.trading_dt = dt
        self.code = asset


class GateReq:
    """
        when window is None specify obtain data from  the initial
        field : str or list of str or expr , nullable
    """
    def __init__(self, event, fields, window=None, extend=2):
        self._verify(event.trading_dt, window)
        self.session = event.trading_dt
        self.asset = event.code
        self.field = fields
        self.window = window * extend if window else None

    def _verify(self, dt, window):
        if isinstance(dt, (tuple, list)) and window is not None:
            raise ValueError('when dt is tuple or list ,window must be None')

def type_to_float64(f):
    @wraps(f)
    def wrapper(*args,**kwargs):
        res = f(*args,**kwargs)
        for c in res.columns:
            if c not in ['trade_dt','code','h_code']:
                res[c] = res[c].astype('float64')
        return res
    return wrapper


def adjust(f):
    """相关的计算逻辑需要为series,如果为dataframe产生结果为list(因为columns)"""
    def wrapper(*args,**kwargs):
        res = f(*args,**kwargs)
        if  isinstance(res,pd.DataFrame) and len(res.columns) ==1:
            res = res.iloc[:,0]
        return res
    return wrapper


class Quandle:

    def __init__(self):

        self.bar = BarReader()

    def query_calendar_session(self,sdate,edate):
        trade_list = self.bar.load_trading_calendar(sdate,edate)
        return trade_list

    def query_calendar_offset(self, dt,window):
        """向前"""
        offset = self.bar.load_calendar_offset(dt, -window)
        return offset

    def _parse_request_session(self,request):
        if request.window:
            start = self.query_calendar_offset(request.session,request.window)
            end = request.session
        else:
            start = request.session[0]
            end = request.session[1]
        return start,end

    @adjust
    @type_to_float64
    def query_ashare_kline(self, request,mode = None):
        sdate,edate = self._parse_request_session(request)
        if mode == 'hfq':
            response = self.bar.load_stock_hfq_kline(sdate, edate,request.field,request.asset)
        else:
            response = self.bar.load_stock_kline(sdate, edate,request.field,request.asset)
        return response

    @adjust
    @type_to_float64
    def query_hk_kline(self, request):
        s,e = self._parse_request_session(request)
        response = self.bar.load_hk_kline(s, e,request.field,request.asset)
        return response

    @adjust
    @type_to_float64
    def query_index_kline(self, request):
        begin,end = self._parse_request_session(request)
        response = self.bar.load_index_kline(begin,end,request.field,request.asset)
        return response

    @adjust
    @type_to_float64
    def query_etf_kline(self, request):
        s,e = self._parse_request_session(request)
        response = self.bar.load_etf_kline(s,e,request.field,request.asset)
        return response

    @adjust
    @type_to_float64
    def query_convertible_kline(self, request):
        s,e = self._parse_request_session(request)
        response = self.bar.load_convertible_kline(s,e,request.field,request.asset)
        return response

    @adjust
    @type_to_float64
    def query_ashare_cap(self, request):
        """获取流通市值"""
        s, e = self._parse_request_session(request)
        data = self.bar.load_market_value(s, e, request.asset)
        cap = data.loc[:, ['trade_dt', 'code', 'cap']]
        return cap

    @adjust
    @type_to_float64
    def query_ashare_strict(self, request):
        """获取受限市值"""
        s, e = self._parse_request_session(request)
        data = self.bar.load_market_value(s, e, request.asset)
        strict = data.loc[:, ['trade_dt', 'code', 'strict']]
        return strict

    @adjust
    @type_to_float64
    def query_ashare_hk(self, request):
        """获取港股流通市值 --- 对象AH"""
        s, e = self._parse_request_session(request)
        data = self.bar.load_market_value(s, e, request.asset)
        hk = data.loc[:, ['trade_dt', 'code', 'hk']]
        return hk

    @adjust
    @type_to_float64
    def query_ashare_mkv(self, request):
        """获取总值市值"""
        s, e = self._parse_request_session(request)
        data = self.bar.load_market_value(s, e, request.asset)
        mkt = data.loc[:, ['trade_dt', 'code', 'mkv']]
        return mkt

    def query_mass(self, request):
        """股票大宗交易"""
        s,e = self._parse_request_session(request)
        raw = self.bar.load_ashare_mass(s,e)
        raw = raw.loc[:,['TDATE', 'SECUCODE', 'PRICE', 'TVOL','TVAL','RCHANGE','CPRICE','Zyl','Cjeltszb']]
        if request.asset:
            mass = raw[raw['SECUCODE']==request.asset]
        else:
            mass = raw
        #列名重命名
        mass.columns = ['交易日期','代码','成交价','成交量','成交额','涨跌幅','收盘价','折溢率','成交额占流通市值比率']
        mass.iloc[:,2:] = mass.iloc[:,2:].astype('float64')
        return mass

    def query_release(self,request):
        """A股解禁"""
        s,e = self._parse_request_session(request)
        raw = self.bar.load_ashare_release(s, e)
        if request.asset:
            release = raw[raw['代码']==request.asset]
        else:
            release = raw
        raw['解禁占流通市值比例'] = raw['解禁占流通市值比例'].astype('float64')
        return release

    def query_market_margin(self,s,e):
        margin = self.bar.load_market_margin(s,e)
        margin.iloc[:,1:] = margin.iloc[:,1:].astype('float64')
        return margin

    def query_hold_event(self,request):
        """ 增持、减持"""
        s,e = self._parse_request_session(request)
        hold = self.bar.load_stock_holdings(s,e,request.asset)
        hold.loc[:,['变动股本','占总流通比例','总持仓','占总股本比例','总流通股']] = \
            hold.loc[:,['变动股本','占总流通比例','总持仓','占总股本比例','总流通股']].astype('float64')
        return hold

    @adjust
    @type_to_float64
    def query_periphera_index(self,request,exchange = 'hk'):
        """us.DJI 道琼斯 us.IXIC 纳斯达克 us.INX  标普500 hkHSI 香港恒生指数 hkHSCEI 香港国企指数 hkHSCCI 香港红筹指数"""
        s, e = self._parse_request_session(request)
        foreign_index = self.bar.load_periphera_index(s,e,request.field,request.asset,exchange)
        return foreign_index

    def query_daily_minute(self,sid,ndays = None):
        """返回A股最近Ndays分时数据"""
        minute = self.bar.load_minute_kline(sid,ndays)
        return minute

    def query_5dhk_minute(self, sid):
        """返回最近5个交易日的分时数据 -- 价格、成交量"""
        minute_hk = self.bar.load_5d_minute_hk(sid)
        return minute_hk

    def query_basics(self, sid=None):
        basics = self.bar.load_ashare_basics(sid)
        return basics

    def query_split_divdend(self, sid):
        sd = self.bar.load_splits_divdend(sid)
        return sd

    def query_equity(self, sid):
        eq = self.bar.load_equity_info(sid)
        return eq

    def query_status(self,sid):
        """返回股票状态是否、退市--D、暂停上市--P"""
        status = self.bar.load_stock_status(sid)
        return status

    def query_convertible_desc(self, bond):
        desc = self.bar.load_convertible_basics(bond)
        return desc

    def query_hk_con(self, exchange,flag = 1):
        """获取沪港通、深港通标的"""
        res = self.bar.load_ashare_hk_con(exchange,flag)
        return res

    def query_gross_value(self):
        gdp = self.bar.load_gdp()
        return gdp


if __name__ == '__main__':

    quandle = Quandle()

    start = '2020-01-01'
    end = '2020-02-13'
    window = 20
    mode = 'hfq'
    sid = '000001'
    hk_sid = '00168'
    fields = ['close']
    bond = '128079'
    exchange = 'SH'
    # request = GateReq(Event([start,end]),fields)
    request = GateReq(Event(start),fields,50)
    request_hfq = GateReq(Event([start,end],'000001'),fields)
    request_foreign_index = GateReq(Event([start,end],'DJI'),fields)

    # session = quandle.query_calendar_session(start,end)
    # print('session',session)
    #
    # session_offset = quandle.query_calendar_offset(start, window)
    # print('session_offset',session_offset)
    #
    # kline = quandle.query_ashare_kline(request_hfq,mode)
    # print('ashare hfq kline',kline)
    # print(type(kline))

    kline = quandle.query_ashare_kline(request)
    print('ashare kline',kline)
    print(type(kline.iloc[0,0]))
    #
    # hk_kline = quandle.query_hk_kline(request)
    # print('hk_kline',hk_kline)
    # print(type(hk_kline.iloc[0,0]))
    #
    # index_kline = quandle.query_index_kline(request)
    # print('index_kline',index_kline)
    # print(type(index_kline))
    #
    # etf_kline = quandle.query_etf_kline(request)
    # print('etf_kline',etf_kline)
    # print(type(etf_kline.iloc[0,0]))
    #
    # convetible_kline = quandle.query_convertible_kline(request)
    # print('convertible_kline',convetible_kline)
    # print(type(convetible_kline.iloc[0,0]))

    # cap = quandle.query_ashare_cap(request)
    # print('cap',cap)
    # print(type(cap.loc[0,'cap']))
    #
    # strict = quandle.query_ashare_strict(request)
    # print('strict',strict)
    # print(type(strict.loc[0,'strict']))
    #
    # hk = quandle.query_ashare_hk(request)
    # print('hk',hk)
    # print(type(hk.loc[0,'hk']))
    #
    # mkv = quandle.query_ashare_mkv(request)
    # print('mkv',mkv)
    # print(type(mkv.loc[0,'mkv']))
    #
    # mass = quandle.query_mass(request)
    # print('mass',mass)
    #
    # release = quandle.query_release(request)
    # print('release',release)
    #
    # margin = quandle.query_market_margin(start,end)
    # print('margin',margin)
    #
    # daily = quandle.query_daily_minute(sid)
    # print('daily',daily)
    #
    # daily_hk = quandle.query_5dhk_minute(hk_sid)
    # print('daily_hk',daily_hk)

    # holdings = quandle.query_hold_event(request)
    # print('holdings',holdings)

    # basics = quandle.query_basics(sid)
    # print('basics',basics)
    #
    # sd = quandle.query_split_divdend(sid)
    # print('splits_divdend',sd)
    #
    # eq = quandle.query_equity(sid)
    # print('equity',eq)
    #
    # bond_desc = quandle.query_convertible_desc(bond)
    # print('bond_desc',bond_desc)
    #
    # status = quandle.query_status(sid)
    # print('status',status)
    #
    # hk_con = quandle.query_hk_con(exchange)
    # print('hk_con',hk_con)
    #
    # periphera = quandle.query_periphera_index(request_foreign_index,'us')
    # print('periphera',periphera)
    # print(type(periphera.iloc[0,0]))

    # gdp = quandle.query_gross_value()
    # print('gross value',gdp)