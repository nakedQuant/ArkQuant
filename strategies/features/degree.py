# -*- coding : uft-8 -*-

import pandas as pd, numpy as np

from gateWay import Event,GateReq,Quandle
from strategies.features import BaseFeature,EMA,remove_na

class MarketWidth(BaseFeature):
    '''
        市场宽度上涨股票数量 /（上涨股票数量 + 下跌股票数量）
    '''
    @staticmethod
    def _calc_rate(data):
        if len(data):
            ret = data / data.shift(1) - 1
            ratio = len(ret[ret > 0] ) / len(ret[ret !=0])
            return ratio

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        close = raw.pivot(columns = 'code',values = 'close')
        # T -- minutes S -- second M -- month
        mw = close.apply(cls._calc_rate,axis =1)
        mw_windowed = mw.resample('%dD'%window).mean()
        return mw_windowed

class STIX(BaseFeature):
    '''
        a / d = 上涨股票数量 /（上涨股票数量 + 下跌股票数量） STIX为 a / d的22EMA
    '''
    _weight = None

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        mw = MarketWidth.calc_feature(raw,window)
        # mw.fillna(0,inplace = True)
        EMA._weight = cls._weight
        mw_windowed = EMA.calc_feature(mw,window)
        return mw_windowed

class UDR(BaseFeature):
    '''
       上涨股票的交易量之和除以下跌股票的交易量之和
    '''
    _n_fields = ['close','volume']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        raw_close = raw.pivot(columns = 'code',values = 'close')
        raw_volume = raw.pivot(columns = 'code',values = 'volume')
        upper_vol = raw_volume[raw_close > raw_close.shift(1)]
        bottom_vol = raw_volume[raw_close < raw_close.shift(1)]
        udr = upper_vol.sum(axis=1) / bottom_vol.sum(axis=1)
        return udr

class FeatureMO(BaseFeature):
    '''
    正常的牛市伴随大量的价格适度上涨；而变弱的牛市少数股票大涨（背离信号）
    （上涨 - 下跌）的10 % EMA - 5 % 的EMA
    '''
    _pairwise = True

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        close = raw.pivot(columns = 'code',values = 'close')
        sign= np.sign(close - close.shift(1))
        print('sign',sign)
        minus = sign.sum(axis =1)
        print('minus',minus)
        mo_slow = EMA.calc_feature(minus,np.array(window).max())
        print('mo_slow',len(mo_slow),mo_slow)
        mo_fast = EMA.calc_feature(minus,np.array(window).min())
        print('mo_fast',len(mo_fast),mo_fast)
        # mo_windowed = mo_slow - mo_fast[-len(mo_slow):]
        mo_windowed = mo_slow - mo_fast
        return mo_windowed

class Trim(BaseFeature):
    '''
       对阿姆斯指标平滑得出（(上涨股票数量10日加总) /（下跌股票数量10日加总）） / （对应的成交量）
    '''
    _n_fields = ['close','volume']

    @staticmethod
    def _init_trim(data):
        close_pivot = data.pivot(columns = 'code',values = 'close')
        vol_pivot = data.pivot(columns = 'code',values = 'volume')
        sign_up  = close_pivot > close_pivot.shift(1)
        sign_down  = close_pivot < close_pivot.shift(1)
        sign_ratio = (sign_up.sum()).sum() / (sign_down.sum()).sum()
        vol_up = (vol_pivot[sign_up].sum()).sum()
        vol_down =(vol_pivot[sign_down].sum()).sum()
        trim =  sign_ratio * vol_down / vol_up
        return trim

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        trim = cls._init_trim(raw)
        return trim

class NHL(BaseFeature):
    '''
      （创出52周新高的股票 - 52周新低的股票） + 昨天的指标值 10日MA来平滑指标
    '''

    @staticmethod
    def _init_nhl(data):
        pivot_close = data.pivot(columns = 'code',values='close')
        print('pivot_close',pivot_close)
        pivot_close.index = range(len(pivot_close))
        idx_max = pivot_close.idxmax(axis = 0)
        print('idx_max',idx_max)
        idx_min = pivot_close.idxmin(axis = 0)
        print('idx_min',idx_min)
        nhl = (idx_max == len(data)).sum() - (idx_min == 0).sum()
        return nhl

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        nhl = cls._init_nhl(raw)
        return nhl

class Temperature(BaseFeature):
    """
        市值于GDP比率（判断市场是否过热）
    """
    _n_fields = ['mkv','gdp']

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        market_value = raw['mkv'].sum(axis = 1)
        index = {'第一季度':'0330','第二季度':'0630','第三季度':'0930','第四季度':'1230'}
        gdp = raw['gdp']
        gdp.index = [i.replace(index) for i in gdp.index]
        ratio_windowed = market_value.rolling(window).mean() /raw['gdp']
        return ratio_windowed


if __name__ == '__main__':

    quandle = Quandle()
    date = '2019-06-01'
    window = 60
    fields = ['close','volume']
    event = Event(date)
    req = GateReq(event, fields,window)
    feed = quandle.query_ashare_kline(req)
    feed.index = pd.DatetimeIndex(feed.index)

    mw = MarketWidth.calc_feature(feed,5)
    print('marketwidth',mw)

    stix = STIX.calc_feature(feed,5)
    print('stix',stix)

    udr = UDR.calc_feature(feed,5)
    print('udr',udr)
    #
    mo = FeatureMO.calc_feature(feed,(10,5))
    print('mo',mo)
    #
    trim = Trim.calc_feature(feed,5)
    print('trim',trim)
    #
    nhl = NHL.calc_feature(feed,5)
    print('nhl',nhl)
    #
    # temp = Temperature.calc_feature(feed)
    # print('temp',temp)
