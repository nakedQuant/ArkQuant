#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
import numpy as np
from indicator import (
    BaseFeature,
    EMA,
    MA
)
from gateway.driver.data_portal import DataPortal
from gateway.asset.assets import Equity, Convertible, Fund

ma = MA()
ema = EMA()


class MarketWidth(BaseFeature):
    """
        市场宽度上涨股票数量 /（上涨股票数量 + 下跌股票数量）
    """
    @staticmethod
    def _calc_return(data):
        if len(data):
            ret = data / data.shift(1) - 1
            ratio = len(ret[ret > 0]) / len(ret[ret != 0])
            return ratio

    def _calc_feature(self, frame, kwargs):
        window = kwargs['window']
        close = frame.pivot(columns='sid', values='close')
        ret = close - close.shift(1) - 1
        width = ret.apply(lambda x: sum(x > 0) / len(x), axis=1)
        # T -- minutes S -- second M -- month
        # width_windowed = width.resample('D' % window).mean()
        # width_windowed = width.rolling(window=window).mean()
        # return width_windowed
        return width

    def compute(self, frame, kwargs):
        market_width = self._calc_feature(frame, kwargs)
        return market_width


class STIX(BaseFeature):
    """
        a / d = 上涨股票数量 /（上涨股票数量 + 下跌股票数量） STIX为 a / d的22EMA
    """
    @classmethod
    def _calc_feature(cls, frame, kwargs):
        width = MarketWidth().compute(frame, kwargs)
        stix = ema.compute(width, kwargs)
        return stix

    def compute(self, frame, kwargs):
        market_width = self._calc_feature(frame, kwargs)
        return market_width


class UDR(BaseFeature):
    """
       上涨股票的交易量之和除以下跌股票的交易量之和
    """
    @classmethod
    def _calc_feature(cls, frame, kwargs):
        close = frame.pivot(columns='sid', values='close')
        volume = frame.pivot(columns='sid', values='volume')
        upper_vol = volume[close > close.shift(1)]
        bottom_vol = volume[close < close.shift(1)]
        udr = upper_vol.sum() / bottom_vol.sum()
        return udr

    def compute(self, frame, kwargs):
        market_width = self._calc_feature(frame, kwargs)
        return market_width


class FeatureMO(BaseFeature):
    """
        正常的牛市伴随大量的价格适度上涨；而变弱的牛市少数股票大涨（背离信号）
        （上涨 - 下跌）的10 % EMA - 5 % 的EMA
    """
    @classmethod
    def _calc_feature(cls, frame, kwargs):
        window = kwargs['window']
        close = frame.pivot(columns='sid', values='close')
        accumulate = (close - close.shift(1)).apply(lambda x: sum(x > 0) - sum(x < 0), axis=1)
        mo_slow = np.array(ema.compute(accumulate, {'window': max(window)}))
        mo_fast = np.array(ema.compute(accumulate, {'window': min(window)}))
        mo_ema = mo_slow - mo_fast[-len(mo_slow):]
        return mo_ema

    def compute(self, frame, kwargs):
        market_width = self._calc_feature(frame, kwargs)
        return market_width


class Trim(BaseFeature):
    """
       对阿姆斯指标平滑得出（(上涨股票数量10日加总) /（下跌股票数量10日加总）） / （对应的成交量）
    """
    @staticmethod
    def _calculate_trim(data):
        close = data.pivot(columns='sid', values='close')
        vol = data.pivot(columns='sid', values='volume')
        sign = close > close.shift(1)
        ratio = np.sum(sign) / np.sum(~sign)
        trim = ratio * (vol[~sign].sum()).sum() / (vol[sign].sum()).sum()
        return trim

    @classmethod
    def _calc_feature(cls, frame, kwargs):
        window = kwargs['window']
        trim_windowed = frame.rolling(window=window).apply(cls._calculate_trim)
        return trim_windowed

    def compute(self, frame, kwargs):
        frame = feed.copy()
        market_width = self._calc_feature(frame, kwargs)
        return market_width


class NHL(BaseFeature):
    """
      （创出52周新高的股票 - 52周新低的股票） + 昨天的指标值 10日MA来平滑指标
    """
    @staticmethod
    def _calculate_nhl(data):
        pivot_close = data.pivot(columns='sid', values='close')
        pivot_close.index = range(len(pivot_close))
        idx_max = pivot_close.idxmax(axis=0)
        idx_min = pivot_close.idxmin(axis=0)
        nhl = (idx_max == len(data)).sum() - (idx_min == 0).sum()
        return nhl

    @classmethod
    def _calc_feature(cls, frame, kwargs):
        nhl_series = frame.rolling(window=kwargs['period']).apply(cls._calculate_nhl)
        nhl_windowed = MA.compute(nhl_series, kwargs)
        return nhl_windowed

    def compute(self, frame, kwargs):
        nhl = self._calc_feature(frame, kwargs)
        return nhl


if __name__ == '__main__':

    asset = Equity('600000')
    session = '2015-01-01'
    kw = {'window': 10}
    portal = DataPortal()
    feed = portal.get_stack_value('equity', session, 100, 'daily')
    print('feed', feed)
    # mw = MarketWidth().compute(feed, kw)
    # print('mw', mw)
    # sx = STIX().compute(feed, kw)
    # print('sx', sx)
    # ur = UDR().compute(feed, kw)
    # print('ur', ur)
    # kw = {'window': (5, 10)}
    # fm = FeatureMO().compute(feed, kw)
    # print('fm', fm)
    kw = {'window': 10}
    tm = Trim().compute(feed, kw)
    print('tm', tm)
    nhl = NHL().compute(feed, kw)
    print('nhl', nhl)
