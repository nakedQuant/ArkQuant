#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
import numpy as np
from strategy.indicator import (
    BaseFeature,
    EMA,
    MA
)


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
        pass

    @classmethod
    def compute(cls, frame, kwargs):
        window = kwargs['window']
        close = frame.pivot(columns='sid', values='close')
        # T -- minutes S -- second M -- month
        rate = close.apply(cls._calc_return, axis=1)
        rate_windowed = rate.resample('%dD' % window).mean()
        rate_windowed.dropna(inplace=True)
        return rate_windowed


class STIX(BaseFeature):
    """
        a / d = 上涨股票数量 /（上涨股票数量 + 下跌股票数量） STIX为 a / d的22EMA
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        window = kwargs['window']
        width = MarketWidth.calc_feature(frame, window)
        stix = EMA.calc_feature(width, {window: 22})
        return stix


class UDR(BaseFeature):
    """
       上涨股票的交易量之和除以下跌股票的交易量之和

    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        close = frame.pivot(columns='code', values='close')
        volume = frame.pivot(columns='code', values='volume')
        upper_vol = volume[close > close.shift(1)]
        bottom_vol = volume[close < close.shift(1)]
        udr = upper_vol.sum() / bottom_vol.sum()
        return udr


class FeatureMO(BaseFeature):
    """
        正常的牛市伴随大量的价格适度上涨；而变弱的牛市少数股票大涨（背离信号）
        （上涨 - 下跌）的10 % EMA - 5 % 的EMA
    """

    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        window = kwargs['window']
        close = frame.pivot(columns='code', values='close')
        accumulate = np.sum(np.sign(close - close.shift(1)))
        mo_slow = EMA.calc_feature(accumulate, {'window': max(window)})
        mo_fast = EMA.calc_feature(accumulate, {'window': min(window)})
        # 对齐
        mo_ema = mo_slow - mo_fast
        return mo_ema


class Trim(BaseFeature):
    """
       对阿姆斯指标平滑得出（(上涨股票数量10日加总) /（下跌股票数量10日加总）） / （对应的成交量）
    """
    @staticmethod
    def _init_trim(data):
        close_pivot = data.pivot(columns='code', values='close')
        vol_pivot = data.pivot(columns='code', values='volume')
        sign = close_pivot > close_pivot.shift(1)
        ratio = np.sum(sign) / np.sum(~sign)
        trim = ratio * vol_pivot[~sign].sum() / vol_pivot[sign].sum()
        return trim

    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        window = kwargs['window']
        trim_windowed = frame.rolling(window=window).apply(cls._init_trim)
        return trim_windowed


class NHL(BaseFeature):
    """
      （创出52周新高的股票 - 52周新低的股票） + 昨天的指标值 10日MA来平滑指标
    """
    nhl_window = 10

    @staticmethod
    def _init_nhl(data):
        pivot_close = data.pivot(columns='code', values='close')
        print('pivot_close', pivot_close)
        pivot_close.index = range(len(pivot_close))
        idx_max = pivot_close.idxmax(axis=0)
        print('idx_max', idx_max)
        idx_min = pivot_close.idxmin(axis=0)
        print('idx_min', idx_min)
        nhl = (idx_max == len(data)).sum() - (idx_min == 0).sum()
        return nhl

    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        window = kwargs['window']
        nhl_series = frame.rolling(window=window).apply(cls._init_nhl)
        nhl_windowed = MA.compute(nhl_series, {'window': cls.nhl_window})
        return nhl_windowed
