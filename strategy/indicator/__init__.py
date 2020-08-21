# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from functools import reduce
import numpy as np


class BaseFeature(ABC):

    # indicator --- terms --- pipe

    @abstractmethod
    def _calc_feature(self, feed, kwargs):
        raise NotImplementedError()

    def compute(self, feed, kwargs):
        out = self._calc_feature(feed, kwargs)
        return out


class VMA(BaseFeature):
    """
        HF=（开盘价+收盘价+最高价+最低价）/4
        VMA指标比一般平均线的敏感度更高
    """
    def _calc_feature(self, feed, kwargs):
        window = kwargs['window']
        vma = (feed['open'] + feed['high'] + feed['low'] + feed['close'])/4
        vma_windowed = vma.rolling(window=window).mean()
        return vma_windowed


class TR(BaseFeature):
    """
        ATR又称Average true range平均真实波动范围，简称ATR指标，是由J.Welles Wilder发明的，ATR指标主要是用来衡量市场波动的强烈度，
        即为了显示市场变化率
        计算方法：
        1.TR =∣最高价 - 最低价∣，∣最高价 - 昨收∣，∣昨收 - 最低价∣中的最大值(np.abs(preclose.shift(1))
        2.真实波幅（ATR）= MA(TR, N)（TR的N日简单移动平均）
        3.常用参数N设置为14日或者21日
        4、atr min或者atr max真实价格变动high - low, low - prelow
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        frame['pre_close'] = frame['close'].shift(1)
        tr = frame.apply(lambda x: max(abs(x['high'] - x['low']), abs(x['high'] - x['pre_close']),
                         abs(x['pre_close'] - x['low'])), axis=1)
        return tr


class Atr(BaseFeature):

    def _calc_feature(self, feed, kwargs):
        window = kwargs['window']
        tr = TR.calc_feature(feed)
        atr = tr.rolling(window=window).mean()
        return atr


class MA(BaseFeature):

    def _calc_feature(self, feed, kwargs):
        window = kwargs['window']
        frame = feed.copy()
        ma_windowed = frame.rolling(window=window).mean()
        return ma_windowed


class EMA(BaseFeature):
    """
        EMA,EXPMA 1 / a = 1 + (1 - a) + (1 - a) ** 2 + (1 - a) ** n ，基于等比数列，当N趋向于无穷大的 ，一般a = 2 / (n + 1)
        比如EMA10 ， 初始为10个交易日之前的数据，考虑范围放大为20，这样计算出来的指标精确度会有所提高，更加平滑
        --- weight (离现在时点越近权重越高）
    """
    @staticmethod
    def calculate_weight(window):
        weight = 2 / (window + 1)
        assert 0.0 < weight <= 1.0
        return weight

    def _calc_feature(self, feed, kwargs):
        window = kwargs['window']
        frame = feed.copy()
        frame.fillna(method='bfill', inplace=True)
        weight = self.calculate_weight(window)
        ema = [reduce(lambda x, y: x * weight + y * (1 - weight), np.array(frame[loc-window:loc, :]))
               for loc in range(window, len(frame))]
        return ema


class SMA(EMA):
    '''
        SMA区别 wgt = (window/2 +1)/window
    '''
    @staticmethod
    def calculate_weight(window):
        weight = (window / 2 + 1) / window
        return weight


class XEMA(EMA):
    """
        XEMA --- multi（X dimension) ema ,when dimension == 1  xema is EMA
        event : asset,trading_dt,field
    """
    def _recursion(self, feed, kwargs, record=0):
        feed = self.__calc_feature(feed, kwargs)
        record = record + 1
        if record >= kwargs['dimension']:
            return feed
        self._recursion(feed, kwargs)

    def compute(self, feed, kwargs):
        out = self._recursion(feed, kwargs)
        return out


class AEMA(BaseFeature):
    """
        AEMA is distinct from ema where weight is not the same and changes according to the time
        wgt : List e.g. 1- wgt , wgt ,but wgt is changeable ,权重为（1-a) ** i 按照时间的推移权重指数变化
        raw : list or array
        window --- 不断滑动window去基于可变系数去计算aema值 ,离现在越近权重越大
    """
    @staticmethod
    def calculate_weight(window):
        # ewm_weight = [(1 - 2 / (window + 1)) ** n for n in range(1, len(window))]
        # ewm_weight = [(2 / (window - 1)) ** n for n in np.arange(len(window)-1, 0, -1)]
        per = 2 / (window + 1)
        assert 0.0 < per <= 1.0
        ewm_weight = [per ** n for n in range(1, len(window)-1)]
        return ewm_weight

    def align_wgt(self, window):
        weights = self.calculate_weight(window)
        p = weights.copy()
        p.insert(0, 0)
        # 计算可变系数序列
        p2 = weights.copy()
        p2.append(1)
        p2.reverse()
        wgt = np.cumprod(p2) * (1 - p)
        return wgt

    def _calc_feature(self, feed, kwargs):
        frame = feed.copy()
        window = kwargs['window']
        # weight = cls.calculate_weight(window)
        weight = self.align_wgt(window)
        aema = [frame[loc-window:loc, :].apply(lambda x: np.array(x) * weight, axis=0)
                for loc in range(window, len(frame))]
        return aema


class Decay(AEMA):
    """
        half life --- aema
    """

    @staticmethod
    def calculate_weight(window):
        decay_rate = np.exp(np.log(.5) * 2 / window)
        # decay_weight = np.full(window - 1, decay_rate, np.float) ** np.arange(window - 1, 0, -1)
        decay_weight = np.full(window - 1, decay_rate, np.float) ** np.arange(1, window - 1, 1)
        return decay_weight


class Centre(AEMA):
    """
        centre dis --- aema
    """

    @staticmethod
    def calculate_weight(window):
        centre_mass = 1.0 - (1.0 / (1.0 + window))
        decay_rate = [centre_mass ** n for n in range(1, window-1)]
        return decay_rate


class ExponentialWeightedMovingAverage(BaseFeature):

    @staticmethod
    def calculate_weight(window, func):
        """
        Build a weight vector for an exponentially-weighted statistic.

        The resulting ndarray is of the form::

            [decay_rate ** length, ..., decay_rate ** 2, decay_rate]

        Parameters
        ----------
        length : int
            The length of the desired weight vector.
        decay_rate : float --- half_life, centra_mass , span
            The rate at which entries in the weight vector increase or decrease.

        Returns
        -------
        weights : ndarray[float64]
        """
        decay_rate = func(window)
        return np.full(window, decay_rate, np.float64) ** np.arange(window, 0, -1)

    def _calc_feature(self, feed, kwargs):
        frame = feed.copy()
        window = kwargs['window']
        func = kwargs['func']
        exponential_weights = self.calculate_weight(window, func)
        out = np.average(
            frame,
            axis=0,
            weights=exponential_weights,
        )
        return out
