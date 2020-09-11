# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from functools import reduce
import numpy as np, pandas as pd


class BaseFeature(ABC):

    # 涉及数据 为前复权数据
    @abstractmethod
    def _calc_feature(self, frame, kwargs):
        raise NotImplementedError()

    def compute(self, feed, kwargs):
        frame = feed.copy()
        if isinstance(feed, pd.Series):
            out = dict()
            for col in frame.columns():
                out[col] = self._calc_feature(frame, kwargs)
        else:
            out = self._calc_feature(frame, kwargs)
        return out


class VMA(BaseFeature):
    """
        HF=（开盘价+收盘价+最高价+最低价）/4
        VMA指标比一般平均线的敏感度更高
    """
    def _calc_feature(self, frame, kwargs):
        pass

    @classmethod
    def compute(cls, feed, kwargs):
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
    def _calc_feature(self, frame, kwargs):
        pass

    @classmethod
    def compute(cls, feed, kwargs):
        frame = feed.copy()
        frame['pre_close'] = frame['close'].shift(1)
        tr = frame.apply(lambda x: max(abs(x['high'] - x['low']),
                                       abs(x['high'] - x['pre_close']),
                                       abs(x['pre_close'] - x['low'])), axis=1)
        return tr


class Atr(BaseFeature):

    def _calc_feature(self, frame, kwargs):
        window = kwargs['window']
        tr = TR.compute(frame)
        atr = tr.rolling(window=window).mean()
        return atr


class MA(BaseFeature):

    def _calc_feature(self, frame, kwargs):
        window = kwargs['window']
        ma_windowed = frame.rolling(window=window).mean()
        return ma_windowed


class ExponentialMovingAverage(BaseFeature):
    """
        序列 p1 ,p2 ,p3, p4, p5,（对应系数 a0, a1 ,a2, a3, a4 ,a5) --- pn ; e1 = (1- a1)e0 + a1 * p1 , e2 = (1 - a2)*e1 + a2 * p2 , 进行展开
        e0 = 0.0 --- 初始值假设为0 或者 增加一个可变的增量（而增量的系数为(1- a1)(1- a2)(1- a3)）
    """
    @staticmethod
    def _calculate_weights(window):
        rate = 2 / (window + 1)
        return np.full(window, rate, np.float)

    def adjust_weights(self, window):
        weights = self._calculate_weights(window)
        p = weights.copy()
        p = 1 - p
        p.reverse()
        p.insert(0, 1)
        prod_p = np.cumprod(p)[:-1]
        prod_p.reverse()
        decay_rates = prod_p * weights
        return decay_rates

    def _calc_feature(self, frame, kwargs):
        array = np.array(frame)
        window = kwargs['window']
        exponential_weights = self.adjust_weights()
        out = [np.average(array[:loc][-window:], axis=0, weights=exponential_weights)
               for loc in range(window, len(array))]
        return out


class EMA(ExponentialMovingAverage, BaseFeature):
    """
        a = 2 / (n + 1) --- 系数序列都是一样的
        reduce(lambda x, y: x * weight + y * (1 - weight), np.array(frame)
        注意点 --- 比如EMA10 ， 初始为10个交易日之前的数据，考虑范围放大为20，这样计算出来的指标精确度会有所提高，更加平滑 --- weight (离现在时点越近权重越高）
    """
    @staticmethod
    def _calculate_weights(window):
        rate = 2 / (window - 1)
        return np.full(window, rate, np.float)


class AEMA(ExponentialMovingAverage, BaseFeature):
    """
        AEMA changes according to the time 权重为（1-a) ** i 按照时间的推移权重指数变化
    """
    @staticmethod
    def _calculate_weights(window):
        decay_rate = 2 / (window + 1)
        ewm_weight = np.full(window, decay_rate, np.float64) ** np.arange(window, 0, -1)
        return ewm_weight


class SMA(ExponentialMovingAverage, BaseFeature):
    """
        SMA区别 wgt = (window/2 +1)/window
    """
    @staticmethod
    def _calculate_weights(window):
        weight = (window / 2 + 1) / window
        decay_rates = np.full(window, weight, np.float)
        return decay_rates


class Decay(ExponentialMovingAverage, BaseFeature):
    """
        half life --- aema
    """
    @staticmethod
    def _calculate_weights(window):
        decay_rate = np.exp(np.log(.5) * 2 / window)
        decay_weight = np.full(window, decay_rate, np.float) ** np.arange(window, 0, -1)
        return decay_weight


class Centre(ExponentialMovingAverage, BaseFeature):
    """
        centre dis --- aema
    """
    @staticmethod
    def calculate_weight(window):
        centre_mass = 1.0 - (1.0 / (1.0 + window))
        decay_rate = np.full(window, centre_mass, np.float) ** np.arange(window, 0, -1)
        return decay_rate


class NEMA(EMA):
    """
        XEMA --- multi（X dimension) ema ,when dimension == 1  xema is EMA
        event : asset,trading_dt,field
    """
    def _recursion(self, frame, kwargs, record=0):
        feed = self.__calc_feature(frame, kwargs)
        record = record + 1
        if record >= kwargs['dimension']:
            return feed
        self._recursion(feed, kwargs)

    def compute(self, feed, kwargs):
        out = self._recursion(feed, kwargs)
        return out




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
        out = [np.average(frame[-window: loc,], axis=0, weights=exponential_weights)
               for loc in range(window, len(frame))]
        return out
