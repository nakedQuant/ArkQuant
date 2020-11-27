# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np, pandas as pd


class BaseFeature(ABC):

    # 涉及数据 为前复权数据
    @abstractmethod
    def _calc_feature(self, frame, kwargs):
        raise NotImplementedError()

    def compute(self, feed, kwargs):
        frame = feed.copy()
        if isinstance(frame, pd.DataFrame) and len(frame.columns) > 1:
            out = dict()
            for col in frame.columns:
                out[col] = self._calc_feature(frame[col], kwargs)
        else:
            frame = frame.iloc[:, 0] if isinstance(frame, pd.DataFrame) else frame
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


class ATR(BaseFeature):

    def _calc_feature(self, frame, kwargs):
        pass

    def compute(self, frame, kwargs):
        tr = TR.compute(frame, kwargs)
        window = kwargs['window']
        atr = tr.rolling(window=window).mean()
        return atr


class MA(BaseFeature):

    def _calc_feature(self, frame, kwargs):
        window = kwargs['window']
        ma_windowed = frame.rolling(window=window).mean()
        return ma_windowed


class WS(BaseFeature):
    """
        怀尔德平滑 --- 前期MA + （收盘价 - 前期MA） / 期间数
        --- 不一定为收盘价
    """
    def _calc_feature(self, feed, kwargs):
        window = kwargs['window']
        ma = MA.compute(feed, window)
        # ws = ma + (feed['close'] - ma) / window
        ws = ma + (feed - ma) / window
        return ws


class ExponentialMovingAverage(BaseFeature):
    """
        序列 p1 ,p2 ,p3, p4, p5,（对应系数 a0, a1 ,a2, a3, a4 ,a5) --- pn ; e1 = (1- a1)e0 + a1 * p1 , e2 = (1 - a2)*e1 + a2 * p2 , 进行展开
        e0 = 0.0 --- 初始值假设为0 或者 增加一个可变的增量（而增量的系数为(1- a1)(1- a2)(1- a3)）
         e.g. --- EMA(i) = Price(i) * SC + EMA(i - 1) * (1 - SC),SC = 2 / (n + 1) ― EMA平滑常数
         ---   当系数固化时基于递归方式，但是如果系数是动态变化的需要调整逻辑调整
    """
    @staticmethod
    def _calculate_weights(frame, kwargs):
        raise NotImplementedError()

    @staticmethod
    def _shift_weight(weights):
        p = weights.copy()
        p = 1 - p
        p = list(p)
        p.reverse()
        p.insert(0, 1)
        prod_p = list(np.cumprod(p)[:-1])
        prod_p.reverse()
        decay_rates = np.array(prod_p) * weights
        return decay_rates

    def _calc_feature(self, frame, kwargs):
        recursion = kwargs.get('recursion', 1)
        ema_weights = self._calculate_weights(frame, kwargs)
        exponential_weights = self._shift_weight(ema_weights)
        # print('exponential_weights', exponential_weights)
        window = kwargs['window']
        out = frame
        # dimension = kwargs.get('dimension', 1)
        dimension = 1
        while dimension <= recursion:
            out = [np.average(np.array(out)[:loc][-window:], axis=0, weights=exponential_weights)
                   for loc in range(window, len(out)+1)]
            dimension = dimension + 1
        return out


class EMA(ExponentialMovingAverage, BaseFeature):
    """
        a = 2 / (n + 1) --- 系数序列都是一样的
        reduce(lambda x, y: x * weight + y * (1 - weight), np.array(frame)
        注意点 --- 比如EMA10 ， 初始为10个交易日之前的数据，考虑范围放大为20，这样计算出来的指标精确度会有所提高，更加平滑 --- weight (离现在时点越近权重越高）
    """
    @staticmethod
    def _calculate_weights(frame, kwargs):
        window = kwargs['window']
        rate = 2 / (window - 1)
        return np.full(window, rate, np.float)


class SMA(ExponentialMovingAverage, BaseFeature):
    """
        SMA区别 wgt = (window/2 +1)/window
    """
    @staticmethod
    def _calculate_weights(frame, kwargs):
        window = kwargs['window']
        weight = (window / 2 + 1) / window
        decay_rates = np.full(window, weight, np.float)
        return decay_rates


class AEMA(ExponentialMovingAverage, BaseFeature):
    """
        AEMA changes according to the time 权重为（1-a) ** i 按照时间的推移权重指数变化
    """
    @staticmethod
    def _calculate_weights(frame, kwargs):
        window = kwargs['window']
        decay_rate = 2 / (window + 1)
        ewm_weight = np.full(window, decay_rate, np.float64) ** np.arange(window, 0, -1)
        return ewm_weight


class Decay(ExponentialMovingAverage, BaseFeature):
    """
        half life --- aema
    """
    @staticmethod
    def _calculate_weights(frame, kwargs):
        window = kwargs['window']
        decay_rate = np.exp(np.log(.5) * 2 / window)
        decay_weight = np.full(window, decay_rate, np.float) ** np.arange(window, 0, -1)
        return decay_weight


class Centre(ExponentialMovingAverage, BaseFeature):
    """
        centre dis --- aema
    """
    @staticmethod
    def _calculate_weights(frame, kwargs):
        window = kwargs['window']
        centre_mass = 1.0 - (1.0 / (1.0 + window))
        decay_rate = np.full(window, centre_mass, np.float) ** np.arange(window, 0, -1)
        return decay_rate


class ExponentialWeightedMovingAverage(BaseFeature):

    @staticmethod
    def _calculate_weights(kwargs):
        """
        Build a weight vector for an exponentially-weighted statistic.

        The resulting ndarray is of the form::

            [decay_rate ** length, ..., decay_rate ** 2, decay_rate]

        Parameters
        ----------
        window : int
            The length of the desired weight vector.
        func : to calculate decay_rate  float
            --- half_life, centra_mass , span
            The rate at which entries in the weight vector increase or decrease.
        Returns
        -------
        weights : ndarray[float64]
        """
        window = kwargs['window']
        func = kwargs['func']
        decay_rate = func(window)
        return np.full(window, decay_rate, np.float64) ** np.arange(window, 0, -1)

    def _calc_feature(self, feed, kwargs):
        frame = feed.copy()
        exponential_weights = self._calculate_weights(kwargs)
        print('exponential_weights', exponential_weights)
        window = kwargs['window']
        out = [np.average(frame[:loc][-window:], axis=0, weights=exponential_weights)
               for loc in range(window, len(frame)+1)]
        return out


__all__ = ['VMA',
           'TR',
           'ATR',
           'MA',
           'EMA',
           'AEMA',
           'SMA',
           'BaseFeature',
           'ExponentialWeightedMovingAverage',
           'ExponentialMovingAverage']

# if __name__ == '__main__':
#
#     from gateway.driver.data_portal import DataPortal
#     from gateway.asset.assets import Equity, Convertible, Fund
#
#     asset = Equity('600000')
#     session = '2015-01-01'
#     kw = {'window': 10}
#     portal = DataPortal()
#     # dct = portal.get_window([asset], session, 50, ['open', 'high', 'low', 'close'], 'daily')
#     dct = portal.get_window([asset], session, 50, ['close'], 'daily')
#     feed = dct[asset.sid]
#     print('feed', feed)
#     vma = VMA().compute(feed, kw)
#     print('vma', vma)
#     tr = TR().compute(feed, kw)
#     print('tr', tr)
#     atr = ATR().compute(feed, kw)
#     print('atr', atr)
#     ma = MA().compute(feed, kw)
#     print('ma', ma)
#     ema = EMA().compute(feed, kw)
#     print('ema', ema)
#     sma = SMA().compute(feed, kw)
#     print('sma', sma)
#     aema = AEMA().compute(feed, kw)
#     print('aema', aema)
#     decay = Decay().compute(feed, kw)
#     print('decay', decay)
#     centre = Centre().compute(feed, kw)
#     print('centre', centre)
#     kw.update({'func': lambda x: 2/x})
#     ewma = ExponentialWeightedMovingAverage().compute(feed, kw)
#     print('ewma', ewma)
