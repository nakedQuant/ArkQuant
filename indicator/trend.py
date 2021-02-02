#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
import pandas as pd, numpy as np
from functools import partial
from indicator import (
    BaseFeature,
    EMA
)
from util.mathmatics import zoom, coef2deg, _fit_poly

# init
ema = EMA()


class MedianFilter(BaseFeature):
    """
        中位值滤波法
        连续采样N次（N取奇数）
        把N次采样值按大小排列
        取中间值为本次有效值
        优点： 能有效克服因偶然因素引起的波动干扰对变化缓慢的被测参数有良好的滤波效果
        缺点：对流量、速度等快速变化的参数不宜
    """
    @classmethod
    def _calc_feature(cls, frame, kwargs):
        median = frame.rolling(window=kwargs['window']).median()
        return median


class MMedianFilter(BaseFeature):
    """
        中位值平均滤波法:
        相当于“中位值滤波法”+“算术平均滤波法”
        连续采样N个数据，去掉一个最大值和一个最小值
        然后计算N - 2个数据的算术平均值
        N值的选取：3~14
        优点：融合了两种滤波法的优点
        对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差
        缺点：测量速度较慢，和算术平均滤波法一样
    """
    @staticmethod
    def _calc_mmedian(data):
        data.sort_values(ascending=True, inplace=True)
        mmedian = data[1:-1].mean()
        return mmedian

    def _calc_feature(self, frame, kwargs):
        window = kwargs['window']
        mmedian = frame.rolling(window=window).apply(self._calc_mmedian)
        return mmedian


class AmplitudeFilter(BaseFeature):
    """
        限幅波动 , 参考3Q法则
    """
    def _calc_feature(self, frame, kwargs):
        upper = 3 * np.nanstd(frame) + np.nanmean(frame)
        print('upper', upper)
        bottom = - 3 * np.nanstd(frame) + np.nanmean(frame)
        print('bottom', bottom)
        tunnel = np.clip(np.array(frame.values), bottom, upper)
        print('tunnel', tunnel)
        df = pd.Series(tunnel, index=frame.index)
        return df


class GaussianFilter(BaseFeature):

    """ 高斯滤波器
        低通滤波器，高斯平滑比简单平滑要好
        M为元素个数 ；std为高斯分布的标准差
        guassian = scipy.strat.guassian(M=11, std=2)
        guassian / = sum(guassian)
        gaussian_process module:
            Squared exponential correlation model (Radial Basis Function).
            (Infinitely differentiable stochastic process, very smooth)::
                                                  n
                theta, d --> r(theta, d) = exp(  sum  - theta_i * (d_i)^2 )
                                                i = 1
            Parameters
            ----------
            theta : array_like
                An array with shape 1 (isotropic) or n (anisotropic) giving the
                autocorrelation parameter(s).
            d : array_like
                An array with shape (n_eval, n_features) giving the componentwise
                distances between locations x and x' at which the correlation model
                should be evaluated.
            Returns
            -------
            r : array_like
                An array with shape (n_eval, ) containing the values of the
                autocorrelation model.
    """

    @staticmethod
    def _calculate_guassian(x, theta):
        guassian = np.exp(- x ** 2 / (2 * theta ** 2)) / (np.sqrt(2 * np.math.pi) * theta)
        return guassian

    def _calc_feature(self, frame, kwargs):
        func = partial(self._calculate_guassian, theta=frame.std())
        guassian = pd.Series([func(x) for x in np.array(frame)], index=frame.index)
        return guassian


class Detrend(BaseFeature):
    """
        剔除趋势,基于高次方拟合,短期的degree =0 ，默认
    """
    def _calc_feature(self, frame, kwargs):
        _coef = _fit_poly(frame, kwargs['degree'])
        detrend = frame - _coef * np.array(range(1, len(frame) + 1))
        return detrend


class RegRatio(BaseFeature):

    @classmethod
    def _calc_feature(cls, frame, kwargs):
        window = kwargs['window']
        degree = kwargs['degree']
        upper = frame['high'].rolling(window=window).mean()
        bottom = frame['low'].rolling(window=window).mean()
        close = frame['close'].rolling(window=window).mean()
        upper_deg = coef2deg(_fit_poly(zoom(upper), 0))
        bottom_deg = coef2deg(_fit_poly(zoom(bottom), 0))
        close_deg = coef2deg(_fit_poly(zoom(close), 0))
        deg_ratio = (upper_deg - close_deg)/(upper_deg - bottom_deg)
        return deg_ratio

    def compute(self, feed, kwargs):
        frame = feed.copy()
        reg = self._calc_feature(frame, kwargs)
        return reg


class Resistence(BaseFeature):
    """
        投射带：指定期限内最低价、最高价向前投射（线性回归趋势平行）上曲线：Maxhigh + (i - 1) * 上通道的N期斜率下曲线
        :Max low + (i - 1) * 下通道N期斜率
        理论：通道具有延展性，在基本面没有剧烈变化，相连的相对较短的时间窗口的斜率具有延伸性
        实际应用：确定固定窗口，以某一个时间点向前推两个2个时间窗口，基于第一个窗口的计算投射带同时分析第二个时间窗口的时间序列分布在
        投射带的位置分布，如果处于接近通道边界，说明反弹的概率大
        优化：最好基于EMA指标计算投射带，削减了误差 ；存在阈值判断反弹的可能性
        返回 距离下通道的位置，是否存在反转可能性
        分析 ：定义一个上涨的趋势：不断上移的支撑线；下跌的趋势就是不断下降的支撑线 ，通常滞后买入或者卖出但是作为丧失早期机会的补偿
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        ema_h = ema.compute(frame['high'], kwargs)
        ema_l = ema.compute(frame['low'], kwargs)
        ema_h_coef = _fit_poly(zoom(ema_h), 0)
        ema_l_coef = _fit_poly(zoom(ema_l), 0)
        diverse = frame['high'].max() - frame['low'].min()
        tunnel_upper = frame['high'].max() + ema_h_coef * diverse
        tunnel_bottom = frame['low'].min() + ema_l_coef * diverse
        return tunnel_upper, tunnel_bottom

    def compute(self, feed, kwargs):
        frame = feed.copy()
        resistence = self._calc_feature(frame, kwargs)
        return resistence


class Golden(BaseFeature):
    '''
        Fibonacci黄金分割点golden ： 0.191 0.382 0.5 0.618 0.809 1 1.382可视化技术线黄金分割
        stats.mstats.scoreatpercentile(y, 38.2) 61.8 50.0
        回落的水平(retrace)，可以结合Fibonacci
        场景：筛选出前期收益率靠前的股票，分析回落水平水平处于黄金分割位置 ，确定反弹位置
    '''
    _fibonaci = frozenset([0.191, 0.382, 0.5, 0.618, 0.809])

    def _calc_feature(self, feed, kwargs):
        frame = feed.copy()
        retrace = (1 - frame.min()) / frame.max()
        fib = np.array(list(self._fibonaci))
        resistence = (1 - fib[fib > abs(retrace)][0]) * frame.max()
        return resistence


class EMD(BaseFeature):
    """
        empirical mode decomposition 借鉴谐波基函数与小波基函数基础上，不断的分离高频数据，最终得到频率近似于为0的
        原理：分解为有限个本征模函数---IMF，不同时间尺度的局部特征信号
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        raise NotImplementedError()


__all__ = ['MedianFilter',
           'MMedianFilter',
           'AmplitudeFilter',
           'GaussianFilter',
           'Detrend',
           'RegRatio',
           'Resistence',
           'Golden']

# if __name__ == '__main__':

#     from gateway.driver.data_portal import DataPortal
#     from gateway.asset.assets import Equity, Convertible, Fund
#
#     asset = Equity('600000')
#     session = '2015-01-01'
#     kw = {'window': 10, 'degree': 1}
#     portal = DataPortal()
#     dct = portal.get_window([asset], session, 100, ['open', 'high', 'low', 'close'], 'daily')
#     feed = dct[asset.sid]
#     print('feed', feed)
#     median = MedianFilter().compute(feed, kw)
#     print('median', median)
#     mmedian = MMedianFilter().compute(feed, kw)
#     print('mmedian', mmedian)
#     amplitude = AmplitudeFilter().compute(feed, kw)
#     print('amplitude', amplitude)
#     guassian = GaussianFilter().compute(feed, kw)
#     print('guassian', guassian)
#     detrend = Detrend().compute(feed, kw)
#     print('detrend', detrend)
#     reg = RegRatio().compute(feed, kw)
#     print('reg', reg)
#     resistence = Resistence().compute(feed, kw)
#     print('resistence', resistence)
#     golden = Golden().compute(feed, kw)
#     print('golden', golden)
#     emd = EMD().compute(feed, kw)
#     print('emd', emd)
