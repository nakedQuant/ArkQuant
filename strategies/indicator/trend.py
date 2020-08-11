# -*- coding : utf-8 -*-

import pandas as pd ,numpy as np
from functools import partial

from gateWay import Event,GateReq,Quandle
from strategies.features import BaseFeature,EMA,remove_na
from algorithm import zoom,coef2deg
from utils.linear_tool import _fit_poly

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
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        medianFilter = raw.rolling(window = window).median()
        return medianFilter


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

    def _init_mmedian(self,data):
        if isinstance(data,(pd.Series,pd.DataFrame)):
            data.sort_values(ascending = True,inplace=True)
        elif isinstance(data,(list,tuple)):
            data.sort(reversed = False)
        mmedian = data[1:-1].mean()
        return mmedian

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        instance = cls()
        mmedianFilter = raw.rolling(window = window).apply(instance._init_mmedian)
        return mmedianFilter

class AmplitudeFilter(BaseFeature):
    """
        限幅波动 , 参考3Q法则
    """

    @classmethod
    def calc_feature(cls,feed):
        raw = feed.copy()
        upper = 3 * raw.std() + raw.mean()
        bottom = - 3 * raw.std() + raw.mean()
        filter_upper = [upper if x > upper else x  for x in raw]
        filter = [bottom if x < bottom else x for x in filter_upper]
        filter = pd.Series(filter,index = raw.index)
        return filter


class GaussianFilter(BaseFeature):

    """ 高斯滤波器
        低通滤波器，高斯平滑比简单平滑要好
        M为元素个数 ；std为高斯分布的标准差
        guassian = scipy.signal.guassian(M=11, std=2)
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
    def _init_guassian(x,theta = None):
         guassian = np.exp( - x ** 2 /(2 * theta ** 2)) / (np.sqrt(2 * np.math.pi) * theta)
         return guassian

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        frozen  =  partial(cls._init_guassian,theta = raw.std())
        guassian = [frozen(x) for x in np.array(raw)]
        guassian_windowed = pd.Series(guassian,index = raw.index)
        return guassian_windowed

class Detrend(BaseFeature):
    """
        剔除趋势,基于高次方拟合,短期的degree =0 ，默认
    """
    _degree = 0

    @classmethod
    def calc_feature(cls,feed):
        raw = feed.copy()
        print('raw',raw)
        _coef = _fit_poly(range(1,len(raw) + 1),raw,degree = cls._degree)
        print('coef',_coef)
        detrend = raw - _coef * np.array(range(1,len(raw) +1))
        return detrend

class RegRatio(BaseFeature):

    _n_fields = ['high', 'close', 'low']

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        upper = raw['high'].rolling(window = window).mean()
        bottom = raw['low'].rolling(window = window).mean()
        close = raw['close'].rolling(window = window).mean()
        upper_deg = coef2deg(_fit_poly(zoom(upper),0))
        print('upper',upper_deg)
        bottom_deg = coef2deg(_fit_poly(zoom(bottom),0))
        print('bottom',bottom_deg)
        close_deg = coef2deg(_fit_poly(zoom(close),0))
        print('close_deg',close_deg)
        deg_ratio = (upper_deg - close_deg)/(upper_deg - bottom_deg)
        return deg_ratio

class Resistence(BaseFeature):
    '''
        投射带：指定期限内最低价、最高价向前投射（线性回归趋势平行）上曲线：Maxhigh + (i - 1) * 上通道的N期斜率下曲线
        :Max low + (i - 1) * 下通道N期斜率
        理论：通道具有延展性，在基本面没有剧烈变化，相连的相对较短的时间窗口的斜率具有延伸性
        实际应用：确定固定窗口，以某一个时间点向前推两个2个时间窗口，基于第一个窗口的计算投射带同时分析第二个时间窗口的时间序列分布在
        投射带的位置分布，如果处于接近通道边界，说明反弹的概率大
        优化：最好基于EMA指标计算投射带，削减了误差 ；存在阈值判断反弹的可能性
        返回 距离下通道的位置，是否存在反转可能性
        分析 ：定义一个上涨的趋势：不断上移的支撑线；下跌的趋势就是不断下降的支撑线 ，通常滞后买入或者卖出但是作为丧失早期机会的补偿

    '''
    _n_fields = ['high','low','close']

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        ema_h = EMA.calc_feature(raw['high'],window)
        ema_l = EMA.calc_feature(raw['low'],window)
        ema_h_coef = _fit_poly(zoom(ema_h),0)
        ema_l_coef = _fit_poly(zoom(ema_l),0)
        diverse = raw['high'].max() - raw['low'].min()
        tunnel_upper = raw['high'].max() + ema_h_coef * diverse
        tunnel_bottom = raw['low'].min() +  ema_l_coef * diverse
        return tunnel_upper,tunnel_bottom

class Golden(BaseFeature):
    '''
        Fibonacci黄金分割点golden ： 0.191 0.382 0.5 0.618 0.809 1 1.382可视化技术线黄金分割
        stats.mstats.scoreatpercentile(y, 38.2) 61.8 50.0
        回落的水平(retrace)，可以结合Fibonacci
        场景：筛选出前期收益率靠前的股票，分析回落水平水平处于黄金分割位置 ，确定反弹位置
    '''
    _fibonaci = np.array([0.191,0.382,0.5,0.618,0.809])

    @classmethod
    def calc_feature(cls,feed):
        raw = feed.copy()
        retrace = (1 - raw.min()) / raw.max()
        resistence = (1 - cls._fibonaci[cls._fibonaci > abs(retrace)][0]) * raw.max()
        return resistence

class EMD(BaseFeature):
    """
        empirical mode decomposition 借鉴谐波基函数与小波基函数基础上，不断的分离高频数据，最终得到频率近似于为0的
        原理：分解为有限个本征模函数---IMF，不同时间尺度的局部特征信号
    """
    pass


if __name__ == '__main__':

    date = '2019-06-01'
    edate = '2020-02-20'
    asset = '000001'
    window = 100

    fields = ['open','close','high','low','volume']
    event = Event(date,asset)
    req = GateReq(event,fields,window)
    quandle = Quandle()
    feed = quandle.query_ashare_kline(req)
    feed.index = pd.DatetimeIndex(feed.index)

    median_filter = MedianFilter.calc_feature(feed,window)
    print('meidan_filter',median_filter)

    mmedian_filter = MMedianFilter.calc_feature(feed,window)
    print('mmedian_filter',mmedian_filter)

    amplitude = AmplitudeFilter.calc_feature(feed['close'])
    print('amplitude',amplitude)

    gaussian = GaussianFilter.calc_feature(feed['close'],window)
    print('gaussian',gaussian)

    reg = RegRatio.calc_feature(feed,5)
    print('reg',reg)

    u,b = Resistence.calc_feature(feed,5)
    print('resistence',u,b)

    golden = Golden.calc_feature(feed['close'])
    print('golden',golden)

    # emd  = EMD.calc_feature(feed,window)
    # print('emd',emd)