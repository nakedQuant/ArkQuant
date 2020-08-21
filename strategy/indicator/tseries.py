#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from statsmodels.tsa.stattools import adfuller, coint, pacf, acf
import numpy as np, pandas as pd
from strategy.indicator import BaseFeature


class ADF(BaseFeature):
    """
        The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative that
        there is no unit root. If the pvalue is above a critical size, then we cannot reject that there is a unit root.

        model args :
            Maximum lag which is included in test, default 12*(nobs/100)^{1/4}
            regression{‘c’,’ct’,’ctt’,’nc’}
            ‘c’ : constant only (default)
            ‘ct’ : constant and trend
            ‘ctt’ : constant, and linear and quadratic trend
            ‘nc’ : no constant, no trend
            autolag{‘AIC’, ‘BIC’, ‘t-stat’, None}
            if None, then maxlag lags are used
            if ‘AIC’ (default) or ‘BIC’, then the number of lags is chosen to minimize the corresponding information criterion
            ‘t-stat’ based choice of maxlag. Starts with maxlag and drops a lag until the t-statistic on the last lag length
            is significant using a 5%-sized test

        序列的稳定性:
            1、价格序列
            2、对数序列
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        adf, p_adf, lag, nobs, critical_dict, ic_best = adfuller(np.array(frame), regression=kwargs['mode'])
        return adf, critical_dict


class Coint(BaseFeature):
    """
        协整检验 --- coint_similar(协整关系)
            1、筛选出相关性的两个标的 ，
            2、判断序列是否平稳 --- 数据差分进行处理
            3、协整模块

        coint返回值三个如下:
            coint_t: float t - statistic of unit - root test on residuals
            pvalue: float MacKinnon's approximate p-value based on MacKinnon (1994)
            crit_value: dict Critical  values for the test statistic at the 1 %, 5 %, and 10 % levels.
        Coint 参数：
            statsmodels.tsa.stattools.coint(y0，y1，trend ='c'，method ='aeg'，maxlag = None，autolag ='aic'，
    """
    @classmethod
    def _calc_feature(cls, feeds, kwargs):
        x, y = feeds
        result = coint(y, x)
        return result[0], result[-1]


class ACF(BaseFeature):
    """
        statsmodels.tsa.stattools.acf(x, unbiased=False, nlags=40, qstat=False, fft=None, alpha=None)
        qstat --- If True, returns the Ljung-Box q statistic for each autocorrelation coefficient.
        qstat ---表示序列之间的相关性是否显著（自回归）
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        lag = kwargs['lag']
        correlation = acf(frame, nlags=lag, fft=kwargs['fft'])
        acf_corrleation = pd.Series(correlation, index=frame.index[: lag + 1])
        return acf_corrleation


class PACF(BaseFeature):
    """
        statsmodels.tsa.stattools.pacf(x, nlags=40, method='ywunbiased', alpha=None)

        return:
            partial :autocorrelations, nlags elements, including lag zero
            confint :array, optional Confidence intervals for the PACF. Returned if confint is not None.
    """
    _n_lags = 10
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        raw = feed.copy()
        n_lags = kwargs['lag']
        coef = pacf(raw, nlags=n_lags)
        pacf_coef = pd.Series(coef, index=frame.index[: n_lags + 1])
        return pacf_coef


class VRT(BaseFeature):
    """
        Lo和Mackinlay(1988)假定，样本区间内的随机游走增量(RW3)的方差为线性。
        若股价的自然对数服从随机游走，则方差比率与收益水平成比例,其方差比率VR期望值为1。
        由于Lo-MacKinlay方差比检验为渐近检验，其统计量的样本分布渐近服从标准正态分布，在有限样本的情况下，其分布常常是有偏的;
        在基础上提出了一种基于秩和符号的非参数方差比检验方法。在样本量相对较小的情况下，而不依赖于大样本渐近极限分布
        方差比检验: 若股价的自然对数服从随机游走，则方差比率与收益水平成比例
        与adf搭配使用，基于adf中的滞后项
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        frame = feed.copy()
        window = kwargs['window']
        adjust_x = pd.Series(np.log(frame))
        var_shift = adjust_x / adjust_x.shift(window)
        var_per = adjust_x / adjust_x.shift(1)
        vrt = var_shift.var() / (window * var_per.var())
        return vrt


class FRAMA(BaseFeature):
    """
        多重分形理论一个重要的应用就是Hurst指数，Hurst指数和相应的时间序列分为3种类型：当H=0.5时，时间序列是随机游走的，序列中不同时间的
        值是随机的和不相关的，即现在不会影响将来；当0≤H≤0.5时，这是一种反持久性的时间序列，常被称为“均值回复”。如果一个序列在前个一时期是
        向上走的，那么它在下一个时期多半是向下走，反之亦然。这种反持久性的强度依赖于H离零有多近，越接近于零，这种时间序列就具有比随机序列更
        强的突变性或易变性；当0.5≤H≤1时，表明序列具有持续性，存在长期记忆性的特征。即前一个时期序列是向上(下)走的，那下一个时期将多半继续
        是向上(下)走的，趋势增强行为的强度或持久性随H接近于1而增加
        R/S(重标极差分析）:
            1、对数并差分，价格序列转化为了对数收益率序列
            2、对数收益率序列等划分为A个子集
            3、计算相对该子集均值的累积离差
            4、计算每个子集内对数收益率序列的波动范围：累积离差最大值和最小值的差值
            5、计算每个子集内对数收益率序列的标准差
            6、用第五步值对第4步值进行标准化
            7、增大长度并重复前六步，得出6的序列
            8、将7步的序列对数与长度的对数进行回归，斜率Hurst指数
    """
    @classmethod
    def _calc_feature(cls, feed, kwargs):
        raise NotImplementedError()


class PCA(BaseFeature):
    '''
    主成分分析法理论：选择原始数据中方差最大的方向，选择与其正交而且方差最大的方向，不断重复这个过程
    pca.fit_transform()
    具体的算法：
    PCA算法：
    1）将原始数据按列组成n行m列矩阵X

    2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值

    3）求出协方差矩阵C=X * XT

    4）求出协方差矩阵的特征值及对应的特征向量

    5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P

    6）Y=PX  即为降维到k维后的数据
    '''
    _topNfeat = 2

    @classmethod
    def _calc_feature(cls, feed, kwargs):
        datamat = feed.copy()
        meanval = np.mean(datamat, axis=0)
        meanremoved = datamat - meanval
        print('mean',meanremoved)
        covmat = np.cov(meanremoved, rowvar=0)
        eigval, eigvect = np.linalg.eig(np.mat(covmat))
        eigvalind = np.argsort(eigval)
        eigvalind = eigvalind[-cls._topNfeat :]
        redeigvect = eigvect[:, eigvalind]
        reconmat = meanremoved * redeigvect * redeigvect.T + meanval
        return reconmat
