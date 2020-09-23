# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from itertools import product
from toolz import keyfilter
import numpy as np
from strategy.indicator.tseries import ADF


class PairWise(object):
    """
        不同ETF之间的配对交易 ；相当于个股来讲更加具有稳定性
        1、价格比率交易（不具备协整关系，但是具有优势）
        2、计算不同ETF的比率的平稳性(不具备协整关系，但是具有优势）
        3、平稳性 --- 协整检验
        4、半衰期 ： -log2/r --- r为序列的相关系数
        单位根检验、协整模型
        pval = ADF.calc_feature(ratio_etf)
        coef = _fit_statsmodel(np.array(raw_y), np.array(raw_x))
        residual = raw_y - raw_x * coef
        acf = ACF.calc_feature(ratio_etf)[0]
        if pval <= 0.05 and acf < 0:
            half = - np.log(2) / acf
        zscore = (nowdays - ratio_etf.mean()) / ratio_etf.std()
    """
    def __init__(self, params):
        self.params = params

    @staticmethod
    def _calculate_deviation(ratio, scale):
        tunnel_upper = np.nanmean(ratio) + scale * np.nanstd(ratio)
        tunnel_bottom = np.nanmean(ratio) - scale * np.nanstd(ratio)
        if ratio[-1] > tunnel_upper:
            return True, 'bottom'
        elif ratio[-1] < tunnel_bottom:
            return True, 'upper'
        return False, None

    def _compute(self, data, mask):
        y, x = data.keys()
        ratio = data[y]['close'] / data[x]['close']
        status, lags = ADF.compute(ratio, self.params)
        excess, direction = self._calculate_deviation(ratio, self.params['scale'])
        if status and excess:
            out = y if direction == 'upper' else x
            return out
        return False

    def compute(self, data, mask):
        product_sets = product(data.keys(), data.keys())
        _mask = []
        for count, tuples in enumerate(product_sets):
            frame = keyfilter(lambda x: x in tuples, data)
            result = self._compute(frame, mask)
            _mask.append(result)
        # filter
        _mask = [sid for sid in _mask if sid]
        return _mask


class AHRatio(object):
    """
        数据处理：对数价差
        获取股票每天A/H股价比率
        1、平衡配对轮动配对交易 - 以回测期间股价对数差分在配对交易的基础上增加了协整判断
        2、标的为ETF50，--- 主要原因 ： ETF50成分股的大市值特征稳定性强由此算出的协整关系不易变动；而中、小市值的股票更加
           容易受到公告等突发事件的影响，大公司的对于突发事件的消化能力比较强而且频率较低（国内许多公募的基金持股）
        3、如果两个对数差序列存在协整检验，通过OLS获取系数，同时计算残差（价差）的z_score如果超过1或者N倍标准差，即买入

        由于AH价差过大，最好通过比率来定义反转， 但是用于竞价机制不一样导致分析结果存在不合理性
    """


class SLTrading(object):
    """
        主要针对于ETF或者其他的自定义指数
        度量动量配对交易策略凸优化(Convex Optimization)
        1、etf 国内可以卖空
        2、构建一个协整关系的组合与etf 进行多空交易
        逻辑：
        1、以ETF50为例，找出成分股中与指数具备有协整关系的成分股
        2、买入具备协整关系的股票集，并卖出ETF50指数
        3、如果考虑到交易成本，微弱的价差刚好覆盖成本，没有利润空间
        筛选etf成分股中与指数具备有协整关系的成分股
        将具备协整关系的成分股组合买入，同时卖出对应ETF
        计算固定周期内对冲收益率，定期去更新_coint_test
    """

