# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from strategy import Strategy
from itertools import product
from toolz import keyfilter
import numpy as np
from strategy.indicator.tseries import ADF


class PairWise(Strategy):
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
