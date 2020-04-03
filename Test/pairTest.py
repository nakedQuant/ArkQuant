# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import numpy as np

from GateWay import Event,GateReq,Quandle
from Algorithm.Feature.Tseries import Coint
from Algorithm.Mathmetics.linear_tool import _fit_statsmodel

quandle = Quandle()

class PairTrading:
    """
        数据处理：对数价差

        获取股票每天A/H股价比率

        1、平衡配对轮动配对交易 - 以回测期间股价对数差分在配对交易的基础上增加了协整判断
        2、标的为ETF50，--- 主要原因 ： ETF50成分股的大市值特征稳定性强由此算出的协整关系不易变动；而中、小市值的股票更加
           容易受到公告等突发事件的影响，大公司的对于突发事件的消化能力比较强而且频率较低（国内许多公募的基金持股）
        3、如果两个对数差序列存在协整检验，通过OLS获取系数，同时计算残差（价差）的z_score如果超过1或者N倍标准差，即买入
    """
    _n_field = ['close']
    _thres_p = 0.05

    def __init__(self, window):
        """
        :param window: 回测周期
        """
        self.window = window

    def _load_raw_arrays(self, dt):
        event = Event(dt,'ETF50')
        req = GateReq(event, self._n_field, self.window)
        etf_array = quandle.query_etf_kline(req)

        event = Event(dt,etf_component)
        req =  GateReq(event, [self._n_field], self.window)
        raw_array = quandle.addBars(req)
        return etf_array,raw_array

    def _pair_coint(self, dt):
        etf,kl_pd = self._load_raw_arrays(dt, 'close')
        raw_x = np.array(etf)
        pvalue_dict = {}
        for col in kl_pd.columns:
            raw_y = kl_pd.loc[:, col]
            pvalue = Coint.calc_feature(raw_y, raw_x)
            if pvalue <= self._thres_p:
                coef = _fit_statsmodel(np.array(raw_y), np.array(raw_x))
                residual = raw_y - raw_x * coef
                pvalue_dict.update({col : {'pvalue':pvalue,'rmean':residual.mean(),'rstd':residual.std()}})
        return pvalue_dict

    def _diverse(self, dt):
        """
            基于pvalue \ zscore 筛选出显著、偏离度最大的ETF成分股
            if pval <= 0.05 and acf < 0 :
                zscore = (nowdays - ratio_etf.mean())/ratio_etf.std()
                half =  - np.log(2) / acf
                return zscore , half
        """
        pass

    def run(self, dt):
        """
            执行标的交易 -- 循坏
        """
        pass


if __name__ == '__main__':

    pair = PairTrading()