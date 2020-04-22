# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import numpy as np
from gateWay import Event,GateReq,Quandle
from algorithm.features.Tseries import ADF,Coint

quandle = Quandle()

class PairWise:
    """
        不同ETF之间的配对交易 ；相当于个股来讲更加具有稳定性
        1、价格比率交易（不具备协整关系，但是具有优势）
        2、计算不同ETF的比率的平稳性(不具备协整关系，但是具有优势）
        3、平稳性 --- 协整检验
        4、半衰期 ： -log2/r --- r为序列的相关系数
    """
    _n_field = ['close']

    def __init__(self,window):
        self.window = window

    def _load_raw_array(self,dt):
        event = Event(dt)
        req = GateReq(event,self._n_field,self.window)
        raw = quandle.query_etf_kline(req)
        kline = raw.pivot(columns = 'code',values = 'close')
        return kline

    def _tseries_test(self,x,y):
        """
            单位根检验、协整模型
            pval = ADF.calc_feature(ratio_etf)
            coef = _fit_statsmodel(np.array(raw_y), np.array(raw_x))
            residual = raw_y - raw_x * coef
            acf = ACF.calc_feature(ratio_etf)[0]
            if pval <= 0.05 and acf < 0:
                half = - np.log(2) / acf
            zscore = (nowdays - ratio_etf.mean()) / ratio_etf.std()
        """
        pass

    def run(self,dt):
        """
            param assets: two etfs code
        """
        etf_kline = self._load_raw_array(dt)
        etf_num = len(etf_kline.columns)
        for x in range(etf_num-1):
            for y in range(x+1,etf_num):
                self._tseries_test(etf_kline.iloc[:,[x,y]])


if __name__ == '__main__':

    pair = PairWise()