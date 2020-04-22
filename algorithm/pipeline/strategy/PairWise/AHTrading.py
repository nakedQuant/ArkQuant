# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import numpy as np

from gateWay import Event,GateReq,Quandle

quandle = Quandle()

class AHRatio:
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
        event = Event(dt)
        req = GateReq(event, self._n_field, self.window)
        raw = quandle.query_hk_kline(req)
        #ah对应关系
        corr = raw.loc[:,['h_code','code']]
        corr.drop_duplicates(inplace = True)
        #获取H股kline
        h_kline = raw.pivot(columns = 'h_code',values = 'close')
        #获取A股kline
        code = np.unique(raw['code'])
        event = Event(dt,code)
        req = GateReq(event, self._n_field, self.window)
        a_kline = quandle.query_ashare_kline(req)
        return h_kline,a_kline,corr.values

    def _pair_coint(self, dt):
        hk_pd,sz_pd,ah = self._load_raw_arrays(dt)
        for h,a in ah:
            self.exec_pair(hk_pd[h],sz_pd[a])

    def exec_pair(self, dt):
        """
            获取股票每天A/H股价比率
        """
        pass

    def run(self, dt):
        """
            执行标的交易 -- 循坏
        """
        pass


if __name__ == '__main__':

    pair = AHRatio()