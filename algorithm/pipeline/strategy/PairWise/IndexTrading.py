# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

from gateWay import Event,GateReq,Quandle

qundle = Quandle()

class SLTrading:
    """
        主要针对于ETF或者其他的自定义指数
        度量动量配对交易策略凸优化(Convex Optimization)
        1、etf 国内可以卖空
        2、构建一个协整关系的组合与etf 进行多空交易
        逻辑：
        1、以ETF50为例，找出成分股中与指数具备有协整关系的成分股
        2、买入具备协整关系的股票集，并卖出ETF50指数
        3、如果考虑到交易成本，微弱的价差刚好覆盖成本，没有利润空间
    """
    def _load_array(self):
        """
            etf prices
        """
        pass

    def _load_etf_component(self,dt):
        """
            obtain component asset of etf
        """
        pass

    def _coint_test(self):
        """
            筛选etf成分股中与指数具备有协整关系的成分股
        """
        pass

    def _ls_construct(self):
        """
            将具备协整关系的成分股组合买入，同时卖出对应ETF
        """
        pass

    def run(self):
        """
            计算固定周期内对冲收益率，定期去更新_coint_test
        """
        pass


if __name__ == '__main__':

    sl = SLTrading()