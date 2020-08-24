# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from strategy import Strategy


class Proba(Strategy):
    """
        1、序列中基于中位数的性质更加能代表趋势动向
        2、预警指标的出现股票集中，容易出现后期的大黑马，由此推导出异动逻辑以及在持续性
        3、统计套利:(pt > pt-1) / (pt-1 > m) 概率为75% 引发的思考(close - pre_high)
    """
