# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import datetime as dt

class Cycle:
    """
        周期性:股票与经济的存在的周期的联动性，表明为滞后或者提前
        逻辑：
        1、按照行业类别进行划分
        2、计算行业每个月的sharp ratio
        3、计算每个月的sharp ratio 平稳性
        4、筛选出平稳性最大的月份作为该行业的周期性代表
    """
    _field = ['close']
    pass
