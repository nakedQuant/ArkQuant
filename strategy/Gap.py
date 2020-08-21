# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


class GapTrading:
    """
        gap : 低开上破昨日收盘价（preclose < open and close > preclose）
              高开高走 (open > preclose and close > open9
        gap power :delta vol * (close - open) / (preclose - open)
        逻辑:
        1、统计出现次数
        2、计算跳空能量
    """

