# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


class Discretizer(object):
    """
        encode
        --- 股价涨跌的波动的关联性
        encoding algorithm to classify the dataset and conf ,dense method is not avaivable for classify the stock ;
        ordinal method ---  encoded as an integer value suits for seek the pattern of stock price change
        原理：股票波动剔除首日以及科创板 ，波动范围：-10%至10% ，将其划分为N档，计算符号相关性
        e.g : 2,3,4,6,7,9
              4,6,8,3,10,4
        序列之差 --- 1，1，2，1，2
                    2，2，-5，7，-6
        转化为sign : 1,1,1,1,1
                    1,1,-1,1,-1
        计算相关性 : 序列相乘之和除以序列长度
    """