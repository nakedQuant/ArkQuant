# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

"""
    Dividend.com is a financial services website focused on providing comprehensive dividend stock research information. 
    The company uses their proprietary DARS™, or Dividend Advantage Rating System, to rank nearly 1,600 dividend-paying
    stocks across five distinct criteria: relative strength, overall yield attractiveness, dividend reliability,
    dividend uptrend, and earnings growth.

    Automatic Trendline Detection
    Support and Resistance Visualizations
    Automatic Fibonacci Retracements
    Manual Trendline Tuning Mode
    Automated Candlestick Pattern Detection
    Automated Price Gap Detection
    
    trade via volatity

    波动率 --- 并非常数，均值回复，聚集，存在长期记忆性的
    大收益率会发生的相对频繁 --- 存在后续的波动
    在大多数市场中，波动率与收益率呈现负相关，在股票市场中的最为明显
    波动率和成交量之间存在很强的正相关性
    波动率分布接近正太分布
    
    基于大类过滤
    标的 --- 按照市值排明top 10% 标的标的集合 --- 度量周期的联动性
    a 计算每个月的月度收益率，筛选出10%集合 / 12的个数，
    b 获取每个月的集合 --- 作为当月的强周期集合
    c 基于技术指标等技术获取对应的标的
"""
