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
    _n_fields = ['close','open','volume']

    def _init__(self,window):
        self.assets = quandle.query_basics()
        self.window = window

    def _load_raw_arrays(self,dt,asset):
        event = Event(dt,asset)
        req = GateReq(event,self._n_fields,self.window)
        kl = quandle.query_ashare_kline(req)
        return kl

    def _stats_gap(self,dt):
        raw = self._load_raw_arrays(dt)
        power =  (raw['volume'].diff() / raw['volume'].shift(1)) * \
                     (raw['close'] - raw['open']) / (raw['close'].shift(-1) - raw['open'])
        bottom_flag = (raw['close'].shift(1) > raw['open']) & (raw['close'].shift(1) < raw['close'])
        upper_flag = (raw['close'].shift(1) < raw['open']) & (raw['open'] < raw['close'])
        gap_flag = bottom_flag & upper_flag
        accurrence = gap_flag.sum() / len(gap_flag)
        gap_power = power[gap_flag].sum() - power[~gap_flag].sum()
        return gap_power ,accurrence

    def compute(self,dt):
        for asset in self.assets['代码']:
            self._stats_gap(asset,dt)

