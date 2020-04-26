# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""

class Benchmark(object):

    def __init__(self,bar_reader):
        self.bar_reader = bar_reader

    def load_native_index(self, sdate, edate,asset):
        """
            返回特定时间区间日基准指数K线
        """
        index = self.bar_reader._load_raw_arrays(sdate,edate,asset)


    def load_periphera_index(self, sdate, edate,fields,index, exchange):
        """us.DJI 道琼斯 us.IXIC 纳斯达克 us.INX  标普500 hkHSI 香港恒生指数 hkHSCEI 香港国企指数 hkHSCCI 香港红筹指数"""
        raw = self.extra.download_periphera_index(sdate, edate,index, exchange)
        raw.index = raw['trade_dt']
        index_price = raw.loc[:,fields]
        return index_price

    def get_benchmark_returns(self):

        pass

    def get_treasure_returns(self):

        pass