# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np, pandas as pd
from itertools import chain
from abc import ABC, abstractmethod


class Simulation(ABC):

    @staticmethod
    @abstractmethod
    def simulate_dist(num, open_pct):
        raise NotImplementedError('distribution of price')

    @staticmethod
    @abstractmethod
    def simulate_ticker(self, num):
        raise NotImplementedError('simulate ticker of trading_day')


class SimpleSimulation(Simulation):

    @staticmethod
    def simulate_dist(num, open_pct):
        """模拟价格分布，以开盘振幅为参数"""
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        if num > 0:
            # 模拟价格分布
            dist = 1 + np.copysign(alpha, np.random.beta(alpha, 100, num))
        else:
            dist = [1 + alpha / 100]
        return dist

    @staticmethod
    def simulate_ticker(num):
        # ticker arranged on sequence
        interval = 4 * 60 / num
        # 按照固定时间去执行
        upper = pd.date_range(start='09:30', end='11:30', freq='%dmin' % interval)
        bottom = pd.date_range(start='13:00', end='14:57', freq='%dmin' % interval)
        # 确保首尾
        tick_intervals = list(chain(*zip(upper, bottom)))[:num - 1]
        tick_intervals.append(pd.Timestamp('2020-06-17 14:57:00', freq='%dmin' % interval))
        return tick_intervals
