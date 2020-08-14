# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from multiprocessing import Pool
import numpy as np

__all__ = [
    'Equal',
    'Delta',
    'Kelly',
]


class CapitalManagement(ABC):
    """
        distribution base class
    """
    @abstractmethod
    def compute(self,assets,cash):
        raise NotImplementedError


class Equal(CapitalManagement):

    def compute(self,assets,cash):
        mappings = {asset:cash / len(assets) for asset in assets}
        return mappings


class Delta(CapitalManagement):
    """
        基于波动率测算持仓比例 --- 基于策略形成的净值的波动性分配比例 --- 类似于海龟算法
    """
    def __init__(self,
                 delta_func,
                 data_portal,
                 window,
                 frequency = 'daily'):
        # delta_func --- (asset,result)
        self._func = delta_func
        self.data_portal = data_portal
        self._window = window
        self.frequency = frequency

    @property
    def window(self):
        return self._window

    def handle_data(self,assets,dt):
        his = self.data_portal.get_history_window(
                               assets,
                               dt,
                               self.window,
                               ['open', 'high', 'low', 'close', 'volume'],
                               self.frequency
                                                            )
        return his

    def compute(self,assets,cash,dts):
        """
            基于数据的波动性以及均值
            e.g. [5,4,8,3,6]  --- [6,5,3,4]
        """
        datas = self.handle_data(assets,dts)
        with Pool(processes = len(assets)) as pool:
            result = [pool.apply_async(self._func(datas[asset])) for asset in assets]
            assets,values = list(zip(result))
            reverse_idx = len(values) - np.argsort(values) - 1
            reverse_values = values[reverse_idx]
            capital = list(zip(assets,reverse_values))
            return capital


class Kelly(CapitalManagement):

    def compute(self,assets,cash):

        raise NotImplementedError('kelly 基于策略的胜率进行分配')