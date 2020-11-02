# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from gateway.driver.data_portal import portal
from util.math_utils import measure_volatity


class CapitalUsage(ABC):
    """
        distribution base class
    """
    @abstractmethod
    def compute(self, assets, capital, dts):
        raise NotImplementedError


class Equal(CapitalUsage):

    def compute(self, assets, capital, dts):
        mappings = {asset: capital / len(assets) for asset in assets}
        return mappings


class Turtle(CapitalUsage):
    """
        基于波动率测算持仓比例 --- 基于策略形成的净值的波动性分配比例 --- 类似于海龟算法
    """
    def __init__(self,
                 window,
                 delta_func=None):
        self._window = window
        self._func = delta_func if delta_func else measure_volatity

    @property
    def window(self):
        return self._window

    def handle_data(self, assets, dt):
        his = portal.get_history_window(
                               assets,
                               dt,
                               -abs(self.window),
                               ['open', 'high', 'low', 'close', 'volume'],
                               'daily')
        return his

    def compute(self, assets, capital, dts):
        """
            类比于海龟交易 --- 基于数据的波动性划分仓位 最小波动率对应最大仓位
            由于相同的sid但是不一定相同的asset需要进行处理
        """
        if len(assets) > 1:
            his_window = self.handle_data(assets, dts)
            # print('window', his_window)
            from toolz import valmap, groupby
            results = valmap(lambda x: self._func(x), his_window)
            aggregate = sum(results.values())
            # print('results', results)
            allocation = valmap(lambda x: capital * (1 - x / aggregate), results)
            group_sid = groupby(lambda x: x.sid, assets)
            output = {symbol: allocation[symbol.sid] / len(group_sid[symbol.sid]) for symbol in assets}
        else:
            output = {assets[0]: capital}
        return output


class Kelly(CapitalUsage):

    def compute(self, assets, available_capital, dts):

        raise NotImplementedError('kelly 基于策略的胜率进行分配')


__all__ = [
    'Equal',
    'Turtle',
    'Kelly',
]


if __name__ == '__main__':

    from gateway.asset.assets import Equity
    # assets = [Equity('600438'), Equity('600000')]
    assets = [Equity('600438')]
    delta = Turtle(5)
    allocation = delta.compute(assets, 100000, '2019-09-02')
    print('allocation', allocation)
