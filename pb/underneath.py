# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np, pandas as pd
from itertools import chain
from abc import ABC, abstractmethod
from gateway.driver.data_portal import portal


class BaseUncover(ABC):

    @staticmethod
    @abstractmethod
    def _uncover_by_price(size, asset, dts):
        """
        :param size: int , the length of price_arrays
        :param asset: Asset
        :param dts: str '%Y-%m-%d'
        :return: to simulate distribution of price arrays
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _uncover_by_ticker(size, asset, dts):
        """
        :param size: int, the length of ticker_arrays
        :param asset: Asset
        :param dts: str '%Y-%m-%d'
        :return: to simulate distribution of ticker arrays
        """
        raise NotImplementedError()

    @abstractmethod
    def _underneath_size(self, asset, amount, base_amount, dts):
        """
        :param asset: Asset
        :param amount: int
        :param base_amount: per_amount (min amount)
        :param dts: str '%Y-%m-%d'
        :return: based on the amount to simulate size_arrays via base_amount
        """
        raise NotImplementedError()

    def create_iterables(self, asset, amount, per_amount, dt):
        amount_arrays, size = self._underneath_size(asset, amount, per_amount, dt)
        if asset.bid_mechanism:
            dist_arrays = self._uncover_by_ticker(size, asset, dt)
        else:
            dist_arrays = self._uncover_by_price(size, asset, dt)
        iterables = zip(amount_arrays, dist_arrays)
        return iterables


class UncoverAlgorithm(BaseUncover):

    @staticmethod
    def _uncover_by_price(size, asset, dts):
        print('uncover by price', size, asset, dts)
        # 模拟价格分布
        restricted_change = asset.restricted_change(dts)
        open_pct, pre_close = portal.get_open_pct(asset, dts)
        # alpha = 1 if open_pct == 0.00 else 100 * open_pct
        # print('alpha', alpha, size)
        # if size > 0:
        #     # dist = 1 + np.copysign(alpha, np.random.beta(abs(alpha), 100, size))/10
        #     dist = np.copysign(alpha, np.random.beta(abs(alpha), 100, size))/10
        #     print('beta', dist)
        #     dist = dist + 1
        # else:
        #     dist = [1 + alpha / 100]
        dist = 1 + np.random.normal(- 3 * open_pct, open_pct * 3, size) if size > 0 else 1 + open_pct
        print('dist', dist)
        print('restricted_change', restricted_change)
        clip_pct = np.clip(dist, (1 - restricted_change), (1 + restricted_change))
        print('clip_pct', clip_pct)
        sim_prices = clip_pct * pre_close
        print('sim_prices', sim_prices)
        return sim_prices

    @staticmethod
    def _uncover_by_ticker(size, asset, dts):
        # ticker arranged on sequence
        interval = 4 * 60 / size
        print('uncover_by_ticker', interval)
        # 按照固定时间去执行
        upper = pd.date_range(start='09:30', end='11:30', freq='%dmin' % interval)
        print('uncover by ticker', upper)
        bottom = pd.date_range(start='13:00', end='14:57', freq='%dmin' % interval)
        print('uncover by ticker', bottom)
        # 确保首尾
        tick_intervals = list(chain(*zip(upper, bottom)))[:size - 1]
        print('tick_intervals', tick_intervals)
        tick_intervals.append(pd.Timestamp('2020-06-17 14:57:00', freq='%dmin' % interval))
        return tick_intervals

    def _underneath_size(self, asset, amount, base_amount, dts):
        tick_size = asset.tick_size
        print('underneath tick_size', tick_size)
        size = int(np.floor(amount / base_amount))
        print('underneath size', size)
        amount_array = np.tile([base_amount], size)
        print('underneath amount_array', amount_array)
        abundant = int(amount % base_amount)
        print('underneath abundant', abundant)
        if asset.increment:
            num = np.floor(abundant / tick_size)
            random_idx = np.random.randint(0, size, int(num))
            try:
                for r in random_idx:
                    amount_array[r] += tick_size
            except IndexError:
                print('abundant is less than asset tick size')
                pass
        else:
            random_idx = np.random.randint(0, size, abundant)
            for r in random_idx:
                amount_array[r] += 1
        print('increment amount array', amount_array)
        return amount_array, size

    def create_iterables(self, asset, amount, per_amount, dt):
        amount_arrays, size = self._underneath_size(asset, amount, per_amount, dt)
        if asset.bid_mechanism:
            dist_arrays = self._uncover_by_ticker(size, asset, dt)
        else:
            dist_arrays = self._uncover_by_price(size, asset, dt)
        iterables = zip(amount_arrays, dist_arrays)
        return iterables


__all__ = ['UncoverAlgorithm']
