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


class SimpleUncover(BaseUncover):

    @staticmethod
    def _uncover_by_price(size, asset, dts):
        # 模拟价格分布
        restricted_change = asset.restricted_change(dts)
        open_pct, pre_close = portal.get_open_pct(asset, dts)
        print('open_pct, pre_close', open_pct, pre_close)
        dist = 1 + np.random.uniform(- 2 * abs(open_pct), abs(open_pct) * 2, size) if size > 0 else \
            np.array([1 + open_pct])
        print('dist', dist)
        clip_pct = np.clip(dist, (1 - restricted_change), (1 + restricted_change))
        # print('clip_pct', clip_pct)
        sim_prices = clip_pct * pre_close
        print('sim_prices', sim_prices)
        return sim_prices

    @staticmethod
    def _uncover_by_ticker(size, asset, dts):
        # 按照固定时间区间去拆分
        dts = pd.Timestamp(dts) if isinstance(dts, str) else dts
        interval = 4 * 60 / size
        # print('uncover_by_ticker', interval)
        upper = pd.date_range(start=dts + pd.Timedelta(hours=9, minutes=30),
                              end=dts + pd.Timedelta(hours=11, minutes=30),
                              freq='%dmin' % interval)
        # print('uncover by ticker upper', upper)
        bottom = pd.date_range(start=dts + pd.Timedelta(hours=13, minutes=30),
                               end=dts + pd.Timedelta(hours=14, minutes=57),
                               freq='%dmin' % interval)
        # print('uncover by ticker bottom', bottom)
        intervals = list(chain(*zip(upper, bottom)))
        intervals if len(intervals) == size else intervals.append(dts + pd.Timedelta(hours=14, minutes=57))
        # print('tick_intervals', len(intervals), intervals)
        return intervals

    def _underneath_size(self, asset, amount, base_amount, dts):
        sign = np.sign(amount)
        tick_size = asset.tick_size
        size = int(np.floor(abs(amount) / base_amount))
        # print('underneath size', size)
        amount_array = np.tile([base_amount], size)
        print('underneath base_amount array', amount_array)
        abundant = int(abs(amount) % base_amount)
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
        amount_array = amount_array * sign
        print('_underneath amount array', amount_array)
        return amount_array, size

    def create_iterables(self, asset, amount, per_amount, dt):
        dt = pd.Timestamp(dt) if isinstance(dt, str) else dt
        if amount > per_amount:
            amount_arrays, size = self._underneath_size(asset, amount, per_amount, dt)
            dist_arrays = self._uncover_by_ticker(size, asset, dt)
            iterables = zip(amount_arrays, dist_arrays)
        else:
            dt = dt + pd.Timedelta(hours=9, minutes=30)
            iterables = [[amount, dt]]
        return iterables


__all__ = ['SimpleUncover']
