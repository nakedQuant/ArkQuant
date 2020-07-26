# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import defaultdict,OrderedDict
import numpy as np , pandas as pd
from functools import partial
from finance.position import Position


class PositionStats(object):

    gross_exposure = None
    gross_value = None
    long_exposure = None
    net_exposure = None
    net_value = None
    short_exposure = None
    longs_count = None
    shorts_count = None
    position_exposure_array = np.array()
    position_exposure_series = pd.series()

    def __new__(cls):
        self = cls()
        es = pd.Series(np.array([],dtype = 'float64'),index = np.array([],dtpe = 'int64'))
        self._underlying_value_array = es.values
        self._underlying_index_array = es.index.values
        return self


def calculate_position_tracker_stats(positions,stats):
    """
        stats ---- PositionStats
        return portfolio value and num of position
    """
    count = 0
    net_exposure = 0

    for outer_position in positions.values():
        position = outer_position.inner_position
        #daily更新价格
        exposure = position.amount * position.last_sale_price
        count += 1
        net_exposure += exposure

    stats.net_exposure = net_exposure
    stats.count = count


class PositionTracker(object):
    """
        持仓变动
        the current state of position held
    """
    def __init__(self,data_portal):
        self.data_portal = data_portal
        self.positions = OrderedDict()
        #根据时间记录关闭的交易
        self.record_closed_position = defaultdict(list)
        #cache the stats until
        self._dirty_stats = True
        self._stats = PositionStats.new()

    def _calculate_adjust_ratio(self,asset,dt):
        """
            股权登记日 ex_date
            股权除息日（为股权登记日下一个交易日）
            但是红股的到账时间不一致（制度是固定的）
            根据上海证券交易规则，对投资者享受的红股和股息实行自动划拨到账。股权（息）登记日为R日，除权（息）基准日为R+1日，
            投资者的红股在R+1日自动到账，并可进行交易，股息在R+2日自动到帐，
            其中对于分红的时间存在差异

            根据深圳证券交易所交易规则，投资者的红股在R+3日自动到账，并可进行交易，股息在R+5日自动到账，

            持股超过1年：税负5%;持股1个月至1年：税负10%;持股1个月以内：税负20%新政实施后，上市公司会先按照5%的最低税率代缴红利税
        """
        divdend = self.data_portal.load_divdends_for_sid(asset.sid,dt)
        try:
            amount_ratio = (divdend['sid_bonus'] +divdend['sid_transfer']) / 10
            cash_ratio = divdend['bonus'] / 10
        except :
            amount_ratio = 0.0
            cash_ratio = 0.0
        return amount_ratio,cash_ratio

    def _retrieve_right_from_sqlite(self,asset,dt):
        """
            配股机制有点复杂 ， freeze capital
            如果不缴纳款，自动放弃到期除权相当于亏损,在股权登记日卖出，一般的配股缴款起止日为5个交易日
        """
        rights = self.data_portal.load_rights_for_sid(asset.sid,dt)
        return rights

    def handle_splits(self,dt):
        total_left_cash = 0
        for asset,position in self.positions.items():
            amount_ratio,cash_ratio = self._calculate_adjust_ratio(asset,dt)
            left_cash = position.handle_split(amount_ratio,cash_ratio)
            total_left_cash += left_cash
        return total_left_cash

    def _update_position(self,transaction):
        asset = transaction.asset
        try:
            position = self.positions[asset]
        except KeyError:
            position = self.positions[asset] = Position(asset)
        finally:
            cash_flow = position.update(transaction)
        if position._closed:
            self.record_closed_position[transaction.dt].append(position)
            del self.positions[asset]
        return cash_flow

    def execute_transaction(self,transactions):
        """执行完交易cash变动"""
        aggregate_cash_flow = 0.0
        for txn in transactions:
            aggregate_cash_flow += self._update_position(txn)
        self._dirty_stats = False
        return aggregate_cash_flow

    def sync_last_date(self,dt):
        for outer_position in self.positions.values():
            inner_position = outer_position.inner_position
            inner_position.last_sync_date = dt

    def sync_last_price(self):
        """update last_sale_price of position"""
        assets =[position.inner_position.asset for position in self.positions]
        dts = [position.inner_position.last_sync_date for position in self.positions]
        if len(set(dts)) >1 :
            raise ValueError('sync all the position date')
        dt = dts[0]
        get_price = partial(self.data_portal.get_window_data,
                            dt = dt,
                            field = 'close',
                            days_in_window = 0,
                            frequency = 'daily'
                            )
        last_sync_prices = get_price(assets = assets)
        for asset,outer_position in self.positions.items():
            inner_position = outer_position.inner_position
            asset = inner_position.asset
            inner_position.last_sync_price = last_sync_prices[asset]

    @property
    def stats(self):
        """基于sync_last_sale_price  --- 计算每天的暴露度也就是porfolio"""
        calculate_position_tracker_stats(self.positions,self._stats)
        return self._stats

    def get_positions(self):
        # protocol
        positions = {}
        for asset, position in self.positions.items():
            # Adds the new position if we didn't have one before, or overwrite
            # one we have currently
            positions[asset] = position.protocol_position

        return positions

    def maybe_create_close_position_transaction(self,txn):
        """强制平仓机制 --- 设立仓位 ，改变脚本运行逻辑"""

        raise NotImplementedError('all action determined by alogrithm')