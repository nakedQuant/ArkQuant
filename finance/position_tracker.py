# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import defaultdict, OrderedDict
from functools import partial
from finance.position import Position
from gateway.driver.data_portal import DataPortal

__all__ = ['PositionTracker']

# init data_portal
data_portal = DataPortal()


class PositionTracker(object):
    """
        track the change of position
    """
    def __init__(self):
        self.portal = data_portal
        self.positions = OrderedDict()
        # 根据时间记录关闭的交易
        self.record_closed_position = defaultdict(list)
        self.update_sync_date = None
        self._dirty_stats = True

    @property
    def is_synchronized(self):
        return False

    def _calculate_adjust_ratio(self, asset, dt):
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
        dividend = self.portal.get_dividends_for_asset(asset, dt)
        try:
            amount_ratio = (dividend['sid_bonus'] + dividend['sid_transfer']) / 10
            cash_ratio = dividend['bonus'] / 10
        except ZeroDivisionError:
            amount_ratio = 0.0
            cash_ratio = 0.0
        return amount_ratio,cash_ratio

    def handle_splits(self):
        assert self.is_synchronized, ValueError('must sync first')
        total_left_cash = 0
        for asset, position in self.positions.items():
            amount_ratio, cash_ratio = self._calculate_adjust_ratio(asset, self.update_sync_date)
            left_cash = position.handle_split(amount_ratio, cash_ratio)
            total_left_cash += left_cash
        return total_left_cash

    def _handle_transaction(self, transaction):
        event = transaction.event
        asset = event.asset
        try:
            position = self.positions[asset]
        except KeyError:
            position = self.positions[asset] = Position(event)
        cash_flow = position.update(transaction)
        if position.closed:
            dts = transaction.created_dt.strftime('%Y-%m-%d')
            self.record_closed_position[dts].append(position)
            del self.positions[asset]
        return cash_flow

    def execute_transaction(self, transactions):
        """执行完交易cash变动"""
        aggregate_cash_flow = 0.0
        for txn in transactions:
            aggregate_cash_flow += self._handle_transaction(txn)
        self._dirty_stats = False
        return aggregate_cash_flow

    def sync_last_date(self, dt):
        self.update_sync_date = dt
        for position in self.positions.values():
            position.last_sync_date = dt

    def sync_last_prices(self):
        """update last_sale_price of position --- 第二天更新仓位"""
        get_price = partial(
                            self.portal.get_window,
                            dt=self.update_sync_date,
                            field='close',
                            days_in_window=-1,
                            frequency='daily'
                            )
        last_sync_prices = get_price(assets=set(self.positions))
        for asset, inner_position in self.positions.items():
            assert asset == inner_position.asset, ValueError('unmatched position')
            inner_position.last_sync_price = last_sync_prices[asset]

    def sync_position_returns(self):
        for p in self.positions.values():
            p.calculate_returns()

    def get_positions(self):
        # protocol list
        protocols = [position.protocol for position in self.positions.values()]
        return protocols

    def retrieve_right_of_equity(self, asset, dt):
        """
            配股机制有点复杂 ， freeze capital
            如果不缴纳款，自动放弃到期除权相当于亏损,在股权登记日卖出，一般的配股缴款起止日为5个交易日
        """
        rights = self.portal.get_rights_for_asset(asset, dt)
        return rights


if __name__ == '__main__':

    tracker = PositionTracker()
    print('tracker', tracker)
