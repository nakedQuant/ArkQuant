# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from gateway.driver.data_portal import portal
from finance.order import Order, PriceOrder, TickerOrder, transfer_to_order
from finance.transaction import create_transaction
from util.dt_utilty import locate_pos


class SimulationBlotter(object):
    """
        前5个交易日,科创板科创板还设置了临时停牌制度，当盘中股价较开盘价上涨或下跌幅度首次达到30%、60%时，都分别进行一次临时停牌
        单次盘中临时停牌的持续时间为10分钟。每个交易日单涨跌方向只能触发两次临时停牌，最多可以触发四次共计40分钟临时停牌。
        如果跨越14:57则复盘

                科创板盘后固定价格交易 15:00 --- 15:30
        若收盘价高于买入申报指令，则申报无效；若收盘价低于卖出申报指令同样无效
        原则 --- 以收盘价为成交价，按照时间优先的原则进行逐笔连续撮合

        transform orders which are simulated by gen module to transactions
        撮合成交逻辑基于时间或者价格
    """
    def __init__(self,
                 commission_model,
                 slippage_model,
                 execution_model):
        self.commission = commission_model
        self.slippage = slippage_model
        self.execution = execution_model

    def _trigger_check(self, order, dts):
        """
            trigger orders checked by execution_style and slippage_style
        """
        if order:
            assert isinstance(order, Order), 'unsolved order type'
            asset = order.asset
            price = order.price
            # 基于订单的设置的上下线过滤订单
            upper = 1 + self.execution.get_limit_ratio(asset, dts)
            bottom = 1 - self.execution.get_stop_ratio(asset, dts)
            # avoid asset price reach price restriction
            if bottom < price < upper:
                # 计算滑价系数
                slippage = self.slippage.calculate_slippage_factor(asset, dts)
                order.price = price * (1+slippage)
                return order
            return False
        else:
            return False

    def _validate(self, order, dts):
        # fulfill the missing attr of PriceOrder and TickerOrder
        asset = order.asset
        price = order.price
        direction = np.sign(order.amount)
        minutes = portal.get_spot_value(dts, asset, 'minute', ['close'])
        # print('minutes price', minutes)
        if isinstance(order, PriceOrder):
            print('blotter order', order)
            ticker = locate_pos(price, minutes['close'], direction)
            print('ticker', ticker)
            new_order = transfer_to_order(order, ticker=ticker)
        elif isinstance(order, TickerOrder):
            ticker = order.created_dt
            print('TickerOrder, ticker', ticker)
            print('minutes', minutes['close'])
            price = minutes['close'][ticker]
            print('ticker price', price)
            new_order = transfer_to_order(order, price=price)
        elif isinstance(order, Order):
            new_order = order
        else:
            raise ValueError
        trigger_order = self._trigger_check(new_order, dts)
        print('trigger_order', trigger_order)
        return trigger_order

    def create_bulk_transactions(self, orders, dts):
        trigger_orders = [self._validate(order, dts) for order in orders]
        trigger_orders = [order for order in trigger_orders if order]
        print('trigger_orders', trigger_orders)
        # create txn
        transactions = [create_transaction(order_obj, self.commission) for order_obj in trigger_orders]
        print('transactions', transactions)
        return transactions


__all__ = ['SimulationBlotter']
