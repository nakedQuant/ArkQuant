#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from contextlib import contextmanager
from enum import Enum


class TradeSimilarity(object):
    """
        计算交易策略的相关性思路：研究不同交易策略的对应的订单集合
        包装交易订单构成的pd.DataFrame对象，外部debug因子的交易结果，寻找交易策略的问题使用，
        支持两个orders_pd的并集，交集，差集，类似set的操作，同时支持相等，不等，大于，小于
        的比较操作，eg如下：
            with orders_pd1.proxy_work(orders_pd2) as (order1, order2):
                a = order1 | order2 # 两个交易结果的并集
                b = order1 & order2 # 两个交易结果的交集
                c = order1 - order2 # 两个交易结果的差集(在order1中，但不在order2中)
                d = order2 - order1 # 两个交易结果的差集(在order2中，但不在order1中)
                eq = order1 == order2 # 两个交易结果是否相同
                lg = order1 > order2 # order1唯一的交易数量是否大于order2
                lt = order1 < order2 # order1唯一的交易数量是否小于order2
    """

    def __init__(self, orders_pd, same_rule):
        """
        初始化函数需要pd.DataFrame对象，暂时未做类型检测
        :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
        :param same_rule: order判断为是否相同使用的规则, 默认EOrderSameRule.ORDER_SAME_BSPD
                          即：order有相同的symbol和买入日期和相同的卖出日期和价格才认为是相同
        """
        # 需要copy因为会添加orders_pd的列属性等
        self.orders_pd = orders_pd.copy()
        self.same_rule = same_rule
        # 并集, 交集, 差集运算结果存储
        self.op_result = None
        self.last_op_metrics = {}

    @contextmanager
    def proxy_work(self, orders_pd):
        """
        传人需要比较的orders_pd，构造ABuOrderPdProxy对象，返回使用者，
        对op_result进行统一分析
        :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
        :return:
                self.last_op_metrics['win_rate'] = metrics.win_rate
                self.last_op_metrics['gains_mean'] = metrics.gains_mean
                self.last_op_metrics['losses_mean'] = metrics.losses_mean
                self.last_op_metrics['sum_profit'] = self.op_result['profit'].sum()
                self.last_op_metrics['sum_profit_cg'] = self.op_result['profit_cg'].sum()
        """
        # 运算集结果重置

    def __and__(self, other):
        """ & 操作符的重载，计算两个交易集的交集"""
        # self.op = 'intersection(order1 & order2)'

    def __or__(self, other):
        """ | 操作符的重载，计算两个交易集的并集"""
        # self.op = 'union(order1 | order2)'

    def __sub__(self, other):
        """ - 操作符的重载，计算两个交易集的差集"""

    def __eq__(self, other):
        """ == 操作符的重载，计算两个交易集的是否相同"""

    def __gt__(self, other):
        """ > 操作符的重载，计算两个交易集的大小, 类被total_ordering装饰，可以支持lt等操作符"""


class EOrderSameRule(Enum):
    """对order_pd中对order判断为是否相同使用的规则"""

    """order有相同的symbol和买入日期就认为是相同"""
    ORDER_SAME_BD = 0
    """order有相同的symbol, 买入日期，和卖出日期，即不考虑价格，只要日期相同就相同"""
    ORDER_SAME_BSD = 1
    """order有相同的symbol, 买入日期，相同的买入价格，即单子买入时刻都相同"""
    ORDER_SAME_BDP = 2
    """order有相同的symbol, 买入日期, 买入价格, 并且相同的卖出日期和价格才认为是相同，即买入卖出时刻都相同"""
    ORDER_SAME_BSPD = 3


def _same_pd(order, other_orders_pd, same_rule):
    """
    根据same_rule的规则从orders_pd和other_orders_pd中返回相同的df

    :param order: orders_pd中的一行order记录数据
    :param other_orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param same_rule: order判断为是否相同使用的规则
    :return: 从orders_pd和other_orders_pd中返回相同的df
    """
    symbol = order.symbol
    buy_day = order['buy_date']
    buy_price = order['buy_price']

    sell_day = order['sell_date']
    sell_price = order['sell_price']

    if same_rule == EOrderSameRule.ORDER_SAME_BD:
        # 只根据买入时间和买入symbol确定是否相同，即认为在相同的交易日买入相同的股票，两笔交易就一样，忽略其它所有order中的因素
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)]
    elif same_rule == EOrderSameRule.ORDER_SAME_BSD:
        # 根据买入时间，卖出时间和买入symbol确定是否相同
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)
                                  & (other_orders_pd['sell_date'] == sell_day)]
    elif same_rule == EOrderSameRule.ORDER_SAME_BDP:
        # 根据买入时间，买入价格和买入symbol确定是否相同
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)
                                  & (other_orders_pd['buy_price'] == buy_price)]
    elif same_rule == EOrderSameRule.ORDER_SAME_BSPD:
        # 根据买入时间，卖出时间, 买入价格和卖出价格和买入symbol确定是否相同
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)
                                  & (other_orders_pd['sell_date'] == sell_day)
                                  & (other_orders_pd['buy_price'] == buy_price)
                                  & (other_orders_pd['sell_price'] == sell_price)]
    else:
        raise TypeError('same_rule type is {}!!'.format(same_rule))
    return same_pd
