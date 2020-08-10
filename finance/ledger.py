# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, numpy as np, warnings
from ._protocol import Portfolio, Account
from .position_tracker import PositionTracker
from ._protocol import MutableView


class Ledger(object):
    """
        the ledger tracks all orders and transactions as well as the current state of the portfolio and positions
        逻辑 --- 核心position_tracker （ process_execution ,handle_splits , handle_divdend) --- 生成position_stats
        更新portfolio --- 基于portfolio更新account
    """
    __slots__ = ['_portfolio', 'position_tracker', '_processed_transaction',
                 '_previous_total_returns', '_dirty_portfolio', 'daily_returns_series']

    def __init__(self, trading_sessions, capital_base):
        """构建可变、不可变的组合、账户"""
        if not len(trading_sessions):
            raise Exception('calendars must not be null')

        self._portfolio = Portfolio(capital_base)
        self._processed_transaction = []
        self._previous_total_returns = 0
        self._dirty_portfolio = True
        self.daily_returns_series = pd.Series(np.nan, index=trading_sessions)
        self.position_tracker = PositionTracker()

    @property
    def synchronized_clock(self):
        dts = set([p.last_sync_date
                   for p in self.positions.values()])
        assert len(dts) == 1, Exception('positions must sync at the same time')
        return dts

    @property
    def positions(self):
        # 获取position protocol
        return self.position_tracker.get_positions()

    @property
    def daily_returns(self):
        if self._dirty_portfolio:
            raise Exception('today_returns is avaiable at the end of session ')
        return (
            (self._portfolio.returns + 1) /
            (self._previous_total_returns + 1) - 1
        )

    @property
    def portfolio(self):
        if self._dirty_portfolio:
            raise Exception('portofilio is accurate at the end of session ')
        return MutableView(self._portfolio)

    @property
    def account(self):
        return Account(self._portfolio)

    def _cash_flow(self, capital_amount):
        """
            update the cash of portfolio
        """
        p = self._portfolio
        p.cash_flow += capital_amount

    def _process_dividends(self):
        """ splits and divdend"""
        left_cash = self.position_tracker.handle_spilts()
        self._cash_flow(left_cash)

    def start_of_session(self, session_ix):
        # 每天同步时间
        self.position_tracker.sync_last_date(session_ix)
        self._process_dividends()
        self._previous_total_returns = self._portfolio.returns
        self._dirty_portfolio = True

    def process_transaction(self, transactions):
        """每天不断产生的transactions，进行处理 """
        txn_capital = self.position_tracker.execute_transaction(transactions)
        self._cash_flow(txn_capital)
        self._processed_transaction.append(transactions)

    def _calculate_portfolio_stats(self):
        """同步持仓的close价格"""
        self.position_tracker.sync_last_prices()
        # 计算持仓组合净值
        position_values = sum([p.amount * p.last_sync_price
                               for p in self.positions.values()])
        # 持仓组合
        portfolio = self._portfolio
        self._previous_total_returns = portfolio.returns
        start_value = portfolio.portfolio_value
        portfolio.portfolio_value = end_value = \
            position_values + portfolio.start_cash
        # 资金使用效率
        portfolio.utility = position_values / end_value
        # 更新组合投资的收益，并计算组合的符合收益率
        pnl = end_value - start_value
        returns = pnl / start_value
        portfolio.pnl += pnl
        # 复合收益率
        portfolio.returns = (
            (1+portfolio.returns) *
            (1+returns) - 1
        )
        # 定义属性
        self.portfolio.positions = self.positions
        self._dirty_portfolio = False

    #计算账户当天的收益率
    def end_of_session(self):
        self._calculate_portfolio_stats()
        session_ix = self.position_tracker.update_sync_date
        self.daily_returns_series[session_ix] = self.daily_returns
        self._dirty_portfolio = False

    def get_rights_positions(self, dts):
        # 获取当天为配股登记日的仓位 --- 卖出 因为需要停盘产生机会成本
        right_positions = []
        for position in self.portfolio.positions:
            asset = position.inner_position.asset
            right = self.position_tracker.retrieve_right_from_sqlite(asset.sid, dts)
            if len(right):
                right_positions.append(position)
        return right_positions

    def get_transactions(self, dt):
        """
        :param dt: %Y-%m-%d
        :return: transactions on the dt
        """
        txns_on_dt = [txn for txn in self._processed_transaction
                      if txn.dt.strftime('%Y-%m-%d') == dt]
        return txns_on_dt

    # position_stats
    def daily_position_stats(self, dts):
        """
        :param dts: %Y-%m-%d --- 包括已有持仓以及当天关闭持仓的收益率
        engine -- conflicts解决了策略冲突具体就是标的冲突
        """
        assert not self._dirty_portfolio, 'stats is accurate after end_session'
        stats = dict()
        for asset, p in self.positions.items():
            stats[asset] = p.amount * (p.last_sync_price - p.cost_basis)

        closed_position = self.position_tracker.record_closed_position[dts]
        for p in closed_position:
            stats[p.asset] = p.amount * (p.last_sync_price - p.cost_basis)
        return stats

    def manual_withdraw_operation(self, assets):
        """
            self.position_tracker.maybe_create_close_position_transaction
            self.process_transaction(txn)
        """
        warnings.warn('avoid interupt automatic process')
        self.position_tracker.maybe_create_close_position_transaction(assets)
