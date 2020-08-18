# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, numpy as np, warnings
from finance.portfolio import Portfolio
from finance._protocol import Account, MutableView
from finance.position_tracker import PositionTracker
from risk.alert import UnionRisk, PortfolioRisk


class Ledger(object):
    """
        the ledger tracks all orders and transactions as well as the current state of the portfolio and positions
        逻辑 --- 核心position_tracker （ process_execution ,handle_splits , handle_divdend) --- 生成position_stats
        更新portfolio --- 基于portfolio更新account
        --- 简化 在交易开始之前决定是否退出position 由于risk management
    """
    __slots__ = ['_portfolio', 'position_tracker', '_processed_transaction', '_previous_total_returns',
                 '_dirty_portfolio', '_synchronized', 'daily_returns_series', 'risk_alert']

    def __init__(self,
                 sim_params,
                 risk_models):
        """构建可变、不可变的组合、账户"""
        self._portfolio = MutableView(Portfolio(sim_params.capital_base))
        self.daily_returns_series = pd.Series(np.nan, index=sim_params.sessions)
        self.position_tracker = PositionTracker()
        self.risk_alert = UnionRisk(risk_models)
        self._processed_transaction = []
        self._previous_total_returns = 0
        self._dirty_portfolio = True
        self._synchronized = False

    @property
    def synchronized_clock(self):
        assert self._synchronized, 'must synchronize dts for positions first '
        dts = set([p.last_sync_date
                   for p in self.positions.values()])
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
        return self._portfolio

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
        self._synchronized = True

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
        # 组合净值（持仓净值 + 现金）
        start_value = portfolio.portfolio_value
        portfolio.portfolio_value = end_value = \
            position_values + portfolio.start_cash
        # 资金使用效率
        portfolio.utility = position_values / end_value
        # 更新持仓价值
        portfolio.positions_values = position_values
        # 更新组合投资的收益，并计算组合的符合收益率
        pnl = end_value - start_value
        returns = pnl / start_value
        portfolio.pnl += pnl
        # 复合收益率
        portfolio.returns = (
            (1+portfolio.returns) *
            (1+returns) - 1
        )
        # 更新组合持仓
        self.portfolio.positions = self.positions
        self._dirty_portfolio = False

    # 计算账户当天的收益率
    def end_of_session(self):
        self._calculate_portfolio_stats()
        session_ix = self.position_tracker.update_sync_date
        self.position_tracker.sync_position_returns()
        self.daily_returns_series[session_ix] = self.daily_returns
        self.portfolio.record_returns(session_ix)
        self._dirty_portfolio = False
        self._synchronized = False

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

    def get_rights_positions(self, dts):
        # 获取当天为配股登记日的仓位 --- 卖出 因为需要停盘产生机会成本
        assert self._synchronized, 'must synchronize dts for positions first '
        right_positions = []
        for position in self.positions:
            asset = position.asset
            right = self.position_tracker.retrieve_right_of_equity(asset, dts)
            if len(right):
                right_positions.append(position.protocol)
        return right_positions

    def get_violate_risk_positions(self):
        # 获取违反风控管理的仓位
        violate_positions = [p.protocol for p in self.positions
                             if self.risk_alert.should_trigger(p)]
        return violate_positions

    def _cleanup_expired_assets(self, dt):
        """
        Clear out any assets that have expired before starting a new sim day.

        Finds all assets for which we have positions and generates
        close_position events for any assets that have reached their
        close_date.
        """
        def past_close_date(asset):
            acd = asset.last_traded_date
            return acd is not None and acd <= dt

        # Remove positions in any sids that have reached their auto_close date.
        positions_to_clear = \
            [p for p in self.positions if past_close_date(p.asset)]
        return positions_to_clear

    def get_expired_positions(self, dts):
        expires = self._cleanup_expired_assets(dts)
        return expires
