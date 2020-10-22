# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from finance.portfolio import Portfolio
from finance.account import Account
from finance._protocol import MutableView
from finance.position_tracker import PositionTracker
from risk.alert import UnionRisk


class Ledger(object):
    """
        the ledger tracks all orders and transactions as well as the current state of the portfolio and positions
        逻辑 --- 核心position_tracker （ process_execution ,handle_splits , handle_dividend ) --- 生成position_stats
        更新portfolio --- 基于portfolio更新account
        --- 在交易开始之前决定是否退出position 由于risk management
        优化:
        # event manager 用于处理ledger(righted violated expired postion) trigger -- callback pattern
    """
    __slots__ = ['_portfolio', 'position_tracker', 'fuse_model','risk_alert',
                 '_processed_transaction', '_previous_total_returns',
                 '_dirty_portfolio', '_dirty_positions', 'daily_returns_series']

    def __init__(self,
                 sim_params,
                 risk_models,
                 fuse_model):
        """构建可变、不可变的组合、账户"""
        self._portfolio = MutableView(Portfolio(sim_params.capital_base))
        self.position_tracker = PositionTracker()
        self.risk_alert = UnionRisk(risk_models)
        self.fuse_model = fuse_model
        self._processed_transaction = []
        self._previous_total_returns = 0
        self._dirty_positions = True
        self._dirty_portfolio = True

    @property
    def positions(self):
        # 获取position protocol
        assert not self._dirty_positions, 'positions are not accurate'
        return self.position_tracker.get_positions()

    @property
    def portfolio(self):
        assert self._dirty_portfolio, 'portfolio is not accurate'
        return self._portfolio

    @property
    def account(self):
        return Account(self.portfolio)

    def _cash_flow(self, capital_amount):
        """
            update the cash of portfolio
        """
        p = self._portfolio
        p.cash_flow += capital_amount

    def start_of_session(self, session_ix):
        left_cash = self.position_tracker.handle_splits(session_ix)
        self._cash_flow(left_cash)
        self._previous_total_returns = self._portfolio.returns
        self._dirty_portfolio = True
        self._dirty_positions = True

    def process_transaction(self, transactions):
        txn_capital = self.position_tracker.handle_transaction(transactions)
        self._cash_flow(txn_capital)
        self._processed_transaction.append(transactions)

    def daily_position_stats(self, dts):
        """
            param dts: %Y-%m-%d --- 包括已有持仓以及当天关闭持仓的收益率
        """
        assert not self._dirty_portfolio, 'stats is accurate after end_session'
        stats_returns = dict()
        for asset, p in self.positions.items():
            stats_returns[asset] = p.amount * (p.last_sync_price - p.cost_basis)

        closed_position = self.position_tracker.record_closed_position[dts]
        for p in closed_position:
            stats_returns[p.asset] = p.amount * (p.last_sync_price - p.cost_basis)
        return stats_returns

    def _calculate_portfolio_stats(self):
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

        # 更新组合投资的收益，并计算组合的符合收益率
        pnl = end_value - start_value
        returns = pnl / start_value
        portfolio.pnl += pnl
        # 复合收益率
        portfolio.returns = (
            (1+portfolio.returns) *
            (1+returns) - 1
        )
        # 更新持仓价值
        portfolio.positions_values = position_values
        # 更新组合持仓
        self.portfolio.positions = self.positions
        # 资金使用效率
        portfolio.utility = position_values / end_value
        self._dirty_portfolio = False

    def end_of_session(self):
        session_ix = self.position_tracker.synchronize()
        self._dirty_positions = False
        self._calculate_portfolio_stats()
        self.portfolio.daily_returns(session_ix)
        self._dirty_portfolio = False
        self.fuse_model.trigger(self._portfolio)

    def get_transactions(self, dt):
        """
        :param dt: %Y-%m-%d
        :return: transactions on the dt
        """
        dt_txns = [txn for txn in self._processed_transaction
                   if txn.dt.strftime('%Y-%m-%d') == dt]
        return dt_txns

    def get_violate_risk_positions(self):
        # 获取违反风控管理的仓位
        violate_positions = [p.protocol for p in self.positions
                             if self.risk_alert.should_trigger(p)]
        return violate_positions

    def get_rights_positions(self, dts):
        # 获取当天为配股登记日的仓位 --- 卖出 因为需要停盘产生机会成本
        assets = [p.asset for p in self.positions]
        rights = self.position_tracker.retrieve_equity_rights(assets, dts)
        p_mapping = {p.asset.sid: p for p in self.positions}
        right_positions = None if rights.empty else [p_mapping[s].protocol for s in rights.index]
        return right_positions

    def _cleanup_expired_assets(self, dt):
        """
        Clear out any assets that have expired before starting a new sim day.

        Finds all assets for which we have positions and generates
        close_position events for any assets that have reached their
        close_date.
        """
        def past_close_date(asset):
            acd = asset.last_traded
            return acd is not None and acd == dt
        # Remove positions in any sids that have reached their auto_close date.
        positions_to_clear = \
            [p for p in self.positions if past_close_date(p.asset)]
        return positions_to_clear

    def get_expired_positions(self, dts):
        expires = self._cleanup_expired_assets(dts)
        return expires


__all__ = ['Ledger']
