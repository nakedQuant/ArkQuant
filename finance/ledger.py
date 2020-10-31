# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from toolz import keymap, keyfilter
from finance.portfolio import Portfolio
from finance.account import Account
# from finance._protocol import MutableView
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
    __slots__ = ['_portfolio', 'position_tracker', 'fuse_risk','risk_alert',
                 '_processed_transaction', '_previous_total_returns',
                 '_dirty_portfolio', '_dirty_positions', 'daily_returns_series']

    def __init__(self,
                 sim_params,
                 risk_models,
                 fuse_model):
        """构建可变、不可变的组合、账户"""
        # self._portfolio = MutableView(Portfolio(sim_params))
        self._portfolio = Portfolio(sim_params)
        self.position_tracker = PositionTracker()
        self.risk_alert = UnionRisk(risk_models)
        self.fuse_risk = fuse_model
        self._processed_transaction = []
        self._previous_total_returns = 0
        # self._dirty_positions = True
        self._dirty_portfolio = True

    @property
    def positions(self):
        # assert not self._dirty_positions, 'positions are not accurate'
        return self.position_tracker.get_positions()

    @property
    def portfolio(self):
        assert not self._dirty_portfolio, 'portfolio is not accurate'
        return self._portfolio

    @property
    def account(self):
        return Account(self.portfolio)

    def _cash_flow(self, capital_amount):
        """
            update the cash of portfolio
        """
        p = self._portfolio
        # p.cash_flow += capital_amount
        p.portfolio_cash = capital_amount

    def start_of_session(self, session_ix):
        # handle splits and  update last_sync_date
        left_cash = self.position_tracker.handle_splits(session_ix)
        print('start handle split cash', left_cash)
        self._cash_flow(left_cash)
        self._previous_total_returns = self._portfolio.returns
        self._portfolio.positions = self.positions
        # update portfolio daily value
        self._portfolio.record_daily_value(session_ix)
        self._dirty_portfolio = False
        print('start portfolio cash', self.portfolio.portfolio_cash)
        # self._dirty_positions = True

    def process_transaction(self, transactions):
        print('ledger process_transaction')
        txn_capital = self.position_tracker.handle_transactions(transactions)
        print('txn_capital', txn_capital)
        self._cash_flow(txn_capital)
        self._processed_transaction.extend(transactions)

    def _calculate_portfolio_stats(self):
        # 计算持仓组合净值
        position_values = sum([p.amount * p.last_sync_price
                               for p in self.positions.values()])
        # print('position_values', position_values)
        # 持仓组合
        portfolio = self._portfolio
        self._previous_total_returns = portfolio.returns
        # 组合净值（持仓净值 + 现金）
        start_value = portfolio.portfolio_value
        # print('start_value', start_value)
        # portfolio.portfolio_value = end_value = \
        #     position_values + portfolio.start_cash
        portfolio.portfolio_value = end_value = \
            position_values + portfolio.portfolio_cash
        # print('portfolio_cash', portfolio.portfolio_cash)
        # 更新组合投资的收益，并计算组合的符合收益率
        pnl = end_value - start_value
        # print('pnl', pnl)
        returns = pnl / start_value
        # print('daily returns', returns)
        portfolio.pnl += pnl
        # 复合收益率
        portfolio.returns = \
            (1+portfolio.returns) * (1+returns) - 1
        # print('cum returns', portfolio.returns)
        # 更新持仓价值
        portfolio.positions_values = position_values
        # 更新组合持仓
        portfolio.positions = self.positions
        # 资金使用效率
        portfolio.utility = position_values / end_value
        # print('portfolio utility', portfolio.utility)
        self._dirty_portfolio = False

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

    def end_of_session(self):
        self._dirty_portfolio = True
        # synchronize() -- update position attr and calculate position returns
        self.position_tracker.synchronize()
        # self._dirty_positions = False
        self._calculate_portfolio_stats()
        self._dirty_portfolio = False
        self.fuse_risk.trigger(self._portfolio)
        print('end session ledger positions', self.positions)
        # print('end session portfolio daily value', self.portfolio.portfolio_daily_value)

    def get_transactions(self, dt):
        """
        :param dt: %Y-%m-%d
        :return: transactions on the dt
        """
        transactions_on_dt = [txn for txn in self._processed_transaction
                              if txn.created_dt.strftime('%Y-%m-%d') == dt.strftime('%Y-%m-%d')]
        return transactions_on_dt

    def get_violate_risk_positions(self):
        # 获取违反风控管理的仓位
        violate_positions = [p for p in self.positions.values()
                             if self.risk_alert.should_trigger(p)]
        return violate_positions

    def get_rights_positions(self, dts):
        # 获取当天为配股登记日的仓位 --- 卖出 因为需要停盘产生机会成本
        assets = set(self.positions)
        # print('ledger assets', assets)
        rights = self.position_tracker.retrieve_equity_rights(assets, dts)
        # print('ledger rights', rights)
        mapping_protocol = keymap(lambda x: x.sid, self.positions)
        # print('ledger mapping_protocol', mapping_protocol)
        union_assets = set(mapping_protocol) & set(rights.index)
        # print('ledger union_assets', union_assets)
        union_positions = keyfilter(lambda x: x in union_assets, mapping_protocol) if union_assets else None
        # print('ledger union_positions', union_positions)
        right_positions = list(union_positions.values()) if union_positions else []
        # print('right_positions', right_positions)
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
            [p for symbol, p in self.positions.items() if past_close_date(symbol)]
        return positions_to_clear

    def get_expired_positions(self, dts):
        expires = self._cleanup_expired_assets(dts)
        return expires


__all__ = ['Ledger']

