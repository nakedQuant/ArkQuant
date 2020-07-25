# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd , numpy as np


class Ledger(object):
    """
        the ledger tracks all orders and transactions as well as the current state of the portfolio and positions
        逻辑 --- 核心position_tracker （ process_execution ,handle_splits , handle_divdend) --- 生成position_stats
        更新portfolio --- 基于portfolio更新account

    """
    def __init__(self,trading_sessions,capital_base):
        """构建可变、不可变的组合、账户"""
        if not len(trading_sessions):
            raise Exception('calendars must not be null')

        # here is porfolio
        self._immutable_porfolio = Portfolio(start,capital_base)
        self._portfolio = MutableView(self._immutable_portfolio)
        self._immutable_account = Account()
        self._account = MutableView(self._immutable_account)
        self.position_tracker = PositionTracker()
        self._position_stats = None

        self._processed_transaction = []
        self._previous_total_returns = 0
        self.daily_returns_series = pd.Series(np.nan,index = trading_sessions)

    #账户每天起始
    def start_of_session(self,dt):
        self._prevoius_total_returns = self.portfolio.returns
        self.process_dividends(dt)
        self.position_tracker.sync_last_date(dt)
        self._dirty_portfolio = True

    def get_rights_positions(self,dts):
        # 获取当天为配股登记日的仓位 --- 卖出 因为需要停盘产生机会成本
        right_positions = []
        for position in self.portfolio.positions:
            asset = position.inner_position.asset
            right = self.position_tracker.retrieve_right_from_sqlite(asset.sid,dts)
            if len(right):
                right_positions.append(position)
        return right_positions

    def _cash_flow(self,amount):
        """
            update the cash of portfolio
        """
        self._dirty_portfolio = True
        p = self._portfolio
        p.cash_flow += amount
        p.cash += amount

    @property
    def capital_change(self,amount):
        self._cash_flow(amount)

    def process_transaction(self,transactions):
        """每天不断产生的transactions，进行处理 """
        txn_cash_flow = self.position_tracker.execute_transaction(transactions)
        self._cash_flow(txn_cash_flow)
        self._processed_transaction.append(transactions)

    def process_dividends(self,dt):
        """ splits and divdend"""
        left_cash = self.position_tracker.handle_spilts(dt)
        self._cash_flow(left_cash)

    def calculate_payoff(self,dt):
        """划分为持仓以及关闭的持仓"""
        closed_payoff = 0
        closed_positions = self.position_tracker.record_closed_position[dt]
        self._sync_last_prices(dt)
        #计算收益
        payoff = self._calculate_payout()
        for position in closed_positions:
            closed_payoff += position.cost_basis * position.amount
        total_payoff = payoff + closed_payoff
        return total_payoff

    def _calculate_payout(self):

        def _calculate(
                amount,
                old_price,
                price,
                multiplier=1):
            return (price - old_price) * multiplier * amount

        total = 0
        for position in self.positions.values():
            amount = position.amount
            old_price = position.cost_basis
            price = position.last_sale_price
            total += _calculate(
                amount,
                old_price,
                price
            )
        return total

    def _sync_last_prices(self):
        if self.position_tracker.dirty_stats:
            self.position_tracker.sync_last_sale_prices()

    def _update_porfolio(self):
        #计算持仓净值
        tracker = self.position_tracker
        position_stats = tracker.stats
        #更新投资组合收益
        portfolio = self._portfolio
        start_value = portfolio.portfolio_value
        portfolio.portfolio_value = end_value = \
            position_stats.net_exposure + portfolio.cash
        #更新组合投资的收益，并计算组合的符合收益率
        pnl = end_value - start_value
        returns = pnl / start_value
        portfolio.pnl += pnl
        #复合收益率
        portfolio.returns = (
            (1+portfolio.returns) *
            (1+returns) - 1
        )
        self.portfolio._dirty_portfolio = False

    #计算杠杆率
    def calculate_period_stats(self):
        position_stats = self.position_tracker.stats
        portfolio_value = self.portfolio.portfolio_value

        if portfolio_value == 0:
            gross_leverage = np.inf
        else:
            gross_leverage = position_stats.gross_exposure / portfolio_value
            net_leverage = position_stats.net_exposure / portfolio_value
        return gross_leverage,net_leverage

    @property
    def portfolio(self):
        if self._dirty_portfolio:
            raise Exception('portofilio is accurate at the end of session ')
        return self._immutable_porfolio

    @property
    def account(self):
        portfolio = self.portfolio
        account = self._account
        account.settled_cash = portfolio
        account.total_positions_values = portfolio.portfolio_value - portfolio.cash
        account.total_position_exposure = portfolio.positions_exposure
        account.cushion = portfolio.cash / portfolio.positions_value
        account.gross_leverage,account.net_leverage = self.calculate_period_stats()

    @property
    def todays_returns(self):
        if self._dirty_portfolio:
            raise Exception('today_returns is avaiable at the end of session ')
        return (
            (self.portfolio.returns +1) /
            (self._previous_total_returns +1 ) - 1
        )

    #计算账户当天的收益率
    def end_of_session(self,session_ix):
        """同步持仓的close价格"""
        self._sync_last_prices()
        self._update_porfolio()
        self._dirty_portfolio = False
        self.daily_returns_series[session_ix] = self.todays_returns

    @property
    def positions(self):
        return self.position_tracker.get_positions()

    def get_transactions(self, dt=None):
        if dt:
            return [
                txn
                for txn in self._processed_transaction
                if txn.dt == dt]
        return self._processed_transaction

    def manual_close_position(self,txns):
        """
            self.position_tracker.maybe_create_close_position_transaction
            self.process_transaction(txn)
        """
        self.position_tracker.maybe_create_close_position_transaction(txns)
