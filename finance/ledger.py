# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd , numpy as np,warnings


class MutableView(object):
    """A mutable view over an "immutable" object.

    Parameters
    ----------
    ob : any
        The object to take a view over.
    """
    # add slots so we don't accidentally add attributes to the view instead of
    # ``ob``
    __slots__ = ('_mutable_view_obj')

    def __init__(self,ob):
        object.__setattr__(self,'_mutable_view_ob',ob)

    def __getattr__(self, item):
        return getattr(self._mutable_view_ob,item)

    def __setattr(self,attr,value):
        #vars() 函数返回对象object的属性和属性值的字典对象 --- 扩展属性类型
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return '%s(%r)'%(type(self).__name__,self._mutable_view_ob)


class Account(object):
    """
        the account object tracks information about the trading account
    """
    def __init__(self):
        # 避免直接定义属性
        self_ = MutableView(self)
        self_.settle_cash = 0.0
        self_.buying_power = float('inf')
        self_.equity_with_loan = 0.0
        self_.total_position_value = 0.0
        self_.total_position_exposure = 0.0
        self_.cushion = 0.0
        self_.leverage = 0.0

    def __setattr__(self,attr,value):
        raise AttributeError

    def __repr__(self):
        return "Account ({0})".format(self.__dict__)


class Positions(dict):
    """A dict-like object containing the algorithm's current positions.
    __missing__ default
    """

    def __missing__(self, key):
        if isinstance(key, Asset):
            return Position(InnerPosition(key))
        elif isinstance(key, int):
            warnings("Referencing positions by integer is deprecated."
                 " Use an asset instead.")
        else:
            warnings("Position lookup expected a value of type Asset but got {0}"
                 " instead.".format(type(key).__name__))

        return _DeprecatedSidLookupPosition(key)


class Portfolio(object):
    """Object providing read-only access to current portfolio state.

    Parameters
    ----------
    start_date : pd.Timestamp
        The start date for the period being recorded.
    capital_base : float
        The starting value for the portfolio. This will be used as the starting
        cash, current cash, and portfolio value.

    Attributes
    ----------
    positions : zipline.protocol.Positions
        Dict-like object containing information about currently-held positions.
    cash : float
        Amount of cash currently held in portfolio.
    portfolio_value : float
        Current liquidation value of the portfolio's holdings.
        This is equal to ``cash + sum(shares * price)``
    starting_cash : float
        Amount of cash in the portfolio at the start of the backtest.
    """

    def __init__(self, start_date=None, capital_base=0.0):
        self_ = MutableView(self)
        self_.cash_flow = 0.0
        self_.starting_cash = capital_base
        self_.portfolio_value = capital_base
        self_.pnl = 0.0
        self_.returns = 0.0
        self_.cash = capital_base
        self_.positions = Positions()
        self_.start_date = start_date
        self_.positions_value = 0.0
        self_.positions_exposure = 0.0

    @property
    def capital_used(self):
        return self.cash_flow

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Portfolio objects')

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)

    @property
    def current_portfolio_weights(self):
        """
        Compute each asset's weight in the portfolio by calculating its held
        value divided by the total value of all positions.

        Each equity's value is its price times the number of shares held. Each
        futures contract's value is its unit price times number of shares held
        times the multiplier.
        """
        position_values = pd.Series({
            asset: (
                    position.last_sale_price *
                    position.amount *
                    asset.price_multiplier
            )
            for asset, position in self.positions.items()
        })
        return position_values / self.portfolio_value


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

    # cash_utilization = 1 - (cash_blance /capital_blance).mean()
