# -*- coding : utf-8 -*-
import logging,pandas as pd
from abc import ABC ,abstractmethod


class ZiplineError(Exception):
    msg = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @lazyval
    def message(self):
        return str(self)

    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg

    __unicode__ = __str__
    __repr__ = __str__


class TradingControlViolation(ZiplineError):
    """
    Raised if an order would violate a constraint set by a TradingControl.
    """
    msg = """
            Order for {amount} shares of {asset} at {datetime} violates trading constraint
            {constraint}.
        """.strip()


class TradingControl(ABC):
    """
        abstract base class represent a fail-safe control on the behavior of any algorithm
    """
    def __init__(self,on_error,**kwargs):

        self.on_error = on_error
        self._fail_args = kwargs

    @abstractmethod
    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        raise NotImplementedError

    def __repr__(self):
        return "{name}({attr})".format(name = self.__class__.__name__,
                                       attr = self._fail_args)

    def _constraint_msg(self,metadata = None):
        constraint = repr(self)
        if metadata :
            constraint = "{contraint}(Metadata:{metadata})".format(constraint,metadata)
        return constraint

    def handle_violation(self,asset,amount,datetime,metadata = None):
        constraint = self._constraint_msg(metadata)

        if self.on_error == 'fail':
            raise TradingControlViolation(
                asset = asset,
                amount = amount,
                datetime = datetime,
                constraint = constraint
            )
        elif self.on_error == 'log':
            logging.error('order for amount shares of asset at dt')


class LongOnly(TradingControl):

    def __init__(self,on_error):
        super(LongOnly,self).__init__(on_error)

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        asset = txn.asset
        amount = txn.amount
        if portfolio.positons[asset].amount + amount  < 0 :
            self.handle_violation(asset,amount,algo_datetime)


class RestrictedListOrder(TradingControl):
    """ represents a restricted list of assets that canont be ordered by the algorithm"""
    def __init__(self,on_error,restrictions):
        super(RestrictedListOrder,self).__init__(on_error)
        self.restrictions = restrictions

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        asset = txn.asset
        amount = txn.amount
        if self.restrictions.is_restricted(asset,algo_datetime):
            self.handle_violation(asset,amount,algo_datetime)


class AssetDateBounds(TradingControl):
    """
        prohibition against an asset before start_date or after end_date
    """
    def __init__(self,on_error):
        super(AssetDateBounds,self).__init__(on_error)

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        asset = txn.asset
        amount = txn.amount
        if amount == 0 :
            return

        normalized_algo_dt = pd.Timestamp(algo_datetime).normalize()

        if asset.start_date :
            normalized_start = pd.Timestamp(asset.start_date).normarlize()
            if normalized_start > normalized_algo_dt :
                metadata = {
                    'asset_start_date':normalized_start
                }
                self.handle_violation(asset,amount,algo_datetime,metadata = metadata)
        if asset.end_date:
            normalized_end = pd.Timestamp(asset.end_date).normalize()
            if normalized_end < normalized_algo_dt :
                metadata = {
                    'asset_end_date':normalized_end
                }
                self.handle_violation(asset,amount,algo_datetime,metadata = metadata)


class MaxPositionOrder(TradingControl):
    """represent a limit on the maximum position size that ca be held by an algo for a given asset
      股票最大持仓比例 asset -- sid reason
    """
    def __init__(self,on_error,
                 max_notional = None):

        self.on_error = on_error
        self.max_notional = max_notional

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        amount = txn.amount
        current_price = txn.price
        asset = txn.asset

        current_share_count = portfolio.positions[asset].amount
        share_post_order  = current_share_count + amount

        value_post_order_ratio = share_post_order * current_price / portfolio.portfolio_value

        too_many_value =  value_post_order_ratio > self.max_notional

        if too_many_value:
            self.handle_violation(asset,amount,algo_datetime)