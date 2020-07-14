# -*- coding : utf-8 -*-

import abc,enum,logging,operator,pandas as pd
from numpy import vectorize
from functools import partial, reduce
from six import with_metaclass, iteritems
from toolz import groupby
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


def vectorized_is_element(array, choices):
    """
    Check if each element of ``array`` is in choices.

    Parameters
    ----------
    array : np.ndarray
    choices : object
        Object implementing __contains__.

    Returns
    -------
    was_element : np.ndarray[bool]
        Array indicating whether each element of ``array`` was in ``choices``.
    """
    return vectorize(choices.__contains__, otypes=[bool])(array)


def _cleanup_expired_assets(self, dt, position_assets):
    """
    Clear out any assets that have expired before starting a new sim day.

    Performs two functions:

    1. Finds all assets for which we have open orders and clears any
       orders whose assets are on or after their auto_close_date.

    2. Finds all assets for which we have positions and generates
       close_position events for any assets that have reached their
       auto_close_date.
    """
    algo = self.algo

    def past_auto_close_date(asset):
        acd = asset.auto_close_date
        return acd is not None and acd <= dt

    # Remove positions in any sids that have reached their auto_close date.
    assets_to_clear = \
        [asset for asset in position_assets if past_auto_close_date(asset)]
    metrics_tracker = algo.metrics_tracker
    data_portal = self.data_portal
    for asset in assets_to_clear:
        metrics_tracker.process_close_position(asset, dt, data_portal)

    # Remove open orders for any sids that have reached their auto close
    # date. These orders get processed immediately because otherwise they
    # would not be processed until the first bar of the next day.
    blotter = algo.blotter
    assets_to_cancel = [
        asset for asset in blotter.open_orders
        if past_auto_close_date(asset)
    ]
    for asset in assets_to_cancel:
        blotter.cancel_all_orders_for_asset(asset)

    # Make a copy here so that we are not modifying the list that is being
    # iterated over.
    for order in copy(blotter.new_orders):
        if order.status == ORDER_STATUS.CANCELLED:
            metrics_tracker.process_order(order)
            blotter.new_orders.remove(order)


class Restrictions(with_metaclass(abc.ABCMeta)):
    """
    Abstract restricted list interface, representing a set of assets that an
    algorithm is restricted from trading.
    """

    @abc.abstractmethod
    def is_restricted(self, assets, dt):
        """
        Is the asset restricted (RestrictionStates.FROZEN) on the given dt?

        Parameters
        ----------
        asset : Asset of iterable of assets
            The asset(s) for which we are querying a restriction
        dt : pd.Timestamp
            The timestamp of the restriction query

        Returns
        -------
        is_restricted : bool or pd.Series[bool] indexed by asset
            Is the asset or assets restricted on this dt?

        """
        raise NotImplementedError('is_restricted')

    def __or__(self, other_restriction):
        """Base implementation for combining two restrictions.
        """
        # If the right side is a _UnionRestrictions, defers to the
        # _UnionRestrictions implementation of `|`, which intelligently
        # flattens restricted lists
        if isinstance(other_restriction, _UnionRestrictions):
            return other_restriction | self
        return _UnionRestrictions([self, other_restriction])


class _UnionRestrictions(Restrictions):
    """
    A union of a number of sub restrictions.

    Parameters
    ----------
    sub_restrictions : iterable of Restrictions (but not _UnionRestrictions)
        The Restrictions to be added together

    Notes
    -----
    - Consumers should not construct instances of this class directly, but
      instead use the `|` operator to combine restrictions
    """

    def __new__(cls, sub_restrictions):
        # Filter out NoRestrictions and deal with resulting cases involving
        # one or zero sub_restrictions
        sub_restrictions = [
            r for r in sub_restrictions if not isinstance(r, NoRestrictions)
        ]
        if len(sub_restrictions) == 0:
            return NoRestrictions()
        elif len(sub_restrictions) == 1:
            return sub_restrictions[0]

        new_instance = super(_UnionRestrictions, cls).__new__(cls)
        new_instance.sub_restrictions = sub_restrictions
        return new_instance

    def __or__(self, other_restriction):
        """
        Overrides the base implementation for combining two restrictions, of
        which the left side is a _UnionRestrictions.
        """
        # Flatten the underlying sub restrictions of _UnionRestrictions
        if isinstance(other_restriction, _UnionRestrictions):
            new_sub_restrictions = \
                self.sub_restrictions + other_restriction.sub_restrictions
        else:
            new_sub_restrictions = self.sub_restrictions + [other_restriction]

        return _UnionRestrictions(new_sub_restrictions)

    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return any(
                r.is_restricted(assets, dt) for r in self.sub_restrictions
            )

        #series --- bool --- operator or_
        return reduce(
            operator.or_,
            (r.is_restricted(assets, dt) for r in self.sub_restrictions)
        )


class NoRestrictions(Restrictions):
    """
    A no-op restrictions that contains no restrictions.
    """
    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return False
        return pd.Series(index=pd.Index(assets), data=False)


class StaticRestrictions(Restrictions):
    """
    Static restrictions stored in memory that are constant regardless of dt
    for each asset.

    Parameters
    ----------
    restricted_list : iterable of assets
        The assets to be restricted
    """

    def __init__(self, restricted_list):
        self._restricted_set = frozenset(restricted_list)

    def is_restricted(self, assets, dt):
        """
        An asset is restricted for all dts if it is in the static list.
        """
        if isinstance(assets, Asset):
            return assets in self._restricted_set
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, self._restricted_set)
        )

RESTRICTION_STATES = enum(
    'ALLOWED',
    'FROZEN',
)


class HistoricalRestrictions(Restrictions):
    """
    Historical restrictions stored in memory with effective dates for each
    asset.

    Parameters
    ----------
    restrictions : iterable of namedtuple Restriction
        The restrictions, each defined by an asset, effective date and state
    """

    def __init__(self, restrictions):
        # A dict mapping each asset to its restrictions, which are sorted by
        # ascending order of effective_date
        self._restrictions_by_asset = {
            asset: sorted(
                restrictions_for_asset, key=lambda x: x.effective_date
            )
            for asset, restrictions_for_asset
            in iteritems(groupby(lambda x: x.asset, restrictions))
        }

    def is_restricted(self, assets, dt):
        """
        Returns whether or not an asset or iterable of assets is restricted
        on a dt.
        """
        if isinstance(assets, Asset):
            return self._is_restricted_for_asset(assets, dt)

        is_restricted = partial(self._is_restricted_for_asset, dt=dt)
        return pd.Series(
            index=pd.Index(assets),
            data=vectorize(is_restricted, otypes=[bool])(assets)
        )

    def _is_restricted_for_asset(self, asset, dt):
        state = RESTRICTION_STATES.ALLOWED
        for r in self._restrictions_by_asset.get(asset, ()):
            if r.effective_date > dt:
                break
            state = r.state
        return state == RESTRICTION_STATES.FROZEN


class SecurityListRestrictions(Restrictions):
    """
    Restrictions based on a security list.

    Parameters
    ----------
    restrictions : zipline.utils.security_list.SecurityList
        The restrictions defined by a SecurityList
    """

    def __init__(self, security_list_by_dt):
        self.current_securities = security_list_by_dt.current_securities

    def is_restricted(self, assets, dt):
        securities_in_list = self.current_securities(dt)
        if isinstance(assets, Asset):
            return assets in securities_in_list
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, securities_in_list)
        )


