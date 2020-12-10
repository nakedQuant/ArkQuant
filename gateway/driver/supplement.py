# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import OrderedDict
import numpy as np, pandas as pd

_MINUTE_TO_SESSION_OHCLV_HOW = OrderedDict((
    ('open', 'first'),
    ('high', 'max'),
    ('low', 'min'),
    ('close', 'last'),
    ('volume', 'sum'),
))


def minute_frame_to_session_frame(minute_frame, calendar):

    """
    Resample a DataFrame with minute data into the frame expected by a
    BcolzDailyBarWriter.

    Parameters
    ----------
    minute_frame : pd.DataFrame
        A DataFrame with the columns `open`, `high`, `low`, `close`, `volume`,
        and `dt` (minute dts)
    calendar : trading_calendars.trading_calendar.TradingCalendar
        A TradingCalendar on which session labels to resample from minute
        to session.

    Return
    ------
    session_frame : pd.DataFrame
        A DataFrame with the columns `open`, `high`, `low`, `close`, `volume`,
        and `day` (datetime-like).
    """
    how = OrderedDict((c, _MINUTE_TO_SESSION_OHCLV_HOW[c])
                      for c in minute_frame.columns)
    # index --- session lable : ticker
    labels = calendar.minute_index_to_session_labels(minute_frame.index)
    # how --- field : func(string)
    return minute_frame.groupby(labels).agg(how)


class DailyHistoryAggregator(object):
    """
    Converts minute pricing data into a daily summary, to be used for the
    last slot in a call to history with a frequency of `1d`.

    This summary is the same as a daily bar rollup of minute data, with the
    distinction that the summary is truncated to the `dt` requested.
    i.e. the aggregation slides forward during a the course of ArkQuant day.

    Provides aggregation for `open`, `high`, `low`, `close`, and `volume`.
    The aggregation rules for each price type is documented in their respective

    """

    def __init__(self, market_opens, minute_reader, trading_calendar):
        self._market_opens = market_opens
        self._minute_reader = minute_reader
        self._trading_calendar = trading_calendar

        # The caches are structured as (date, market_open, entries), where
        # entries is a dict of asset -> (last_visited_dt, value)
        #
        # Whenever an aggregation method determines the current value,
        # the entry for the respective asset should be overwritten with a new
        # entry for the current dt.value (int) and aggregation value.
        #
        # When the requested dt's date is different from date the cache is
        # flushed, so that the cache entries do not grow unbounded.
        #
        # Example cache:
        # cache = (date(2016, 3, 17),
        #          pd.Timestamp('2016-03-17 13:31', tz='UTC'),
        #          {
        #              1: (1458221460000000000, np.nan),
        #              2: (1458221460000000000, 42.0),
        #         })
        self._caches = {
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': None
        }

        # The int value is used for deltas to avoid extra computation from
        # creating new Timestamps.
        self._one_min = pd.Timedelta('1 min').value

    def _prelude(self, dt, field):
        session = self._trading_calendar.minute_to_session_label(dt)
        dt_value = dt.value
        cache = self._caches[field]
        if cache is None or cache[0] != session:
            market_open = self._market_opens.loc[session]
            cache = self._caches[field] = (session, market_open, {})

        _, market_open, entries = cache
        market_open = market_open.tz_localize('UTC')
        if dt != market_open:
            prev_dt = dt_value - self._one_min
        else:
            prev_dt = None
        return market_open, prev_dt, dt_value, entries

    def opens(self, assets, dt):
        """
        The open field's aggregation returns the first value that occurs
        for the day, if there has been no data on or before the `dt` the open
        is `nan`.

        Once the first non-nan open is seen, that value remains constant per
        asset for the remainder of the day.

        Returns
        -------
        np.array with dtype=float64, in order of asset parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'open')

        opens = []
        session_label = self._trading_calendar.minute_to_session_label(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                opens.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'open')
                entries[asset] = (dt_value, val)
                opens.append(val)
                continue
            else:
                try:
                    last_visited_dt, first_open = entries[asset]
                    if last_visited_dt == dt_value:
                        opens.append(first_open)
                        continue
                    elif not pd.isnull(first_open):
                        opens.append(first_open)
                        entries[asset] = (dt_value, first_open)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['open'],
                            after_last,
                            dt,
                            [asset],
                        )[0]
                        nonnan = window[~pd.isnull(window)]
                        if len(nonnan):
                            val = nonnan[0]
                        else:
                            val = np.nan
                        entries[asset] = (dt_value, val)
                        opens.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['open'],
                        market_open,
                        dt,
                        [asset],
                    )[0]
                    nonnan = window[~pd.isnull(window)]
                    if len(nonnan):
                        val = nonnan[0]
                    else:
                        val = np.nan
                    entries[asset] = (dt_value, val)
                    opens.append(val)
                    continue
        return np.array(opens)

    def highs(self, assets, dt):
        """
        The high field's aggregation returns the largest high seen between
        the market open and the current dt.
        If there has been no data on or before the `dt` the high is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of asset parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'high')

        highs = []
        session_label = self._trading_calendar.minute_to_session_label(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                highs.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'high')
                entries[asset] = (dt_value, val)
                highs.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_max = entries[asset]
                    if last_visited_dt == dt_value:
                        highs.append(last_max)
                        continue
                    elif last_visited_dt == prev_dt:
                        curr_val = self._minute_reader.get_value(
                            asset, dt, 'high')
                        if pd.isnull(curr_val):
                            val = last_max
                        elif pd.isnull(last_max):
                            val = curr_val
                        else:
                            val = max(last_max, curr_val)
                        entries[asset] = (dt_value, val)
                        highs.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['high'],
                            after_last,
                            dt,
                            [asset],
                        )[0].T
                        val = np.nanmax(np.append(window, last_max))
                        entries[asset] = (dt_value, val)
                        highs.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['high'],
                        market_open,
                        dt,
                        [asset],
                    )[0].T
                    val = np.nanmax(window)
                    entries[asset] = (dt_value, val)
                    highs.append(val)
                    continue
        return np.array(highs)

    def lows(self, assets, dt):
        """
        The low field's aggregation returns the smallest low seen between
        the market open and the current dt.
        If there has been no data on or before the `dt` the low is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of asset parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'low')

        lows = []
        session_label = self._trading_calendar.minute_to_session_label(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                lows.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'low')
                entries[asset] = (dt_value, val)
                lows.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_min = entries[asset]
                    if last_visited_dt == dt_value:
                        lows.append(last_min)
                        continue
                    elif last_visited_dt == prev_dt:
                        curr_val = self._minute_reader.get_value(
                            asset, dt, 'low')
                        val = np.nanmin([last_min, curr_val])
                        entries[asset] = (dt_value, val)
                        lows.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['low'],
                            after_last,
                            dt,
                            [asset],
                        )[0].T
                        val = np.nanmin(np.append(window, last_min))
                        entries[asset] = (dt_value, val)
                        lows.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['low'],
                        market_open,
                        dt,
                        [asset],
                    )[0].T
                    val = np.nanmin(window)
                    entries[asset] = (dt_value, val)
                    lows.append(val)
                    continue
        return np.array(lows)

    def closes(self, assets, dt):
        """
        The close field's aggregation returns the latest close at the given
        dt.
        If the close for the given dt is `nan`, the most recent non-nan
        `close` is used.
        If there has been no data on or before the `dt` the close is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of asset parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'close')

        closes = []
        session_label = self._trading_calendar.minute_to_session_label(dt)

        def _get_filled_close(asset):
            """
            Returns the most recent non-nan close for the asset in this
            session. If there has been no data in this session on or before the
            `dt`, returns `nan`
            """
            window = self._minute_reader.load_raw_arrays(
                ['close'],
                market_open,
                dt,
                [asset],
            )[0]
            try:
                return window[~np.isnan(window)][-1]
            except IndexError:
                return np.NaN

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                closes.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'close')
                entries[asset] = (dt_value, val)
                closes.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_close = entries[asset]
                    if last_visited_dt == dt_value:
                        closes.append(last_close)
                        continue
                    elif last_visited_dt == prev_dt:
                        val = self._minute_reader.get_value(
                            asset, dt, 'close')
                        if pd.isnull(val):
                            val = last_close
                        entries[asset] = (dt_value, val)
                        closes.append(val)
                        continue
                    else:
                        val = self._minute_reader.get_value(
                            asset, dt, 'close')
                        if pd.isnull(val):
                            val = _get_filled_close(asset)
                        entries[asset] = (dt_value, val)
                        closes.append(val)
                        continue
                except KeyError:
                    val = self._minute_reader.get_value(
                        asset, dt, 'close')
                    if pd.isnull(val):
                        val = _get_filled_close(asset)
                    entries[asset] = (dt_value, val)
                    closes.append(val)
                    continue
        return np.array(closes)

    def volumes(self, assets, dt):
        """
        The volume field's aggregation returns the sum of all volumes
        between the market open and the `dt`
        If there has been no data on or before the `dt` the volume is 0.

        Returns
        -------
        np.array with dtype=int64, in order of asset parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'volume')

        volumes = []
        session_label = self._trading_calendar.minute_to_session_label(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                volumes.append(0)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'volume')
                entries[asset] = (dt_value, val)
                volumes.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_total = entries[asset]
                    if last_visited_dt == dt_value:
                        volumes.append(last_total)
                        continue
                    elif last_visited_dt == prev_dt:
                        val = self._minute_reader.get_value(
                            asset, dt, 'volume')
                        val += last_total
                        entries[asset] = (dt_value, val)
                        volumes.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['volume'],
                            after_last,
                            dt,
                            [asset],
                        )[0]
                        val = np.nansum(window) + last_total
                        entries[asset] = (dt_value, val)
                        volumes.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['volume'],
                        market_open,
                        dt,
                        [asset],
                    )[0]
                    val = np.nansum(window)
                    entries[asset] = (dt_value, val)
                    volumes.append(val)
                    continue
        return np.array(volumes)



# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import re
from itertools import chain
from numbers import Number

import numexpr
from numexpr.necompiler import getExprNames
from numpy import (
    full,
    inf,
)


_VARIABLE_NAME_RE = re.compile("^(x_)([0-9]+)$")

# Map from op symbol to equivalent Python magic method name.
ops_to_methods = {
    '+': '__add__',
    '-': '__sub__',
    '*': '__mul__',
    '/': '__div__',
    '%': '__mod__',
    '**': '__pow__',
    '&': '__and__',
    '|': '__or__',
    '^': '__xor__',
    '<': '__lt__',
    '<=': '__le__',
    '==': '__eq__',
    '!=': '__ne__',
    '>=': '__ge__',
    '>': '__gt__',
}
# Map from method name to op symbol.
methods_to_ops = {v: k for k, v in ops_to_methods.items()}

# Map from op symbol to equivalent Python magic method name after flipping
# arguments.
ops_to_commuted_methods = {
    '+': '__radd__',
    '-': '__rsub__',
    '*': '__rmul__',
    '/': '__rdiv__',
    '%': '__rmod__',
    '**': '__rpow__',
    '&': '__rand__',
    '|': '__ror__',
    '^': '__rxor__',
    '<': '__gt__',
    '<=': '__ge__',
    '==': '__eq__',
    '!=': '__ne__',
    '>=': '__le__',
    '>': '__lt__',
}
unary_ops_to_methods = {
    '-': '__neg__',
    '~': '__invert__',
}

UNARY_OPS = {'-'}
MATH_BINOPS = {'+', '-', '*', '/', '**', '%'}
FILTER_BINOPS = {'&', '|'}  # NumExpr doesn't support xor.
COMPARISONS = {'<', '<=', '!=', '>=', '>', '=='}

NUMEXPR_MATH_FUNCS = {
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'log',
    'log10',
    'log1p',
    'exp',
    'expm1',
    'sqrt',
    'abs',
}


def _ensure_element(tup, elem):
    """
    Create a tuple containing all elements of tup, plus elem.

    Returns the new tuple and the index of elem in the new tuple.
    """
    try:
        return tup, tup.index(elem)
    except ValueError:
        return tuple(chain(tup, (elem,))), len(tup)


class BadBinaryOperator(TypeError):
    """
    Called when a bad binary operation is encountered.

    Parameters
    ----------
    op : str
        The attempted operation
    left : zipline.computable.Term
        The left hand side of the operation.
    right : zipline.computable.Term
        The right hand side of the operation.
    """
    def __init__(self, op, left, right):
        super(BadBinaryOperator, self).__init__(
            "Can't compute {left} {op} {right}".format(
                op=op,
                left=type(left).__name__,
                right=type(right).__name__,
            )
        )


def method_name_for_op(op, commute=False):
    """
    Get the name of the Python magic method corresponding to `op`.

    Parameters
    ----------
    op : str {'+','-','*', '/','**','&','|','^','<','<=','==','!=','>=','>'}
        The requested operation.
    commute : bool
        Whether to return the name of an equivalent method after flipping args.

    Returns
    -------
    method_name : str
        The name of the Python magic method corresponding to `op`.
        If `commute` is True, returns the name of a method equivalent to `op`
        with inputs flipped.

    Examples
    --------
    >>> method_name_for_op('+')
    '__add__'
    >>> method_name_for_op('+', commute=True)
    '__radd__'
    >>> method_name_for_op('>')
    '__gt__'
    >>> method_name_for_op('>', commute=True)
    '__lt__'
    """
    if commute:
        return ops_to_commuted_methods[op]
    return ops_to_methods[op]


def unary_op_name(op):
    return unary_ops_to_methods[op]


def is_comparison(op):
    return op in COMPARISONS


class NumericalExpression(object):
    """
    Term binding to a numexpr expression.

    Parameters
    ----------
    expr : string
        A string suitable for passing to numexpr.  All variables in 'expr'
        should be of the form "x_i", where i is the index of the corresponding
        factor input in 'binds'.
    binds : tuple
        A tuple of factors to use as inputs.
    dtype : np.dtype
        The dtype for the expression.
    """
    window_length = 0

    def __new__(cls, expr, binds, dtype):
        # We always allow filters to be used in windowed computations.
        # Otherwise, an expression is window_safe if all its constituents are
        # window_safe.
        window_safe = (
            (dtype == bool_dtype) or all(t.window_safe for t in binds)
        )

        return super(NumericalExpression, cls).__new__(
            cls,
            inputs=binds,
            expr=expr,
            dtype=dtype,
            window_safe=window_safe,
        )

    def _init(self, expr, *args, **kwargs):
        self._expr = expr
        return super(NumericalExpression, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, expr, *args, **kwargs):
        return (
            super(NumericalExpression, cls)._static_identity(*args, **kwargs),
            expr,
        )

    def _validate(self):
        """
        Ensure that our expression string has variables of the form x_0, x_1,
        ... x_(N - 1), where N is the length of our inputs.
        """
        variable_names, _unused = getExprNames(self._expr, {})
        expr_indices = []
        for name in variable_names:
            if name == 'inf':
                continue
            match = _VARIABLE_NAME_RE.match(name)
            if not match:
                raise ValueError("%r is not a valid variable name" % name)
            expr_indices.append(int(match.group(2)))

        expr_indices.sort()
        expected_indices = list(range(len(self.inputs)))
        if expr_indices != expected_indices:
            raise ValueError(
                "Expected %s for variable indices, but got %s" % (
                    expected_indices, expr_indices,
                )
            )
        super(NumericalExpression, self)._validate()

    def _compute(self, arrays, dates, assets, mask):
        """
        Compute our stored expression string with numexpr.
        """
        out = full(mask.shape, self.missing_value, dtype=self.dtype)
        # This writes directly into our output buffer.
        numexpr.evaluate(
            self._expr,
            local_dict={
                "x_%d" % idx: array
                for idx, array in enumerate(arrays)
            },
            global_dict={'inf': inf},
            out=out,
        )
        return out

    def _rebind_variables(self, new_inputs):
        """
        Return self._expr with all variables rebound to the indices implied by
        new_inputs.
        """
        expr = self._expr

        # If we have 11+ variables, some of our variable names may be
        # substrings of other variable names. For example, we might have x_1,
        # x_10, and x_100. By enumerating in reverse order, we ensure that
        # every variable name which is a substring of another variable name is
        # processed after the variable of which it is a substring. This
        # guarantees that the substitution of any given variable index only
        # ever affects exactly its own index. For example, if we have variables
        # with indices going up to 100, we will process all of the x_1xx names
        # before x_1x, which will be before x_1, so the substitution of x_1
        # will not affect x_1x, which will not affect x_1xx.
        for idx, input_ in reversed(list(enumerate(self.inputs))):
            old_varname = "x_%d" % idx
            # Temporarily rebind to x_temp_N so that we don't overwrite the
            # same value multiple times.
            temp_new_varname = "x_temp_%d" % new_inputs.index(input_)
            expr = expr.replace(old_varname, temp_new_varname)
        # Clear out the temp variables now that we've finished iteration.
        return expr.replace("_temp_", "_")

    def _merge_expressions(self, other):
        """
        Merge the inputs of two NumericalExpressions into a single input tuple,
        rewriting their respective string expressions to make input names
        resolve correctly.

        Returns a tuple of (new_self_expr, new_other_expr, new_inputs)
        """
        new_inputs = tuple(set(self.inputs).union(other.inputs))
        new_self_expr = self._rebind_variables(new_inputs)
        new_other_expr = other._rebind_variables(new_inputs)
        return new_self_expr, new_other_expr, new_inputs

    def build_binary_op(self, op, other):
        """
        Compute new expression strings and a new inputs tuple for combining
        self and other with a binary operator.
        """
        if isinstance(other, NumericalExpression):
            self_expr, other_expr, new_inputs = self._merge_expressions(other)
        elif isinstance(other, Term):
            self_expr = self._expr
            new_inputs, other_idx = _ensure_element(self.inputs, other)
            other_expr = "x_%d" % other_idx
        elif isinstance(other, Number):
            self_expr = self._expr
            other_expr = str(other)
            new_inputs = self.inputs
        else:
            raise BadBinaryOperator(op, other)
        return self_expr, other_expr, new_inputs

    @property
    def bindings(self):
        return {
            "x_%d" % i: input_
            for i, input_ in enumerate(self.inputs)
        }

    def __repr__(self):
        return "{typename}(expr='{expr}', bindings={bindings})".format(
            typename=type(self).__name__,
            expr=self._expr,
            bindings=self.bindings,
        )

    def graph_repr(self):
        """Short repr to use when rendering Pipeline graphs."""

        # Replace any floating point numbers in the expression
        # with their scientific notation
        final = re.sub(r"[-+]?\d*\.\d+",
                       lambda x: format(float(x.group(0)), '.2E'),
                       self._expr)
        # Graphviz interprets `\l` as "divide label into lines, left-justified"
        return "Expression:\\l  {}\\l".format(
            final,
        )



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from scipy import optimize
from math import pow
from sys import float_info
import numpy as np, pandas as pd


def gpd_risk_estimates(returns, var_p=0.01):
    """Estimate VaR and ES using the Generalized Pareto Distribution (GPD)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    var_p : float
        The percentile to use for estimating the VaR and ES

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]
        : list[float]
        threshold - the threshold use to cut off exception tail losses
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance)
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution)
        var_estimate - an estimate for the VaR for the given percentile
        es_estimate - an estimate for the ES for the given percentile

    Note
    ----
    seealso::
    `An Application of Extreme Value Theory for
Measuring Risk <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
        A paper describing how to use the Generalized Pareto
        Distribution to estimate VaR and ES.
    """
    if len(returns) < 3:
        result = np.zeros(5)
        if isinstance(returns, pd.Series):
            result = pd.Series(result)
        return result
    return gpd_risk_estimates_aligned(*_aligned_series(returns, var_p))


def gpd_risk_estimates_aligned(returns, var_p=0.01):
    """Estimate VaR and ES using the Generalized Pareto Distribution (GPD)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    var_p : float
        The percentile to use for estimating the VaR and ES

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]
        : list[float]
        threshold - the threshold use to cut off exception tail losses
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance)
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution)
        var_estimate - an estimate for the VaR for the given percentile
        es_estimate - an estimate for the ES for the given percentile

    Note
    ----
    seealso::
    `An Application of Extreme Value Theory for
Measuring Risk <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
        A paper describing how to use the Generalized Pareto
        Distribution to estimate VaR and ES.
    """
    result = np.zeros(5)
    if not len(returns) < 3:

        DEFAULT_THRESHOLD = 0.2
        MINIMUM_THRESHOLD = 0.000000001
        returns_array = pd.Series(returns).as_matrix()
        flipped_returns = -1 * returns_array
        losses = flipped_returns[flipped_returns > 0]
        threshold = DEFAULT_THRESHOLD
        finished = False
        scale_param = 0
        shape_param = 0
        while not finished and threshold > MINIMUM_THRESHOLD:
            losses_beyond_threshold = \
                losses[losses >= threshold]
            param_result = \
                gpd_loglikelihood_minimizer_aligned(losses_beyond_threshold)
            if (param_result[0] is not False and
                    param_result[1] is not False):
                scale_param = param_result[0]
                shape_param = param_result[1]
                var_estimate = gpd_var_calculator(threshold, scale_param,
                                                  shape_param, var_p,
                                                  len(losses),
                                                  len(losses_beyond_threshold))
                # non-negative shape parameter is required for fat tails
                # non-negative VaR estimate is required for loss of some kind
                if shape_param > 0 and var_estimate > 0:
                    finished = True
            if not finished:
                threshold = threshold / 2
        if finished:
            es_estimate = gpd_es_calculator(var_estimate, threshold,
                                            scale_param, shape_param)
            result = np.array([threshold, scale_param, shape_param,
                               var_estimate, es_estimate])
    if isinstance(returns, pd.Series):
        result = pd.Series(result)
    return result


def gpd_es_calculator(var_estimate, threshold, scale_param,
                      shape_param):
    result = 0
    if ((1 - shape_param) != 0):
        # this formula is from Gilli and Kellezi pg. 8
        var_ratio = (var_estimate/(1 - shape_param))
        param_ratio = ((scale_param - (shape_param * threshold)) /
                       (1 - shape_param))
        result = var_ratio + param_ratio
    return result


def gpd_var_calculator(threshold, scale_param, shape_param,
                       probability, total_n, exceedance_n):
    result = 0
    if (exceedance_n > 0 and shape_param > 0):
        # this formula is from Gilli and Kellezi pg. 12
        param_ratio = scale_param / shape_param
        prob_ratio = (total_n/exceedance_n) * probability
        result = threshold + (param_ratio *
                              (pow(prob_ratio, -shape_param) - 1))
    return result


def gpd_loglikelihood_minimizer_aligned(price_data):
    result = [False, False]
    DEFAULT_SCALE_PARAM = 1
    DEFAULT_SHAPE_PARAM = 1
    if len(price_data) > 0:
        gpd_loglikelihood_lambda = \
            gpd_loglikelihood_factory(price_data)
        optimization_results = \
            optimize.minimize(gpd_loglikelihood_lambda,
                              [DEFAULT_SCALE_PARAM,
                               DEFAULT_SHAPE_PARAM],
                              method='Nelder-Mead')
        if optimization_results.success:
            resulting_params = optimization_results.x
            if len(resulting_params) == 2:
                result[0] = resulting_params[0]
                result[1] = resulting_params[1]
    return result


def gpd_loglikelihood_factory(price_data):
    return lambda params: gpd_loglikelihood(params, price_data)


def gpd_loglikelihood(params, price_data):
    if (params[1] != 0):
        return -gpd_loglikelihood_scale_and_shape(params[0],
                                                  params[1],
                                                  price_data)
    else:
        return -gpd_loglikelihood_scale_only(params[0], price_data)


def gpd_loglikelihood_scale_and_shape_factory(price_data):
    # minimize a function of two variables requires a list of params
    # we are expecting the lambda below to be called as follows:
    # parameters = [scale, shape]
    # the final outer negative is added because scipy only minimizes
    return lambda params: \
        -gpd_loglikelihood_scale_and_shape(params[0],
                                           params[1],
                                           price_data)


def gpd_loglikelihood_scale_and_shape(scale, shape, price_data):
    n = len(price_data)
    result = -1 * float_info.max
    if (scale != 0):
        param_factor = shape / scale
        if (shape != 0 and param_factor >= 0 and scale >= 0):
            result = ((-n * np.log(scale)) -
                      (((1 / shape) + 1) *
                       (np.log((shape / scale * price_data) + 1)).sum()))
    return result


def gpd_loglikelihood_scale_only_factory(price_data):
    # the negative is added because scipy only minimizes
    return lambda scale: \
        -gpd_loglikelihood_scale_only(scale, price_data)


def gpd_loglikelihood_scale_only(scale, price_data):
    n = len(price_data)
    data_sum = price_data.sum()
    result = -1 * float_info.max
    if scale >= 0:
        result = ((-n*np.log(scale)) - (data_sum/scale))
    return result


#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from warnings import warn

import pandas as pd

from .assets import Asset
from .utils.enum import enum
from finance._protocol import BarData, InnerPosition


class MutableView(object):
    """A mutable view over an "immutable" object.

    Parameters
    ----------
    ob : any
        The object to take a view over.
    """
    # add slots so we don't accidentally add attributes to the view instead of
    # ``ob``
    __slots__ = ('_mutable_view_ob',)

    def __init__(self, ob):
        object.__setattr__(self, '_mutable_view_ob', ob)

    def __getattr__(self, attr):
        return getattr(self._mutable_view_ob, attr)

    def __setattr__(self, attr, value):
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self._mutable_view_ob)


# Datasource type should completely determine the other fields of a
# message with its type.
DATASOURCE_TYPE = enum(
    'AS_TRADED_EQUITY',
    'MERGER',
    'SPLIT',
    'DIVIDEND',
    'TRADE',
    'TRANSACTION',
    'ORDER',
    'EMPTY',
    'DONE',
    'CUSTOM',
    'BENCHMARK',
    'COMMISSION',
    'CLOSE_POSITION'
)

# Expected fields/index values for a dividend Series.
DIVIDEND_FIELDS = [
    'declared_date',
    'ex_date',
    'gross_amount',
    'net_amount',
    'pay_date',
    'payment_sid',
    'ratio',
    'sid',
]
# Expected fields/index values for a dividend payment Series.
DIVIDEND_PAYMENT_FIELDS = [
    'id',
    'payment_sid',
    'cash_amount',
    'share_count',
]


class Event(object):

    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__.update(initial_values)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self, other):
        return hasattr(other, '__dict__') and self.__dict__ == other.__dict__

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)

    def to_series(self, index=None):
        return pd.Series(self.__dict__, index=index)


def _deprecated_getitem_method(name, attrs):
    """Create a deprecated ``__getitem__`` method that tells users to use
    getattr instead.

    Parameters
    ----------
    name : str
        The name of the object in the warning message.
    attrs : iterable[str]
        The set of allowed attributes.

    Returns
    -------
    __getitem__ : callable[any, str]
        The ``__getitem__`` method to put in the class dict.
    """
    attrs = frozenset(attrs)
    msg = (
        "'{name}[{attr!r}]' is deprecated, please use"
        " '{name}.{attr}' instead"
    )

    def __getitem__(self, key):
        """``__getitem__`` is deprecated, please use attribute access instead.
        """
        warn(msg.format(name=name, attr=key), DeprecationWarning, stacklevel=2)
        if key in attrs:
            return getattr(self, key)
        raise KeyError(key)

    return __getitem__


class Order(Event):
    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'order', {
            'dt',
            'sid',
            'amount',
            'stop',
            'limit',
            'id',
            'filled',
            'commission',
            'stop_reached',
            'limit_reached',
            'created',
        },
    )


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

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'portfolio', {
            'capital_used',
            'starting_cash',
            'portfolio_value',
            'pnl',
            'returns',
            'cash',
            'positions',
            'start_date',
            'positions_value',
        },
    )

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


class Account(object):
    """
    The account object tracks information about the trading account. The
    values are updated as the algorithm runs and its keys remain unchanged.
    If connected to a broker, one can update these values with the trading
    account values as reported by the broker.
    """

    def __init__(self):
        self_ = MutableView(self)
        self_.settled_cash = 0.0
        self_.accrued_interest = 0.0
        self_.buying_power = float('inf')
        self_.equity_with_loan = 0.0
        self_.total_positions_value = 0.0
        self_.total_positions_exposure = 0.0
        self_.regt_equity = 0.0
        self_.regt_margin = float('inf')
        self_.initial_margin_requirement = 0.0
        self_.maintenance_margin_requirement = 0.0
        self_.available_funds = 0.0
        self_.excess_liquidity = 0.0
        self_.cushion = 0.0
        self_.day_trades_remaining = float('inf')
        self_.leverage = 0.0
        self_.net_leverage = 0.0
        self_.net_liquidation = 0.0

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Account objects')

    def __repr__(self):
        return "Account({0})".format(self.__dict__)

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'account', {
            'settled_cash',
            'accrued_interest',
            'buying_power',
            'equity_with_loan',
            'total_positions_value',
            'total_positions_exposure',
            'regt_equity',
            'regt_margin',
            'initial_margin_requirement',
            'maintenance_margin_requirement',
            'available_funds',
            'excess_liquidity',
            'cushion',
            'day_trades_remaining',
            'leverage',
            'net_leverage',
            'net_liquidation',
        },
    )


class Position(object):
    """
    A position held by an algorithm.

    Attributes
    ----------
    asset : zipline.assets.Asset
        The held asset.
    amount : int
        Number of shares held. Short positions are represented with negative
        values.
    cost_basis : float
        Average price at which currently-held shares were acquired.
    last_sale_price : float
        Most recent price for the position.
    last_sale_date : pd.Timestamp
        Datetime at which ``last_sale_price`` was last updated.
    """
    __slots__ = ('_underlying_position',)

    def __init__(self, underlying_position):
        object.__setattr__(self, '_underlying_position', underlying_position)

    def __getattr__(self, attr):
        return getattr(self._underlying_position, attr)

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Position objects')

    @property
    def sid(self):
        # for backwards compatibility
        return self.asset

    def __repr__(self):
        return 'Position(%r)' % {
            k: getattr(self, k)
            for k in (
                'asset',
                'amount',
                'cost_basis',
                'last_sale_price',
                'last_sale_date',
            )
        }

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'position', {
            'sid',
            'amount',
            'cost_basis',
            'last_sale_price',
            'last_sale_date',
        },
    )


# Copied from Position and renamed.  This is used to handle cases where a user
# does something like `context.portfolio.positions[100]` instead of
# `context.portfolio.positions[sid(100)]`.
class _DeprecatedSidLookupPosition(object):
    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0
        self.last_sale_date = None

    def __repr__(self):
        return "_DeprecatedSidLookupPosition({0})".format(self.__dict__)

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'position', {
            'sid',
            'amount',
            'cost_basis',
            'last_sale_price',
            'last_sale_date',
        },
    )


class Positions(dict):
    """A dict-like object containing the algorithm's current positions.
    """

    def __missing__(self, key):
        if isinstance(key, Asset):
            return Position(InnerPosition(key))
        elif isinstance(key, int):
            warn("Referencing positions by integer is deprecated."
                 " Use an asset instead.")
        else:
            warn("Position lookup expected a value of type Asset but got {0}"
                 " instead.".format(type(key).__name__))

        return _DeprecatedSidLookupPosition(key)


# database
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
# 类型检查
from typing import Optional, Sequence, List, Dict, TYPE_CHECKING


class Driver(Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class BaseDatabaseManager(ABC):

    @abstractmethod
    def load_bar_data(
        self,
        symbol: str,
        exchange: "Exchange",
        interval: "Interval",
        start: datetime,
        end: datetime
    ) -> Sequence["BarData"]:
        pass

    @abstractmethod
    def load_tick_data(
        self,
        symbol: str,
        exchange: "Exchange",
        start: datetime,
        end: datetime
    ) -> Sequence["TickData"]:
        pass

    @abstractmethod
    def save_bar_data(
        self,
        datas: Sequence["BarData"],
    ):
        pass

    @abstractmethod
    def save_tick_data(
        self,
        datas: Sequence["TickData"],
    ):
        pass

    @abstractmethod
    def get_newest_bar_data(
        self,
        symbol: str,
        exchange: "Exchange",
        interval: "Interval"
    ) -> Optional["BarData"]:
        """
        If there is data in database, return the one with greatest datetime(newest one)
        otherwise, return None
        """
        pass

    @abstractmethod
    def get_oldest_bar_data(
        self,
        symbol: str,
        exchange: "Exchange",
        interval: "Interval"
    ) -> Optional["BarData"]:
        """
        If there is data in database, return the one with smallest datetime(oldest one)
        otherwise, return None
        """
        pass

    @abstractmethod
    def get_newest_tick_data(
        self,
        symbol: str,
        exchange: "Exchange",
    ) -> Optional["TickData"]:
        """
        If there is data in database, return the one with greatest datetime(newest one)
        otherwise, return None
        """
        pass

    @abstractmethod
    def get_bar_data_statistics(
        self,
        symbol: str,
        exchange: "Exchange",
    ) -> List[Dict]:
        """
        Return data avaible in database with a list of symbol/exchange/interval/count.
        """
        pass

    @abstractmethod
    def clean(self, symbol: str):
        """
        delete all records for a symbol
        """
        pass


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Sequence, List

from mongoengine import DateTimeField, Document, FloatField, StringField, connect

# from vnpy.trader.constant import Exchange, Interval
# from vnpy.trader.object import BarData, TickData
# from .database import BaseDatabaseManager, Driver


def init(_: Driver, settings: dict):
    database = settings["database"]
    host = settings["host"]
    port = settings["port"]
    username = settings["user"]
    password = settings["password"]
    authentication_source = settings["authentication_source"]

    if not username:  # if username == '' or None, skip username
        username = None
        password = None
        authentication_source = None

    connect(
        db=database,
        host=host,
        port=port,
        username=username,
        password=password,
        authentication_source=authentication_source,
    )

    return MongoManager()


class DbBarData(Document):
    """
    Candlestick bar data for database storage.

    Index is defined unique with datetime, interval, symbol
    """

    symbol: str = StringField()
    exchange: str = StringField()
    datetime: datetime = DateTimeField()
    interval: str = StringField()

    volume: float = FloatField()
    open_interest: float = FloatField()
    open_price: float = FloatField()
    high_price: float = FloatField()
    low_price: float = FloatField()
    close_price: float = FloatField()

    meta = {
        "indexes": [
            {
                "fields": ("symbol", "exchange", "interval", "datetime"),
                "unique": True,
            }
        ]
    }

    @staticmethod
    def from_bar(bar: BarData):
        """
        Generate DbBarData object from BarData.
        """
        db_bar = DbBarData()

        db_bar.symbol = bar.symbol
        db_bar.exchange = bar.exchange.value
        db_bar.datetime = bar.datetime
        db_bar.interval = bar.interval.value
        db_bar.volume = bar.volume
        db_bar.open_interest = bar.open_interest
        db_bar.open_price = bar.open_price
        db_bar.high_price = bar.high_price
        db_bar.low_price = bar.low_price
        db_bar.close_price = bar.close_price

        return db_bar

    def to_bar(self):
        """
        Generate BarData object from DbBarData.
        """
        bar = BarData(
            symbol=self.symbol,
            exchange=Exchange(self.exchange),
            datetime=self.datetime,
            interval=Interval(self.interval),
            volume=self.volume,
            open_interest=self.open_interest,
            open_price=self.open_price,
            high_price=self.high_price,
            low_price=self.low_price,
            close_price=self.close_price,
            gateway_name="DB",
        )
        return bar


class DbTickData(Document):
    """
    Tick data for database storage.

    Index is defined unique with (datetime, symbol)
    """

    symbol: str = StringField()
    exchange: str = StringField()
    datetime: datetime = DateTimeField()

    name: str = StringField()
    volume: float = FloatField()
    open_interest: float = FloatField()
    last_price: float = FloatField()
    last_volume: float = FloatField()
    limit_up: float = FloatField()
    limit_down: float = FloatField()

    open_price: float = FloatField()
    high_price: float = FloatField()
    low_price: float = FloatField()
    close_price: float = FloatField()
    pre_close: float = FloatField()

    bid_price_1: float = FloatField()
    bid_price_2: float = FloatField()
    bid_price_3: float = FloatField()
    bid_price_4: float = FloatField()
    bid_price_5: float = FloatField()

    ask_price_1: float = FloatField()
    ask_price_2: float = FloatField()
    ask_price_3: float = FloatField()
    ask_price_4: float = FloatField()
    ask_price_5: float = FloatField()

    bid_volume_1: float = FloatField()
    bid_volume_2: float = FloatField()
    bid_volume_3: float = FloatField()
    bid_volume_4: float = FloatField()
    bid_volume_5: float = FloatField()

    ask_volume_1: float = FloatField()
    ask_volume_2: float = FloatField()
    ask_volume_3: float = FloatField()
    ask_volume_4: float = FloatField()
    ask_volume_5: float = FloatField()

    meta = {
        "indexes": [
            {
                "fields": ("symbol", "exchange", "datetime"),
                "unique": True,
            }
        ],
    }

    @staticmethod
    def from_tick(tick: TickData):
        """
        Generate DbTickData object from TickData.
        """
        db_tick = DbTickData()

        db_tick.symbol = tick.symbol
        db_tick.exchange = tick.exchange.value
        db_tick.datetime = tick.datetime
        db_tick.name = tick.name
        db_tick.volume = tick.volume
        db_tick.open_interest = tick.open_interest
        db_tick.last_price = tick.last_price
        db_tick.last_volume = tick.last_volume
        db_tick.limit_up = tick.limit_up
        db_tick.limit_down = tick.limit_down
        db_tick.open_price = tick.open_price
        db_tick.high_price = tick.high_price
        db_tick.low_price = tick.low_price
        db_tick.pre_close = tick.pre_close

        db_tick.bid_price_1 = tick.bid_price_1
        db_tick.ask_price_1 = tick.ask_price_1
        db_tick.bid_volume_1 = tick.bid_volume_1
        db_tick.ask_volume_1 = tick.ask_volume_1

        if tick.bid_price_2:
            db_tick.bid_price_2 = tick.bid_price_2
            db_tick.bid_price_3 = tick.bid_price_3
            db_tick.bid_price_4 = tick.bid_price_4
            db_tick.bid_price_5 = tick.bid_price_5

            db_tick.ask_price_2 = tick.ask_price_2
            db_tick.ask_price_3 = tick.ask_price_3
            db_tick.ask_price_4 = tick.ask_price_4
            db_tick.ask_price_5 = tick.ask_price_5

            db_tick.bid_volume_2 = tick.bid_volume_2
            db_tick.bid_volume_3 = tick.bid_volume_3
            db_tick.bid_volume_4 = tick.bid_volume_4
            db_tick.bid_volume_5 = tick.bid_volume_5

            db_tick.ask_volume_2 = tick.ask_volume_2
            db_tick.ask_volume_3 = tick.ask_volume_3
            db_tick.ask_volume_4 = tick.ask_volume_4
            db_tick.ask_volume_5 = tick.ask_volume_5

        return db_tick

    def to_tick(self):
        """
        Generate TickData object from DbTickData.
        """
        tick = TickData(
            symbol=self.symbol,
            exchange=Exchange(self.exchange),
            datetime=self.datetime,
            name=self.name,
            volume=self.volume,
            open_interest=self.open_interest,
            last_price=self.last_price,
            last_volume=self.last_volume,
            limit_up=self.limit_up,
            limit_down=self.limit_down,
            open_price=self.open_price,
            high_price=self.high_price,
            low_price=self.low_price,
            pre_close=self.pre_close,
            bid_price_1=self.bid_price_1,
            ask_price_1=self.ask_price_1,
            bid_volume_1=self.bid_volume_1,
            ask_volume_1=self.ask_volume_1,
            gateway_name="DB",
        )

        if self.bid_price_2:
            tick.bid_price_2 = self.bid_price_2
            tick.bid_price_3 = self.bid_price_3
            tick.bid_price_4 = self.bid_price_4
            tick.bid_price_5 = self.bid_price_5

            tick.ask_price_2 = self.ask_price_2
            tick.ask_price_3 = self.ask_price_3
            tick.ask_price_4 = self.ask_price_4
            tick.ask_price_5 = self.ask_price_5

            tick.bid_volume_2 = self.bid_volume_2
            tick.bid_volume_3 = self.bid_volume_3
            tick.bid_volume_4 = self.bid_volume_4
            tick.bid_volume_5 = self.bid_volume_5

            tick.ask_volume_2 = self.ask_volume_2
            tick.ask_volume_3 = self.ask_volume_3
            tick.ask_volume_4 = self.ask_volume_4
            tick.ask_volume_5 = self.ask_volume_5

        return tick


class MongoManager(BaseDatabaseManager):

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> Sequence[BarData]:
        s = DbBarData.objects(
            symbol=symbol,
            exchange=exchange.value,
            interval=interval.value,
            datetime__gte=start,
            datetime__lte=end,
        )
        data = [db_bar.to_bar() for db_bar in s]
        return data

    def load_tick_data(
        self, symbol: str, exchange: Exchange, start: datetime, end: datetime
    ) -> Sequence[TickData]:
        s = DbTickData.objects(
            symbol=symbol,
            exchange=exchange.value,
            datetime__gte=start,
            datetime__lte=end,
        )
        data = [db_tick.to_tick() for db_tick in s]
        return data

    @staticmethod
    def to_update_param(d):
        return {
            "set__" + k: v.value if isinstance(v, Enum) else v
            for k, v in d.__dict__.items()
        }

    def save_bar_data(self, datas: Sequence[BarData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            updates.pop("set__vt_symbol")
            (
                DbBarData.objects(
                    symbol=d.symbol, interval=d.interval.value, datetime=d.datetime
                ).update_one(upsert=True, **updates)
            )

    def save_tick_data(self, datas: Sequence[TickData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            updates.pop("set__vt_symbol")
            (
                DbTickData.objects(
                    symbol=d.symbol, exchange=d.exchange.value, datetime=d.datetime
                ).update_one(upsert=True, **updates)
            )

    def get_newest_bar_data(
        self, symbol: str, exchange: "Exchange", interval: "Interval"
    ) -> Optional["BarData"]:
        s = (
            DbBarData.objects(
                symbol=symbol,
                exchange=exchange.value,
                interval=Interval.Value
            )
            .order_by("-datetime")
            .first()
        )
        if s:
            return s.to_bar()
        return None

    def get_oldest_bar_data(
        self, symbol: str, exchange: "Exchange", interval: "Interval"
    ) -> Optional["BarData"]:
        s = (
            DbBarData.objects(
                symbol=symbol,
                exchange=exchange.value,
                interval=Interval.Value
            )
            .order_by("+datetime")
            .first()
        )
        if s:
            return s.to_bar()
        return None

    def get_newest_tick_data(
        self, symbol: str, exchange: "Exchange"
    ) -> Optional["TickData"]:
        s = (
            DbTickData.objects(symbol=symbol, exchange=exchange.value)
            .order_by("-datetime")
            .first()
        )
        if s:
            return s.to_tick()
        return None

    def get_bar_data_statistics(self) -> List:
        """"""
        s = (
            DbBarData.objects.aggregate({
                "$group": {
                    "_id": {
                        "symbol": "$symbol",
                        "exchange": "$exchange",
                        "interval": "$interval",
                    },
                    "count": {"$sum": 1}
                }
            })
        )

        result = []

        for d in s:
            data = d["_id"]
            data["count"] = d["count"]
            result.append(data)

        return result

    def clean(self, symbol: str):
        DbTickData.objects(symbol=symbol).delete()
        DbBarData.objects(symbol=symbol).delete()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from datetime import datetime
from typing import List, Dict, Optional, Sequence, Type

from peewee import (
    AutoField,
    CharField,
    Database,
    DateTimeField,
    FloatField,
    Model,
    MySQLDatabase,
    PostgresqlDatabase,
    SqliteDatabase,
    chunked,
)

# from vnpy.trader.constant import Exchange, Interval
# from vnpy.trader.object import BarData, TickData
# from vnpy.trader.utility import get_file_path
# from .database import BaseDatabaseManager, Driver


def init(driver: Driver, settings: dict):
    init_funcs = {
        Driver.SQLITE: init_sqlite,
        Driver.MYSQL: init_mysql,
        Driver.POSTGRESQL: init_postgresql,
    }
    assert driver in init_funcs

    db = init_funcs[driver](settings)
    bar, tick = init_models(db, driver)
    return SqlManager(bar, tick)


def init_sqlite(settings: dict):
    database = settings["database"]
    path = str(get_file_path(database))
    db = SqliteDatabase(path)
    return db


def init_mysql(settings: dict):
    keys = {"database", "user", "password", "host", "port"}
    settings = {k: v for k, v in settings.items() if k in keys}
    db = MySQLDatabase(**settings)
    return db


def init_postgresql(settings: dict):
    keys = {"database", "user", "password", "host", "port"}
    settings = {k: v for k, v in settings.items() if k in keys}
    db = PostgresqlDatabase(**settings)
    return db


class ModelBase(Model):

    def to_dict(self):
        return self.__data__


def init_models(db: Database, driver: Driver):
    class DbBarData(ModelBase):
        """
        Candlestick bar data for database storage.

        Index is defined unique with datetime, interval, symbol
        """

        id = AutoField()
        symbol: str = CharField()
        exchange: str = CharField()
        datetime: datetime = DateTimeField()
        interval: str = CharField()

        volume: float = FloatField()
        open_interest: float = FloatField()
        open_price: float = FloatField()
        high_price: float = FloatField()
        low_price: float = FloatField()
        close_price: float = FloatField()

        class Meta:
            database = db
            indexes = ((("symbol", "exchange", "interval", "datetime"), True),)

        @staticmethod
        def from_bar(bar: BarData):
            """
            Generate DbBarData object from BarData.
            """
            db_bar = DbBarData()

            db_bar.symbol = bar.symbol
            db_bar.exchange = bar.exchange.value
            db_bar.datetime = bar.datetime
            db_bar.interval = bar.interval.value
            db_bar.volume = bar.volume
            db_bar.open_interest = bar.open_interest
            db_bar.open_price = bar.open_price
            db_bar.high_price = bar.high_price
            db_bar.low_price = bar.low_price
            db_bar.close_price = bar.close_price

            return db_bar

        def to_bar(self):
            """
            Generate BarData object from DbBarData.
            """
            bar = BarData(
                symbol=self.symbol,
                exchange=Exchange(self.exchange),
                datetime=self.datetime,
                interval=Interval(self.interval),
                volume=self.volume,
                open_price=self.open_price,
                high_price=self.high_price,
                open_interest=self.open_interest,
                low_price=self.low_price,
                close_price=self.close_price,
                gateway_name="DB",
            )
            return bar

        @staticmethod
        def save_all(objs: List["DbBarData"]):
            """
            save a list of objects, update if exists.
            """
            dicts = [i.to_dict() for i in objs]
            with db.atomic():
                if driver is Driver.POSTGRESQL:
                    for bar in dicts:
                        DbBarData.insert(bar).on_conflict(
                            update=bar,
                            conflict_target=(
                                DbBarData.symbol,
                                DbBarData.exchange,
                                DbBarData.interval,
                                DbBarData.datetime,
                            ),
                        ).execute()
                else:
                    for c in chunked(dicts, 50):
                        DbBarData.insert_many(
                            c).on_conflict_replace().execute()

    class DbTickData(ModelBase):
        """
        Tick data for database storage.

        Index is defined unique with (datetime, symbol)
        """

        id = AutoField()

        symbol: str = CharField()
        exchange: str = CharField()
        datetime: datetime = DateTimeField()

        name: str = CharField()
        volume: float = FloatField()
        open_interest: float = FloatField()
        last_price: float = FloatField()
        last_volume: float = FloatField()
        limit_up: float = FloatField()
        limit_down: float = FloatField()

        open_price: float = FloatField()
        high_price: float = FloatField()
        low_price: float = FloatField()
        pre_close: float = FloatField()

        bid_price_1: float = FloatField()
        bid_price_2: float = FloatField(null=True)
        bid_price_3: float = FloatField(null=True)
        bid_price_4: float = FloatField(null=True)
        bid_price_5: float = FloatField(null=True)

        ask_price_1: float = FloatField()
        ask_price_2: float = FloatField(null=True)
        ask_price_3: float = FloatField(null=True)
        ask_price_4: float = FloatField(null=True)
        ask_price_5: float = FloatField(null=True)

        bid_volume_1: float = FloatField()
        bid_volume_2: float = FloatField(null=True)
        bid_volume_3: float = FloatField(null=True)
        bid_volume_4: float = FloatField(null=True)
        bid_volume_5: float = FloatField(null=True)

        ask_volume_1: float = FloatField()
        ask_volume_2: float = FloatField(null=True)
        ask_volume_3: float = FloatField(null=True)
        ask_volume_4: float = FloatField(null=True)
        ask_volume_5: float = FloatField(null=True)

        class Meta:
            database = db
            indexes = ((("symbol", "exchange", "datetime"), True),)

        @staticmethod
        def from_tick(tick: TickData):
            """
            Generate DbTickData object from TickData.
            """
            db_tick = DbTickData()

            db_tick.symbol = tick.symbol
            db_tick.exchange = tick.exchange.value
            db_tick.datetime = tick.datetime
            db_tick.name = tick.name
            db_tick.volume = tick.volume
            db_tick.open_interest = tick.open_interest
            db_tick.last_price = tick.last_price
            db_tick.last_volume = tick.last_volume
            db_tick.limit_up = tick.limit_up
            db_tick.limit_down = tick.limit_down
            db_tick.open_price = tick.open_price
            db_tick.high_price = tick.high_price
            db_tick.low_price = tick.low_price
            db_tick.pre_close = tick.pre_close

            db_tick.bid_price_1 = tick.bid_price_1
            db_tick.ask_price_1 = tick.ask_price_1
            db_tick.bid_volume_1 = tick.bid_volume_1
            db_tick.ask_volume_1 = tick.ask_volume_1

            if tick.bid_price_2:
                db_tick.bid_price_2 = tick.bid_price_2
                db_tick.bid_price_3 = tick.bid_price_3
                db_tick.bid_price_4 = tick.bid_price_4
                db_tick.bid_price_5 = tick.bid_price_5

                db_tick.ask_price_2 = tick.ask_price_2
                db_tick.ask_price_3 = tick.ask_price_3
                db_tick.ask_price_4 = tick.ask_price_4
                db_tick.ask_price_5 = tick.ask_price_5

                db_tick.bid_volume_2 = tick.bid_volume_2
                db_tick.bid_volume_3 = tick.bid_volume_3
                db_tick.bid_volume_4 = tick.bid_volume_4
                db_tick.bid_volume_5 = tick.bid_volume_5

                db_tick.ask_volume_2 = tick.ask_volume_2
                db_tick.ask_volume_3 = tick.ask_volume_3
                db_tick.ask_volume_4 = tick.ask_volume_4
                db_tick.ask_volume_5 = tick.ask_volume_5

            return db_tick

        def to_tick(self):
            """
            Generate TickData object from DbTickData.
            """
            tick = TickData(
                symbol=self.symbol,
                exchange=Exchange(self.exchange),
                datetime=self.datetime,
                name=self.name,
                volume=self.volume,
                open_interest=self.open_interest,
                last_price=self.last_price,
                last_volume=self.last_volume,
                limit_up=self.limit_up,
                limit_down=self.limit_down,
                open_price=self.open_price,
                high_price=self.high_price,
                low_price=self.low_price,
                pre_close=self.pre_close,
                bid_price_1=self.bid_price_1,
                ask_price_1=self.ask_price_1,
                bid_volume_1=self.bid_volume_1,
                ask_volume_1=self.ask_volume_1,
                gateway_name="DB",
            )

            if self.bid_price_2:
                tick.bid_price_2 = self.bid_price_2
                tick.bid_price_3 = self.bid_price_3
                tick.bid_price_4 = self.bid_price_4
                tick.bid_price_5 = self.bid_price_5

                tick.ask_price_2 = self.ask_price_2
                tick.ask_price_3 = self.ask_price_3
                tick.ask_price_4 = self.ask_price_4
                tick.ask_price_5 = self.ask_price_5

                tick.bid_volume_2 = self.bid_volume_2
                tick.bid_volume_3 = self.bid_volume_3
                tick.bid_volume_4 = self.bid_volume_4
                tick.bid_volume_5 = self.bid_volume_5

                tick.ask_volume_2 = self.ask_volume_2
                tick.ask_volume_3 = self.ask_volume_3
                tick.ask_volume_4 = self.ask_volume_4
                tick.ask_volume_5 = self.ask_volume_5

            return tick

        @staticmethod
        def save_all(objs: List["DbTickData"]):
            dicts = [i.to_dict() for i in objs]
            with db.atomic():
                if driver is Driver.POSTGRESQL:
                    for tick in dicts:
                        DbTickData.insert(tick).on_conflict(
                            update=tick,
                            conflict_target=(
                                DbTickData.symbol,
                                DbTickData.exchange,
                                DbTickData.datetime,
                            ),
                        ).execute()
                else:
                    for c in chunked(dicts, 50):
                        DbTickData.insert_many(c).on_conflict_replace().execute()

    db.connect()
    db.create_tables([DbBarData, DbTickData])
    return DbBarData, DbTickData


class SqlManager(BaseDatabaseManager):

    def __init__(self, class_bar: Type[Model], class_tick: Type[Model]):
        self.class_bar = class_bar
        self.class_tick = class_tick

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> Sequence[BarData]:
        s = (
            self.class_bar.select()
                .where(
                (self.class_bar.symbol == symbol)
                & (self.class_bar.exchange == exchange.value)
                & (self.class_bar.interval == interval.value)
                & (self.class_bar.datetime >= start)
                & (self.class_bar.datetime <= end)
            )
            .order_by(self.class_bar.datetime)
        )
        data = [db_bar.to_bar() for db_bar in s]
        return data

    def load_tick_data(
        self, symbol: str, exchange: Exchange, start: datetime, end: datetime
    ) -> Sequence[TickData]:
        s = (
            self.class_tick.select()
                .where(
                (self.class_tick.symbol == symbol)
                & (self.class_tick.exchange == exchange.value)
                & (self.class_tick.datetime >= start)
                & (self.class_tick.datetime <= end)
            )
            .order_by(self.class_tick.datetime)
        )

        data = [db_tick.to_tick() for db_tick in s]
        return data

    def save_bar_data(self, datas: Sequence[BarData]):
        ds = [self.class_bar.from_bar(i) for i in datas]
        self.class_bar.save_all(ds)

    def save_tick_data(self, datas: Sequence[TickData]):
        ds = [self.class_tick.from_tick(i) for i in datas]
        self.class_tick.save_all(ds)

    def get_newest_bar_data(
        self, symbol: str, exchange: "Exchange", interval: "Interval"
    ) -> Optional["BarData"]:
        s = (
            self.class_bar.select()
                .where(
                (self.class_bar.symbol == symbol)
                & (self.class_bar.exchange == exchange.value)
                & (self.class_bar.interval == interval.value)
            )
            .order_by(self.class_bar.datetime.desc())
            .first()
        )
        if s:
            return s.to_bar()
        return None

    def get_oldest_bar_data(
        self, symbol: str, exchange: "Exchange", interval: "Interval"
    ) -> Optional["BarData"]:
        s = (
            self.class_bar.select()
                .where(
                (self.class_bar.symbol == symbol)
                & (self.class_bar.exchange == exchange.value)
                & (self.class_bar.interval == interval.value)
            )
            .order_by(self.class_bar.datetime.asc())
            .first()
        )
        if s:
            return s.to_bar()
        return None

    def get_newest_tick_data(
        self, symbol: str, exchange: "Exchange"
    ) -> Optional["TickData"]:
        s = (
            self.class_tick.select()
                .where(
                (self.class_tick.symbol == symbol)
                & (self.class_tick.exchange == exchange.value)
            )
            .order_by(self.class_tick.datetime.desc())
            .first()
        )
        if s:
            return s.to_tick()
        return None

    def get_bar_data_statistics(self) -> List[Dict]:
        """"""
        s = (
            self.class_bar.select().group_by(
                self.class_bar.symbol,
                self.class_bar.exchange,
                self.class_bar.interval
            )
        )

        result = []

        for data in s:
            result.append({
                "symbol": data.symbol,
                "exchange": data.exchange,
                "interval": data.interval,
                "count": int(str(data))
            })

        return result

    def clean(self, symbol: str):
        self.class_bar.delete().where(self.class_bar.symbol == symbol).execute()
        self.class_tick.delete().where(self.class_tick.symbol == symbol).execute()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
# from .database import BaseDatabaseManager, Driver


def init(settings: dict) -> BaseDatabaseManager:
    driver = Driver(settings["driver"])
    if driver is Driver.MONGODB:
        return init_nosql(driver=driver, settings=settings)
    else:
        return init_sql(driver=driver, settings=settings)


def init_sql(driver: Driver, settings: dict):
    from .database_sql import init
    keys = {'database', "host", "port", "user", "password"}
    settings = {k: v for k, v in settings.items() if k in keys}
    _database_manager = init(driver, settings)
    return _database_manager


def init_nosql(driver: Driver, settings: dict):
    from .database_mongo import init
    _database_manager = init(driver, settings=settings)
    return _database_manager



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from functools import partial
import os
import sqlite3

import sqlalchemy as sa
from six.moves import range

from util.input_validation import coerce_string

SQLITE_MAX_VARIABLE_NUMBER = 998


def group_into_chunks(items, chunk_size=SQLITE_MAX_VARIABLE_NUMBER):
    items = list(items)
    return [items[x:x+chunk_size]
            for x in range(0, len(items), chunk_size)]


def verify_sqlite_path_exists(path):
    if path != ':memory:' and not os.path.exists(path):
        raise ValueError("SQLite file {!r} doesn't exist.".format(path))


def check_and_create_connection(path, require_exists):
    if require_exists:
        verify_sqlite_path_exists(path)
    return sqlite3.connect(path)


def check_and_create_engine(path, require_exists):
    if require_exists:
        verify_sqlite_path_exists(path)
    return sa.create_engine('sqlite:///' + path)


def coerce_string_to_conn(require_exists):
    return coerce_string(
        partial(check_and_create_connection, require_exists=require_exists)
    )


def coerce_string_to_eng(require_exists):
    return coerce_string(
        partial(check_and_create_engine, require_exists=require_exists)
    )
