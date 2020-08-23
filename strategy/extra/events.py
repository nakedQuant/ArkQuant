#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from numpy import (
    array_equal,
    broadcast,
    busday_count,
    empty,
    nan,
    where,
    newaxis
)
# announce_date , ex_date , pay_date


def busday_count_mask_NaT(begindates, enddates, out=None):
    """
    Simple of numpy.busday_count that returns `float` arrays rather than int
    arrays, and handles `NaT`s by returning `NaN`s where the inputs were `NaT`.

    Doesn't support custom weekdays or calendars, but probably should in the
    future.

    See Also
    --------
    np.busday_count --- Mon Fri ( weekmaskstr or array_like of bool, optional)
    """
    if out is None:
        out = empty(broadcast(begindates, enddates).shape, dtype=float)

    beginmask = isnat(begindates)
    endmask = isnat(enddates)

    out = busday_count(
        # Temporarily fill in non-NaT values.
        where(beginmask, _notNaT, begindates),
        where(endmask, _notNaT, enddates),
        out=out,
    )

    # Fill in entries where either comparison was NaT with nan in the output.
    out[beginmask | endmask] = nan
    return out


class BusinessDaysEvent(object):
    """
    Abstract class for business days since a previous event.
    Returns the number of **business days** (not trading days!) since
    the most recent event date for each asset.

    This doesn't use trading days for symmetry with
    BusinessDaysUntilNextEarnings.

    asset which announced or will announce the event today will produce a
    value of 0.0. asset that announced the event on the previous business
    day will produce a value of 1.0.

    asset for which the event date is `NaT` will produce a value of `NaN`.

    Factors describing information about event data (e.g. earnings
    announcements, acquisitions, dividends, etc.).
    """
