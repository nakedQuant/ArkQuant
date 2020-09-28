#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""


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
    # numpy.busday_count(begindates, enddates, weekmask='1111100', holidays=[], busdaycal=None, out=None)
    # Counts the number of valid days between begindates and enddates, not including the day of enddates.
    #  [1,1,1,1,1,0,0]; a length-seven string, like ‘1111100’; or a string like “Mon Tue Wed Thu Fri Sat Sun”
    # np.busday_count --- Mon Fri ( weekmaskstr or array_like of bool, optional)
    # where(condition, x,y --- handles `NaT`s by returning `NaN`s where the inputs were `NaT`.
    # announce_date , ex_date , pay_date (比如举牌、增持、减持、股权转让、重组）



