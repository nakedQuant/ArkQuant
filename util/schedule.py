# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from atexit import register

# 解释器正常终止时执行
# @register --- func() . atexit.register(func, *args, **kwargs)


def schedule_function(self,
                      func,
                      date_rule=None,
                      time_rule=None,
                      half_days=True,
                      calendar=None):
    """
    Schedule a function to be called repeatedly in the future.

    Parameters
    ----------
    func : callable
        The function to execute when the rule is triggered. ``func`` should
        have the same signature as ``handle_data``.
    date_rule : zipline.util.events.EventRule, optional
        Rule for the dates on which to execute ``func``. If not
        passed, the function will run every trading day.
    time_rule : zipline.util.events.EventRule, optional
        Rule for the time at which to execute ``func``. If not passed, the
        function will execute at the end of the first market minute of the
        day.
    half_days : bool, optional
        Should this rule fire on half days? Default is True.
    calendar : Sentinel, optional
        Calendar used to compute rules that depend on the trading _calendar.
    """
    raise NotImplementedError()
