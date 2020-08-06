# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

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
    date_rule : zipline.utils.events.EventRule, optional
        Rule for the dates on which to execute ``func``. If not
        passed, the function will run every trading day.
    time_rule : zipline.utils.events.EventRule, optional
        Rule for the time at which to execute ``func``. If not passed, the
        function will execute at the end of the first market minute of the
        day.
    half_days : bool, optional
        Should this rule fire on half days? Default is True.
    calendar : Sentinel, optional
        Calendar used to compute rules that depend on the trading _calendar.

    See Also
    --------
    :class:`zipline.api.date_rules`
    :class:`zipline.api.time_rules`

    Reference:
        schedule specify the time to run the strategy to get the next action ,this is used in real trading
    sched enter(delay,priority,func,augment or kwargs) ; enterabs(time,priority ,func,augment or kwargs)
    queue ( accumlate the scheduled task)
    import sched,time

    def crontab():
        pass

    if __name__=='__main__':

        while True:
            s = sched.scheduler(time.time, time.sleep)
            task = s.enter(0,1,crontab,argument = ())
            s.queue.append(task)
            s.run()
    """

    # When the user calls schedule_function(func, <time_rule>), assume that
    # the user meant to specify a time rule but no date rule, instead of
    # a date rule and no time rule as the signature suggests
    if isinstance(date_rule, (AfterOpen, BeforeClose)) and not time_rule:
        warnings.warn('Got a time rule for the second positional argument '
                      'date_rule. You should use keyword argument '
                      'time_rule= when calling schedule_function without '
                      'specifying a date_rule', stacklevel=3)

    date_rule = date_rule or date_rules.every_day()
    time_rule = ((time_rule or time_rules.every_minute())
                 if self.sim_params.data_frequency == 'minute' else
                 # If we are in daily mode the time_rule is ignored.
                 time_rules.every_minute())

    # Check the type of the algorithm's schedule before pulling _calendar
    # Note that the ExchangeTradingSchedule is currently the only
    # TradingSchedule class, so this is unlikely to be hit
    if calendar is None:
        cal = self.trading_calendar
    elif calendar is calendars.US_EQUITIES:
        cal = get_calendar('XNYS')
    elif calendar is calendars.US_FUTURES:
        cal = get_calendar('us_futures')
    else:
        raise ScheduleFunctionInvalidCalendar(
            given_calendar=calendar,
            allowed_calendars=(
                '[trading-calendars.US_EQUITIES, trading-calendars.US_FUTURES]'
            ),
        )

    self.add_event(
        make_eventrule(date_rule, time_rule, cal, half_days),
        func,
    )
