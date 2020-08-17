MAX_MONTH_RANGE = 23
MAX_WEEK_RANGE = 5


class AfterOpen(StatelessRule):
    """
    A rule that triggers for some offset after the market opens.
    Example that triggers after 30 minutes of the market opening:

    >>> AfterOpen(minutes=30)  # doctest: +ELLIPSIS
    <zipline.utils.events.AfterOpen object at ...>
    """
    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(
            offset,
            kwargs,
            datetime.timedelta(minutes=1),  # Defaults to the first minute.
        )

        self._period_start = None
        self._period_end = None
        self._period_close = None

        self._one_minute = datetime.timedelta(minutes=1)

    def calculate_dates(self, dt):
        """
        Given a date, find that day's open and period end (open + offset).
        """
        period_start, period_close = self.cal.open_and_close_for_session(
            self.cal.minute_to_session_label(dt),
        )

        # Align the market open and close times here with the execution times
        # used by the simulation clock. This ensures that scheduled functions
        # trigger at the correct times.
        self._period_start = self.cal.execution_time_from_open(period_start)
        self._period_close = self.cal.execution_time_from_close(period_close)

        self._period_end = self._period_start + self.offset - self._one_minute

    def should_trigger(self, dt):
        # There are two reasons why we might want to recalculate the dates.
        # One is the first time we ever call should_trigger, when
        # self._period_start is none. The second is when we're on a new day,
        # and need to recalculate the dates. For performance reasons, we rely
        # on the fact that our clock only ever ticks forward, since it's
        # cheaper to do dt1 <= dt2 than dt1.date() != dt2.date(). This means
        # that we will NOT correctly recognize a new date if we go backwards
        # in time(which should never happen in a simulation, or in live
        # trading)
        if (
            self._period_start is None or
            self._period_close <= dt
        ):
            self.calculate_dates(dt)

        return dt == self._period_end


class BeforeClose(StatelessRule):
    """
    A rule that triggers for some offset time before the market closes.
    Example that triggers for the last 30 minutes every day:

    >>> BeforeClose(minutes=30)  # doctest: +ELLIPSIS
    <zipline.utils.events.BeforeClose object at ...>
    """
    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(
            offset,
            kwargs,
            datetime.timedelta(minutes=1),  # Defaults to the last minute.
        )

        self._period_start = None
        self._period_close = None
        self._period_end = None

        self._one_minute = datetime.timedelta(minutes=1)

    def calculate_dates(self, dt):
        """
        Given a dt, find that day's close and period start (close - offset).
        """
        period_end = self.cal.open_and_close_for_session(
            self.cal.minute_to_session_label(dt),
        )[1]

        # Align the market close time here with the execution time used by the
        # simulation clock. This ensures that scheduled functions trigger at
        # the correct times.
        self._period_end = self.cal.execution_time_from_close(period_end)

        self._period_start = self._period_end - self.offset
        self._period_close = self._period_end

    def should_trigger(self, dt):
        # There are two reasons why we might want to recalculate the dates.
        # One is the first time we ever call should_trigger, when
        # self._period_start is none. The second is when we're on a new day,
        # and need to recalculate the dates. For performance reasons, we rely
        # on the fact that our clock only ever ticks forward, since it's
        # cheaper to do dt1 <= dt2 than dt1.date() != dt2.date(). This means
        # that we will NOT correctly recognize a new date if we go backwards
        # in time(which should never happen in a simulation, or in live
        # trading)
        if self._period_start is None or self._period_close <= dt:
            self.calculate_dates(dt)

        return self._period_start == dt


class NotHalfDay(StatelessRule):
    """
    A rule that only triggers when it is not a half day.
    """
    def should_trigger(self, dt):
        return self.cal.minute_to_session_label(dt) \
            not in self.cal.early_closes




class time_rules(object):
    """Factories for time-based :func:`~zipline.api.schedule_function` rules.

    See Also
    --------
    :func:`~zipline.api.schedule_function`
    """

    @staticmethod
    def market_open(offset=None, hours=None, minutes=None):
        """
        Create a rule that triggers at a fixed offset from market open.

        The offset can be specified either as a :class:`datetime.timedelta`, or
        as a number of hours and minutes.

        Parameters
        ----------
        offset : datetime.timedelta, optional
            If passed, the offset from market open at which to trigger. Must be
            at least 1 minute.
        hours : int, optional
            If passed, number of hours to wait after market open.
        minutes : int, optional
            If passed, number of minutes to wait after market open.

        Returns
        -------
        rule : zipline.utils.events.EventRule

        Notes
        -----
        If no arguments are passed, the default offset is one minute after
        market open.

        If ``offset`` is passed, ``hours`` and ``minutes`` must not be
        passed. Conversely, if either ``hours`` or ``minutes`` are passed,
        ``offset`` must not be passed.
        """
        return AfterOpen(offset=offset, hours=hours, minutes=minutes)

    @staticmethod
    def market_close(offset=None, hours=None, minutes=None):
        """
        Create a rule that triggers at a fixed offset from market close.

        The offset can be specified either as a :class:`datetime.timedelta`, or
        as a number of hours and minutes.

        Parameters
        ----------
        offset : datetime.timedelta, optional
            If passed, the offset from market close at which to trigger. Must
            be at least 1 minute.
        hours : int, optional
            If passed, number of hours to wait before market close.
        minutes : int, optional
            If passed, number of minutes to wait before market close.

        Returns
        -------
        rule : zipline.utils.events.EventRule

        Notes
        -----
        If no arguments are passed, the default offset is one minute before
        market close.

        If ``offset`` is passed, ``hours`` and ``minutes`` must not be
        passed. Conversely, if either ``hours`` or ``minutes`` are passed,
        ``offset`` must not be passed.
        """
        return BeforeClose(offset=offset, hours=hours, minutes=minutes)

    every_minute = Always


class OncePerDay(StatefulRule):
    def __init__(self, rule=None):
        self.triggered = False

        self.date = None
        self.next_date = None

        super(OncePerDay, self).__init__(rule)

    def should_trigger(self, dt):
        if self.date is None or dt >= self.next_date:
            # initialize or reset for new date
            self.triggered = False
            self.date = dt

            # record the timestamp for the next day, so that we can use it
            # to know if we've moved to the next day
            self.next_date = dt + pd.Timedelta(1, unit="d")

        if not self.triggered and self.rule.should_trigger(dt):
            self.triggered = True
            return True


def make_eventrule(date_rule, time_rule, cal, half_days=True):
    """
    Constructs an event rule from the factory api.
    """
    _check_if_not_called(date_rule)
    _check_if_not_called(time_rule)

    if half_days:
        inner_rule = date_rule & time_rule
    else:
        inner_rule = date_rule & time_rule & NotHalfDay()

    opd = OncePerDay(rule=inner_rule)
    # This is where a scheduled function's rule is associated with a calendar.
    opd.cal = cal
    return opd


@api_method
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
    9:25 -- 9:30属于engine --- execution plan include unchange positions
    """
    # When the user calls schedule_function(func, <time_rule>), assume that
    # the user meant to specify a time rule but no date rule, instead of
    # a date rule and no time rule as the signature suggests5
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

    self.add_event(
        make_eventrule(date_rule, time_rule, cal, half_days),
        func,
    )
