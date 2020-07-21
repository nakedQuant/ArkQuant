#
# Copyright 2015 Quantopian, Inc.
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
from contextlib import ExitStack
from copy import copy


class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self, algo,sim_params, clock, benchmark,
                universe_func):

        # ==============
        # Simulation
        # Param Setup
        # ==============
        self.sim_params = sim_params

        # ==============
        # Algo Setup
        # ==============
        self.algo = algo

        self.benchmark = benchmark

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None
        self.clock = clock
        # This object is the way that user algorithms interact with OHLCV data,
        # fetcher data, and some API methods like `data.can_trade`.
        self.current_data = self._create_bar_data(universe_func)

    def get_simulation_dt(self):
        return self.simulation_dt

    #获取日数据，封装为一个API(fetch process flush other api)
    def _create_bar_data(self, universe_func):
        return BarData(
            data_portal=self.data_portal,
            simulation_dt_func=self.get_simulation_dt,
            data_frequency=self.sim_params.data_frequency,
            trading_calendar=self.algo.trading_calendar,
            restrictions=self.restrictions,
            universe_func=universe_func
        )

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        trading_o_and_c = self.trading_calendar.schedule.ix[
            self.sim_params.sessions]
        market_closes = trading_o_and_c['market_close']
        minutely_emission = False

        if self.sim_params.data_frequency == 'minute':
            market_opens = trading_o_and_c['market_open']
            minutely_emission = self.sim_params.emission_rate == "minute"

            # The calendar's execution times are the minutes over which we
            # actually want to run the clock. Typically the execution times
            # simply adhere to the market open and close times. In the case of
            # the futures calendar, for example, we only want to simulate over
            # a subset of the full 24 hour calendar, so the execution times
            # dictate a market open time of 6:31am US/Eastern and a close of
            # 5:00pm US/Eastern.
            execution_opens = \
                self.trading_calendar.execution_time_from_open(market_opens)
            execution_closes = \
                self.trading_calendar.execution_time_from_close(market_closes)
        else:
            # in daily mode, we want to have one bar per session, timestamped
            # as the last minute of the session.
            execution_closes = \
                self.trading_calendar.execution_time_from_close(market_closes)
            execution_opens = execution_closes

        # FIXME generalize these values
        before_trading_start_minutes = days_at_time(
            self.sim_params.sessions,
            time(8, 45),
            "US/Eastern"
        )

        return MinuteSimulationClock(
            self.sim_params.sessions,
            execution_opens,
            execution_closes,
            before_trading_start_minutes,
            minute_emission=minutely_emission,
        )


    def tranform(self):
        """
        Main generator work loop.
        """
        algo = self.algo
        ledger = algo.ledger
        metrics_tracker = algo.metrics_tracker

        def once_a_day(dt):
            # daily metrics run
            metrics_tracker.handle_market_open(dt)
            #生成交易订单
            txns = MatchUp.carry_out(algo.engine,ledger)
            #处理交易订单
            ledger.process_transaction(txns)
            algo.calculate_capital_changes(midnight_dt,
                                           emission_rate=emission_rate,
                                           is_interday=True)
            algo.on_dt_changed(midnight_dt)

        def on_exit():
            # Remove references to algo, data portal, et al to break cycles
            # and ensure deterministic cleanup of these objects when the
            # simulation finishes.
            self.algo = None
            self.benchmark_source = self.current_data = self.data_portal = None

        with ExitStack() as stack:
            """
            由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
            这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
            enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
            callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
            """

            stack.callback(on_exit)
            stack.enter_context(self.processor)
            stack.enter_context(ZiplineAPI(self.algo))

            metrics_tracker.handle_start_of_simulation(self.benchmark)

            for dt, action in self.clock:
                if action == BAR:
                    for capital_change_packet in every_bar(dt):
                        yield capital_change_packet
                elif action == SESSION_START:
                    for capital_change_packet in once_a_day(dt):
                        yield capital_change_packet
                elif action == SESSION_END:
                    # End of the session.
                    positions = metrics_tracker.positions
                    position_assets = algo.asset_finder.retrieve_all(positions)
                    self._cleanup_expired_assets(dt, position_assets)

                    execute_order_cancellation_policy()
                    algo.validate_account_controls()

                    yield self._get_daily_message(dt, algo, metrics_tracker)
                elif action == BEFORE_TRADING_START_BAR:
                    self.simulation_dt = dt
                    algo.on_dt_changed(dt)
                    algo.before_trading_start(self.current_data)
                elif action == MINUTE_END:
                    minute_msg = self._get_minute_message(
                        dt,
                        algo,
                        metrics_tracker,
                    )
                    yield minute_msg

            for dt in algo.trading_calendar:
                once_a_day(dt)

            # risk_message = metrics_tracker.handle_simulation_end(
            #     self.data_portal,
            # )
            risk_message = metrics_tracker.handle_simulation_end()

            yield risk_message

    def _get_daily_message(self, algo, metrics_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = metrics_tracker.handle_market_close()

        # perf_message = metrics_tracker.handle_market_close(
        #     dt,
        #     self.data_portal,
        # )
        perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
        return perf_message

    def _create_benchmark_source(self):
        if self.benchmark_sid is not None:
            benchmark_asset = self.asset_finder.retrieve_asset(
                self.benchmark_sid
            )
            benchmark_returns = None
        else:
            if self.benchmark_returns is None:
                raise ValueError("Must specify either benchmark_sid "
                                 "or benchmark_returns.")
            benchmark_asset = None
            # get benchmark info from trading environment, which defaults to
            # downloading data from IEX Trading.
            benchmark_returns = self.benchmark_returns
        return BenchmarkSource(
            benchmark_asset=benchmark_asset,
            benchmark_returns=benchmark_returns,
            trading_calendar=self.trading_calendar,
            sessions=self.sim_params.sessions,
            data_portal=self.data_portal,
            emission_rate=self.sim_params.emission_rate,
        )

    def _create_metrics_tracker(self):
        #'start_of_simulation','end_of_simulation','start_of_session'，'end_of_session','end_of_bar'
        return MetricsTracker(
            trading_calendar=self.trading_calendar,
            first_session=self.sim_params.start_session,
            last_session=self.sim_params.end_session,
            capital_base=self.sim_params.capital_base,
            emission_rate=self.sim_params.emission_rate,
            data_frequency=self.sim_params.data_frequency,
            asset_finder=self.asset_finder,
            metrics=self._metrics_set,
        )

    def _create_generator(self, sim_params):
        if sim_params is not None:
            self.sim_params = sim_params

        self.metrics_tracker = metrics_tracker = self._create_metrics_tracker()

        # Set the dt initially to the period start by forcing it to change.
        self.on_dt_changed(self.sim_params.start_session)

        if not self.initialized:
            self.initialize(**self.initialize_kwargs)
            self.initialized = True

        benchmark_source = self._create_benchmark_source()

        self.trading_client = AlgorithmSimulator(
            self,
            sim_params,
            self.data_portal,
            self._create_clock(),
            benchmark_source,
            self.restrictions,
            universe_func=self._calculate_universe
        )

        metrics_tracker.handle_start_of_simulation(benchmark_source)
        return self.trading_client.transform()

    def get_generator(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(self.sim_params)

    def analyze(self, perf):
        # 分析stats
        if self._analyze is None:
            return

        with ZiplineAPI(self):
            self._analyze(self, perf)

    def run(self, data_portal=None):
        """Run the algorithm.
        """
        # HACK: I don't think we really want to support passing a data portal
        # this late in the long term, but this is needed for now for backwards
        # compat downstream.
        if data_portal is not None:
            self.data_portal = data_portal
            self.asset_finder = data_portal.asset_finder
        elif self.data_portal is None:
            raise RuntimeError(
                "No data portal in TradingAlgorithm.run().\n"
                "Either pass a DataPortal to TradingAlgorithm() or to run()."
            )
        else:
            assert self.asset_finder is not None, \
                "Have data portal without asset_finder."

        # Create zipline and loop through simulated_trading.
        # Each iteration returns a perf dictionary
        try:
            perfs = []
            for perf in self.get_generator():
                perfs.append(perf)

            # convert perf dict to pandas dataframe
            daily_stats = self._create_daily_stats(perfs)

            self.analyze(daily_stats)
        finally:
            self.data_portal = None
            self.metrics_tracker = None

        return daily_stats

    def _create_daily_stats(self, perfs):
        # create daily and cumulative stats dataframe
        daily_perfs = []
        # TODO: the loop here could overwrite expected properties
        # of daily_perf. Could potentially raise or log a
        # warning.
        for perf in perfs:
            if 'daily_perf' in perf:

                perf['daily_perf'].update(
                    perf['daily_perf'].pop('recorded_vars')
                )
                perf['daily_perf'].update(perf['cumulative_risk_metrics'])
                daily_perfs.append(perf['daily_perf'])
            else:
                self.risk_report = perf

        daily_dts = pd.DatetimeIndex(
            [p['period_close'] for p in daily_perfs], tz='UTC'
        )
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
        return daily_stats

    #根据dt获取change,动态计算，更新数据
    def calculate_capital_changes(self, dt, emission_rate, is_interday,
                                  portfolio_value_adjustment=0.0):
        """
        If there is a capital change for a given dt, this means the the change
        occurs before `handle_data` on the given dt. In the case of the
        change being a target value, the change will be computed on the
        portfolio value according to prices at the given dt

        `portfolio_value_adjustment`, if specified, will be removed from the
        portfolio_value of the cumulative performance when calculating deltas
        from target capital changes.
        """
        try:
            capital_change = self.capital_changes[dt]
        except KeyError:
            return

        self._sync_last_sale_prices()
        if capital_change['type'] == 'target':
            target = capital_change['value']
            capital_change_amount = (
                target -
                (
                    self.portfolio.portfolio_value -
                    portfolio_value_adjustment
                )
            )

            log.info('Processing capital change to target %s at %s. Capital '
                     'change delta is %s' % (target, dt,
                                             capital_change_amount))
        elif capital_change['type'] == 'delta':
            target = None
            capital_change_amount = capital_change['value']
            log.info('Processing capital change of delta %s at %s'
                     % (capital_change_amount, dt))
        else:
            log.error("Capital change %s does not indicate a valid type "
                      "('target' or 'delta')" % capital_change)
            return

        self.capital_change_deltas.update({dt: capital_change_amount})
        self.metrics_tracker.capital_change(capital_change_amount)

        yield {
            'capital_change':
                {'date': dt,
                 'type': 'cash',
                 'target': target,
                 'delta': capital_change_amount}
        }

    @api_method
    def get_environment(self, field='platform'):
        """Query the execution environment.

        Parameters
        ----------
        field : {'platform', 'arena', 'data_frequency',
                 'start', 'end', 'capital_base', 'platform', '*'}
            The field to query. The options have the following meanings:
              arena : str
                  The arena from the simulation parameters. This will normally
                  be ``'backtest'`` but some systems may use this distinguish
                  live trading from backtesting.
              data_frequency : {'daily', 'minute'}
                  data_frequency tells the algorithm if it is running with
                  daily data or minute data.
              start : datetime
                  The start date for the simulation.
              end : datetime
                  The end date for the simulation.
              capital_base : float
                  The starting capital for the simulation.
              platform : str
                  The platform that the code is running on. By default this
                  will be the string 'zipline'. This can allow algorithms to
                  know if they are running on the Quantopian platform instead.
              * : dict[str -> any]
                  Returns all of the fields in a dictionary.

        Returns
        -------
        val : any
            The value for the field queried. See above for more information.

        Raises
        ------
        ValueError
            Raised when ``field`` is not a valid option.
        """
        env = {
            'arena': self.sim_params.arena,
            'data_frequency': self.sim_params.data_frequency,
            'start': self.sim_params.first_open,
            'end': self.sim_params.last_close,
            'capital_base': self.sim_params.capital_base,
            'platform': self._platform
        }
        if field == '*':
            return env
        else:
            try:
                return env[field]
            except KeyError:
                raise ValueError(
                    '%r is not a valid field for get_environment' % field,
                )

    def add_event(self, rule, callback):
        """Adds an event to the algorithm's EventManager.

        Parameters
        ----------
        rule : EventRule
            The rule for when the callback should be triggered.
        callback : callable[(context, data) -> None]
            The function to execute when the rule is triggered.
        """
        self.event_manager.add_event(
            zipline.utils.events.Event(rule, callback),
        )

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
            Calendar used to compute rules that depend on the trading calendar.

        See Also
        --------
        :class:`zipline.api.date_rules`
        :class:`zipline.api.time_rules`
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

        # Check the type of the algorithm's schedule before pulling calendar
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
