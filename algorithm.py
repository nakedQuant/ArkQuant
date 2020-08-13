# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from copy import copy
from datetime import tzinfo
import pytz, pandas as pd, numpy as np, warnings
from contextlib import ExitStack
from itertools import chain, repeat

from .error.errors import (
    AttachPipelineAfterInitialize,
    DuplicatePipelineName,
    IncompatibleCommissionModel,
    IncompatibleSlippageModel,
    RegisterTradingControlPostInit,
    ScheduleFunctionInvalidCalendar,
    SetBenchmarkOutsideInitialize,
    SetCancelPolicyPostInit,
    SetCommissionPostInit,
    SetSlippagePostInit,
    UnsupportedCancelPolicy,
    UnsupportedDatetimeFormat,
    ZeroCapitalError
)
from .finance.oms.blotter import SimulationBlotter
from .finance.control import (
    LongOnly,
    MaxOrderSize,
    MaxPositionSize,
)
from .finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from .finance.cancel_policy import NeverCancel, CancelPolicy

from .finance.restrictions import (
    NoRestrictions,
    StaticRestrictions,
    SecurityListRestrictions,
)

from .gateWay.asset.assets import Asset, Equity, Convertible, Fund
from gateWay.driver.data_portal import DataPortal
from .gens.tradesimulation import AlgorithmSimulator
from .trade.metrics.tracker import MetricsTracker
from .pipe.pipeline import Pipeline
from .pipe.domain import Domain
from .pipe.engine import (
    SimplePipelineEngine,
    NoEngineRegistered,
)
from _calendar.trading_calendar import calendar
from .utils.api_support import (
    api_method,
    # require_initialized,
    # require_not_initialized,
    ZiplineAPI,
    # disallowed_in_before_trading_start)
)
from .utils.input_validation import (
    coerce_string,
    ensure_upper_case,
    # error_keywords,
    expect_dtypes,
    expect_types,
    optional,
    optionally,
)
from .utils.pandas_utils import normalize_date
from .utils.eventManager import EventManager
from .utils.preprocess import preprocess
from .utils.security_list import SecurityList
from .gens.clock import MinuteSimulationClock


class TradingAlgorithm(object):
    """A class that represents a trading strategy and parameters to execute
    the strategy.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``initialize`` unless listed below.
    initialize : callable[context -> None], optional
        Function that is called at the start of the simulation to
        setup the initial context.
    handle_data : callable[(context, data) -> None], optional
        Function called on every bar. This is where most logic should be
        implemented.
    before_trading_start : callable[(context, data) -> None], optional
        Function that is called before any bars have been processed each
        day.
    analyze : callable[(context, DataFrame) -> None], optional
        Function that is called at the end of the backtest. This is passed
        the context and the performance results for the backtest.
    script : str, optional
        Algoscript that contains the definitions for the four algorithm
        lifecycle functions and any supporting code.
    namespace : dict, optional
        The namespace to execute the algoscript in. By default this is an
        empty namespace that will include only python built ins.
    algo_filename : str, optional
        The filename for the algoscript. This will be used in exception
        tracebacks. default: '<string>'.
    data_frequency : {'daily', 'minute'}, optional
        The duration of the bars.
    equities_metadata : dict or DataFrame or file-like object, optional
        If dict is provided, it must have the following structure:
        * keys are the identifiers
        * values are dicts containing the metadata, with the metadata
          field name as the key
        If pandas.DataFrame is provided, it must have the
        following structure:
        * column names must be the metadata fields
        * index must be the different asset identifiers
        * array contents should be the metadata value
        If an object with a ``read`` method is provided, ``read`` must
        return rows containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    futures_metadata : dict or DataFrame or file-like object, optional
        The same layout as ``equities_metadata`` except that it is used
        for futures information.
    identifiers : list, optional
        Any asset identifiers that are not provided in the
        equities_metadata, but will be traded by this TradingAlgorithm.
    get_pipeline_loader : callable[BoundColumn -> pipe], optional
        The function that maps Pipeline columns to their loaders.
    create_event_context : callable[BarData -> context manager], optional
        A function used to create a context mananger that wraps the
        execution of all events that are scheduled for a bar.
        This function will be passed the data for the bar and should
        return the actual context manager that will be entered.
    history_container_class : type, optional
        The type of history container to use. default: HistoryContainer
    platform : str, optional
        The platform the simulation is running on. This can be queried for
        in the simulation with ``get_environment``. This allows algorithms
        to conditionally execute code based on platform it is running on.
        default: 'zipline'
    adjustment_reader : AdjustmentReader
        The interface to the adjustments.
    量化交易系统:
        a.策略识别（搜索策略 ， 挖掘优势 ， 交易频率）
        b.回溯测试（获取数据 ， 分析策略性能 ，剔除偏差）
        c.交割系统（经纪商接口 ，交易自动化 ， 交易成本最小化）
        d.风险管理（最优资本配置 ， 最优赌注或者凯利准则 ， 海龟仓位管理）
    """
    def __init__(self,
                 first_session,
                 last_session,
                 capital_base,
                 data_portal=None,
                 asset_finder=None,
                 slippage=None,
                 commission=None,
                 blotter_class=None,
                 cancel_policy=None,
                 # generator
                 creator=None,

                 # algorithm API
                 namespace=None,
                 script=None,
                 algo_filenames=None,
                 restrictions=None,

                 allocation_policy=average,

                 analyze=None,
                 metrics_set=None,

                 # delay --- transfer put to call
                 delay=1,
                 benchmark_sid=None,
                 # benchmark_returns=None,

                 platform='zipline',
                 capital_changes=None,
                 get_pipeline_loader=None,
                 create_event_context=None,
                 **initialize_kwargs):

        # List of trading controls to be used to validate orders.
        self.sessions = calendar.session_in_range(first_session, last_session , include=True)
        self.trading_controls = []

        # # List of account controls to be checked on each bar.
        # self.account_controls = []

        self._recorded_vars = {}
        self.namespace = namespace or {}

        self._platform = platform
        self.logger = None

        # XXX: This is kind of a mess.
        # We support passing a data_portal in `run`, but we need an asset
        # finder earlier than that to look up asset for things like
        # set_benchmark.
        if capital_base <= 0:
            raise ZeroCapitalError()
        self.ledger = Ledger(capital_base)

        if self.data_portal is None:
            if asset_finder is None:
                raise ValueError(
                    "Must pass either data_portal or asset_finder "
                    "to TradingAlgorithm()"
                )
            self.asset_finder = asset_finder
            self.data_portal = DataPortal(asset_finder)

        else:
            # Raise an error if we were passed two different asset finders.
            # There's no world where that's a good idea.
            if asset_finder is not None \
               and asset_finder is not data_portal.asset_finder:
                raise ValueError(
                    "Inconsistent asset_finders in TradingAlgorithm()"
                )
            self.asset_finder = data_portal.asset_finder
            self.data_portal = data_portal

        if creator is not None:
            self._creator = creator
        else:
            slippage = slippage or NoSlippage()
            commission = commission or NoCommission()
            execution_style = execution_style or MarketOrder()
            cancel_policy = cancel_policy or NeverCancel()
            control = control or MaxOrderSize or MaxPositionSize
            self._order_creator = OrderCreator(data_portal, slippage, commission,
                                               execution_style, cancel_policy, control)
        if blotter_class is not None:
            self.blotter = blotter_class
        else:
            blotter_class = blotter_class or SimulationBlotter
            self.blotter = blotter_class(generator=self._creator)

        if broke is not None:
            self.broke_class = broke
        else:
            self.broke_class = Broke(self.blotter,allocation_policy)

        self.restrictions = NoRestrictions() or restrictions


        # Initialize pipe API data.
        self.init_engine(get_pipeline_loader)
        self._pipelines = {}

        self._backwards_compat_universe = None


        self._symbol_lookup_date = None


        # Prepare the algo for initialization
        self.initialized = False

        self.initialize_kwargs = initialize_kwargs or {}


        self._initialize = None
        self._analyze = None

        def get_pipeline(algo_filenames):
            namespace = dict()
            pipelines = []
            for algo_filename in algo_filenames:
                with open(algo_filename, 'r') as f:
                    exec(f.read(), namespace)
                    pipelines.append(namespace['pipe'])
            return pipelines

        engine = SimplePipelineEngine(
                            pipelines,
                            self.asset_finder,
                            self.data_portal,
                            self.restrictions,
                            alternatives=10)
        # metrics
        self.benchmark_sid = benchmark_sid

        self.metrics_tracker = None
        self._last_sync_time = pd.NaT
        self._metrics_set = metrics_set
        if self._metrics_set is None:
            self._metrics_set = load_metrics_set('default')

        self.event_manager = EventManager(create_event_context)

        self.event_manager.add_event(
            zipline.utils.events.Event(
                zipline.utils.events.Always(),
                # We pass handle_data.__func__ to get the unbound method.
                # We will explicitly pass the algorithm to bind it again.
                self.handle_data.__func__,
            ),
            prepend=True,
        )
        # A dictionary of capital changes, keyed by timestamp, indicating the
        # target/delta of the capital changes, along with values
        self.capital_changes = capital_changes or {}

        # A dictionary of the actual capital change deltas, keyed by timestamp
        self.capital_change_deltas = {}

    def init_engine(self, get_loader):
        """
        Construct and store a PipelineEngine from loader.

        If get_loader is None, constructs an ExplodingPipelineEngine
        """
        if get_loader is not None:
            self.engine = SimplePipelineEngine(
                get_loader,
                self.asset_finder,
                self.default_pipeline_domain(self.trading_calendar),
            )
        else:
            self.engine = ExplodingPipelineEngine()

    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self, *args, **kwargs)


    def analyze(self, perf):
        if self._analyze is None:
            return

        with ZiplineAPI(self):
            self._analyze(self, perf)

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        return MinuteSimulationClock(
            self.sim_params.sessions,
            self.trading_calendar
            )

    def _create_benchmark_returns(self):
        try:
            returns = get_benchmark_returns(self.benchmark_sid)
        except Exception as e:
            returns = get_alternative_returns(self.benchmark_sid)
         return returns.loc[self.sessions, :]

    def _create_metrics_tracker(self):
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
        self.trading_client = AlgorithmSimulator(
            self,
            sim_params,
            self.data_portal,
            self._create_clock(),
            self.restrictions,
            universe_func=self._calculate_universe
        )

        metrics_tracker.handle_start_of_simulation(benchmark_source)
        return self.trading_client.transform()

    def _calculate_universe(self):
        # this exists to provide backwards compatibility for older,
        # deprecated APIs, particularly around the iterability of
        # BarData (ie, 'for sid in data`).
        if self._backwards_compat_universe is None:
            self._backwards_compat_universe = (
                self.asset_finder.retrieve_all(self.asset_finder.sids)
            )
        return self._backwards_compat_universe

    def get_generator(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(self.sim_params)

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

    # 根据dt获取change,动态计算，更新数据
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
            Calendar used to compute rules that depend on the trading _calendar.

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
                    '[calendars.US_EQUITIES, calendars.US_FUTURES]'
                ),
            )

        self.add_event(
            make_eventrule(date_rule, time_rule, cal, half_days),
            func,
        )

    @api_method
    def set_benchmark(self, benchmark):
        """Set the benchmark asset.

        Parameters
        ----------
        benchmark : zipline.asset.Asset
            The asset to set as the new benchmark.

        Notes
        -----
        Any dividends payed out for that new benchmark asset will be
        automatically reinvested.
        """
        if self.initialized:
            raise SetBenchmarkOutsideInitialize()

        self.benchmark_sid = benchmark

    @property
    def recorded_vars(self):
        return copy(self._recorded_vars)

    def _sync_last_sale_prices(self, dt=None):
        """Sync the last sale prices on the metrics tracker to a given
        datetime.

        Parameters
        ----------
        dt : datetime
            The time to sync the prices to.

        Notes
        -----
        This call is cached by the datetime. Repeated calls in the same bar
        are cheap.
        """
        if dt is None:
            dt = self.datetime

        if dt != self._last_sync_time:
            self.metrics_tracker.sync_last_sale_prices(
                dt,
                self.data_portal,
            )
            self._last_sync_time = dt

    @property
    def portfolio(self):
        self._sync_last_sale_prices()
        return self.metrics_tracker.portfolio

    @property
    def account(self):
        self._sync_last_sale_prices()
        return self.metrics_tracker.account

    def set_logger(self, logger):
        self.logger = logger

    def on_dt_changed(self, dt):
        """
        Callback triggered by the simulation loop whenever the current dt
        changes.

        Any logic that should happen exactly once at the start of each datetime
        group should happen here.
        """
        self.datetime = dt
        self.blotter.set_date(dt)

    @api_method
    @preprocess(tz=coerce_string(pytz.timezone))
    @expect_types(tz=optional(tzinfo))
    def get_datetime(self, tz=None):
        """
        Returns the current simulation datetime.

        Parameters
        ----------
        tz : tzinfo or str, optional
            The timezone to return the datetime in. This defaults to utc.

        Returns
        -------
        dt : datetime
            The current simulation datetime converted to ``tz``.
        """
        dt = self.datetime
        assert dt.tzinfo == pytz.utc, "algorithm should have a utc datetime"
        if tz is not None:
            dt = dt.astimezone(tz)
        return dt

    @api_method
    def set_slippage(self, us_equities=None):
        """
        Set the slippage models for the simulation.

        Parameters
        ----------
        us_equities : EquitySlippageModel
            The slippage model to use for trading US equities.
        us_futures : FutureSlippageModel
            The slippage model to use for trading US futures.

        Notes
        -----
        This function can only be called during
        :func:`~zipline.api.initialize`.

        See Also
        --------
        :class:`zipline.finance.slippage.SlippageModel`
        """
        if self.initialized:
            raise SetSlippagePostInit()

        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleSlippageModel(
                    asset_type='equities',
                    given_model=us_equities,
                    supported_asset_types=us_equities.allowed_asset_types,
                )
            self.blotter.slippage_models[Equity] = us_equities

    @api_method
    def set_commission(self, us_equities=None, us_futures=None):
        """Sets the commission models for the simulation.

        Parameters
        ----------
        us_equities : EquityCommissionModel
            The commission model to use for trading US equities.
        us_futures : FutureCommissionModel
            The commission model to use for trading US futures.

        Notes
        -----
        This function can only be called during
        :func:`~zipline.api.initialize`.

        See Also
        --------
        :class:`zipline.finance.commission.PerShare`
        :class:`zipline.finance.commission.PerTrade`
        :class:`zipline.finance.commission.PerDollar`
        """
        if self.initialized:
            raise SetCommissionPostInit()

        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleCommissionModel(
                    asset_type='equities',
                    given_model=us_equities,
                    supported_asset_types=us_equities.allowed_asset_types,
                )
            self.blotter.commission_models[Equity] = us_equities

    @api_method
    def set_cancel_policy(self, cancel_policy):
        """Sets the order cancellation policy for the simulation.

        Parameters
        ----------
        cancel_policy : CancelPolicy
            The cancellation policy to use.

        See Also
        --------
        :class:`zipline.api.EODCancel`
        :class:`zipline.api.NeverCancel`
        """
        if not isinstance(cancel_policy, CancelPolicy):
            raise UnsupportedCancelPolicy()

        if self.initialized:
            raise SetCancelPolicyPostInit()

        self.blotter.cancel_policy = cancel_policy

    ####################
    # Trading Controls #
    ####################

    def register_trading_control(self, control):
        """
        Register a new TradingControl to be checked prior to order calls.
        """
        if self.initialized:
            raise RegisterTradingControlPostInit()
        self.trading_controls.append(control)

    @api_method
    def set_max_position_size(self,
                              asset=None,
                              max_shares=None,
                              max_notional=None,
                              on_error='fail'):
        """Set a limit on the number of shares and/or dollar value held for the
        given sid. Limits are treated as absolute values and are enforced at
        the time that the algo attempts to place an order for sid. This means
        that it's possible to end up with more than the max number of shares
        due to splits/dividends, and more than the max notional due to price
        improvement.

        If an algorithm attempts to place an order that would result in
        increasing the absolute value of shares/dollar value exceeding one of
        these limits, raise a TradingControlException.

        Parameters
        ----------
        asset : Asset, optional
            If provided, this sets the guard only on positions in the given
            asset.
        max_shares : int, optional
            The maximum number of shares to hold for an asset.
        max_notional : float, optional
            The maximum value to hold for an asset.
        """
        control = MaxPositionSize(asset=asset,
                                  max_shares=max_shares,
                                  max_notional=max_notional,
                                  on_error=on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_size(self,
                           asset=None,
                           max_shares=None,
                           max_notional=None,
                           on_error='fail'):
        """Set a limit on the number of shares and/or dollar value of any single
        order placed for sid.  Limits are treated as absolute values and are
        enforced at the time that the algo attempts to place an order for sid.

        If an algorithm attempts to place an order that would result in
        exceeding one of these limits, raise a TradingControlException.

        Parameters
        ----------
        asset : Asset, optional
            If provided, this sets the guard only on positions in the given
            asset.
        max_shares : int, optional
            The maximum number of shares that can be ordered at one time.
        max_notional : float, optional
            The maximum value that can be ordered at one time.
        """
        control = MaxOrderSize(asset=asset,
                               max_shares=max_shares,
                               max_notional=max_notional,
                               on_error=on_error)
        self.register_trading_control(control)

    @api_method
    @expect_types(
        restrictions=Restrictions,
        on_error=str,
    )
    def set_asset_restrictions(self, restrictions, on_error='fail'):
        """Set a restriction on which asset can be ordered.

        Parameters
        ----------
        restricted_list : Restrictions
            An object providing information about restricted asset.

        See Also
        --------
        zipline.finance.asset_restrictions.Restrictions
        """
        control = RestrictedListOrder(on_error, restrictions)
        self.register_trading_control(control)
        self.restrictions |= restrictions

    @api_method
    def set_long_only(self, on_error='fail'):
        """Set a rule specifying that this algorithm cannot take short
        positions.
        """
        self.register_trading_control(LongOnly(on_error))

    ##############
    # pipe API
    ##############
    @api_method
    @require_not_initialized(AttachPipelineAfterInitialize())
    @expect_types(
        pipeline=Pipeline,
        name=string_types,
        chunks=(int, Iterable, type(None)),
    )
    def attach_pipeline(self, pipeline, name, chunks=None, eager=True):
        """Register a Pipeline to be computed at the start of each day.

        Parameters
        ----------
        pipeline : Pipeline
            The Pipeline to have computed.
        name : str
            The name of the Pipeline.
        chunks : int or iterator, optional
            The number of days to compute Pipeline results for. Increasing
            this number will make it longer to get the first results but
            may improve the total runtime of the simulation. If an iterator
            is passed, we will run in chunks based on values of the iterator.
            Default is True.
        eager : bool, optional
            Whether or not to compute this Pipeline prior to
            before_trading_start.

        Returns
        -------
        Pipeline : Pipeline
            Returns the Pipeline that was attached unchanged.

        See Also
        --------
        :func:`zipline.api.pipeline_output`
        """
        if chunks is None:
            # Make the first chunk smaller to get more immediate results:
            # (one week, then every half year)
            chunks = chain([5], repeat(126))
        elif isinstance(chunks, int):
            chunks = repeat(chunks)

        if name in self._pipelines:
            raise DuplicatePipelineName(name=name)

        self._pipelines[name] = AttachedPipeline(pipeline, iter(chunks), eager)

        # Return the Pipeline to allow expressions like
        # p = attach_pipeline(pipe(), 'name')
        return pipeline

    @staticmethod
    def default_pipeline_domain(calendar):
        """
        Get a default Pipeline domain for algorithms running on ``_calendar``.

        This will be used to infer a domain for pipelines that only use generic
        datasets when running in the context of a TradingAlgorithm.
        """
        return _DEFAULT_DOMAINS.get(calendar.name, domain.GENERIC)


    ##################
    # End pipe API
    ##################

    @classmethod
    def all_api_methods(cls):
        """
        Return a list of all the TradingAlgorithm API methods.
        """
        return [
            fn for fn in itervalues(vars(cls))
            if getattr(fn, 'is_api_method', False)
        ]

    @api_method
    def record(self, *args, **kwargs):
        """Track and record values each day.

        Parameters
        ----------
        **kwargs
            The names and values to record.

        Notes
        -----
        These values will appear in the performance packets and the performance
        dataframe passed to ``analyze`` and returned from
        :func:`~zipline.run_algorithm`.
        """
        # Make 2 objects both referencing the same iterator
        args = [iter(args)] * 2

        # Zip generates list entries by calling `next` on each iterator it
        # receives.  In this case the two iterators are the same object, so the
        # call to next on args[0] will also advance args[1], resulting in zip
        # returning (a,b) (c,d) (e,f) rather than (a,a) (b,b) (c,c) etc.
        positionals = zip(*args)
        for name, value in chain(positionals, kwargs.items()):
            self._recorded_vars[name] = value

    def __repr__(self):
        """
        N.B. this does not yet represent a string that can be used
        to instantiate an exact copy of an algorithm.

        However, it is getting close, and provides some value as something
        that can be inspected interactively.
        """
        return """
{class_name}(
    capital_base={capital_base}
    sim_params={sim_params},
    initialized={initialized},
    slippage_models={slippage_models},
    commission_models={commission_models},
    oms={oms},
    recorded_vars={recorded_vars})
""".strip().format(class_name=self.__class__.__name__,
                   capital_base=self.sim_params.capital_base,
                   sim_params=repr(self.sim_params),
                   initialized=self.initialized,
                   slippage_models=repr(self.blotter.slippage_models),
                   commission_models=repr(self.blotter.commission_models),
                   blotter=repr(self.blotter),
                   recorded_vars=repr(self.recorded_vars))
