# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, warnings
from error.errors import (
    IncompatibleCommissionModel,
    IncompatibleSlippageModel,
    RegisterTradingControlPostInit,
    SetBenchmarkOutsideInitialize,
    SetCancelPolicyPostInit,
    SetCommissionPostInit,
    SetSlippagePostInit,
    UnsupportedCancelPolicy,
    ZeroCapitalError
)
from finance.cancel_policy import NeverCancel, CancelPolicy
from finance.commission import NoCommission
from finance.control import (
    LongOnly,
    MaxOrderSize,
    MaxPositionSize,
)
from finance.execution import MarketOrder
from finance.ledger import Ledger
from finance.oms.blotter import SimulationBlotter
from finance.oms.creator import OrderCreator
from finance.restrictions import NoRestrictions
from finance.slippage import NoSlippage
from gateway.asset.assets import Equity
from gateway.driver.benchmark import (
    get_benchmark_returns,
    get_alternative_returns
)
from gateway.driver.data_portal import DataPortal
from gens.clock import MinuteSimulationClock
from gens.tradesimulation import AlgorithmSimulator
from pipe.engine import SimplePipelineEngine
from risk.allocation import Equal
from trade.broke import Broker
from risk.metrics import default_metrics
from risk.metrics.tracker import MetricsTracker
from utils.api_support import (
    api_method,
    ZiplineAPI,
)
from utils.events import EventManager, Event, Always


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
                 sim_params,
                 namespace=None,
                 # dataApi --- only entrance of backtest
                 data_portal=None,
                 asset_finder=None,
                 # finance module intended for order object
                 slippage=None,
                 execution_style=None,
                 commission=None,
                 cancel_policy=None,
                 control=None,
                 restrictions=None,
                 # order generator
                 creator=None,
                 # blotter --- transform order to transaction , delay --- means transfer put to call
                 delay=None,
                 blotter_class=None,
                 # pipeline API
                 scripts=None,
                 alternatives=None,
                 # broker --- combine creator,blotter pipe_engine together
                 broker_engine=None,
                 # capital allocation and portfolio management
                 risk_management=None,
                 # metrics
                 _analyze=None,
                 metrics_set=None,
                 metrics_tracker=None,
                 # os property
                 platform='px_trader',
                 create_event_context=None,
                 logger=None,
                 **initialize_kwargs):

        self.sim_params = sim_params
        # set ledger with capital base
        if sim_params.capital_base <= 0:
            raise ZeroCapitalError()
        self.ledger = Ledger(sim_params.capital_base)
        # set benchmark returns
        self.benchmark_returns = self._calculate_benchmark_returns()
        # set data_portal
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

        # order creator to create orders by capital or amount
        self.trading_controls = control or [MaxOrderSize, MaxPositionSize]

        if creator is not None:
            self._creator = creator
        else:
            slippage = slippage or NoSlippage()
            commission = commission or NoCommission()
            execution_style = execution_style or MarketOrder()
            cancel_policy = cancel_policy or NeverCancel()
            # List of trading controls to be used to validate orders.
            # trading_controls = control or [MaxOrderSize, MaxPositionSize]
            # List of account controls to be checked on each bar.
            # self.account_controls = []
            self._creator = OrderCreator(data_portal,
                                         slippage,
                                         commission,
                                         execution_style,
                                         cancel_policy,
                                         self.trading_controls)

        # simulation blotter
        self.blotter = self._create_blotter(delay)
        # restrictions , alternative
        self.restrictions = NoRestrictions() or restrictions
        self.alternative = alternatives or 10
        # Initialize pipe_engine API
        self.pipeline_engine = self.init_engine(scripts)
        # create generator --- initialized = True
        self.initialized = False

        # capital allocation
        risk_management = Equal or risk_management
        # broker --- combine pipe_engine and blotter ; when live trading broker ---- xtp
        self.broke_class = self._create_broker(broker_engine, risk_management)

        # metrics_set and initialize metrics tracker
        if metrics_set is not None:
            self._metrics_set = metrics_set
        else:
            self._metrics_set = default_metrics()
        if metrics_tracker is not None:
            self.metrics_tracker = metrics_tracker
        else:
            self.metrics_tracker = self._create_metrics_tracker()
        # analyze the metrics
        self._analyze = _analyze

        # set event manager
        self.event_manager = EventManager(create_event_context)
        self.event_manager.add_event(
            Event(Always(), self.handle_data.__func__),
            prepend=True,
        )
        # set additional attr
        self.logger = logger
        self._platform = platform
        self.initialize_kwargs = initialize_kwargs or {}
        self._recorded_vars = {}
        self.namespace = namespace or {}

    def _calculate_universe(self):
        # this exists to provide backwards compatibility for older,
        # deprecated APIs, particularly around the iterability of
        # BarData (ie, 'for sid in data`).
        if self._backwards_compat_universe is None:
            self._backwards_compat_universe = (
                self.asset_finder.retrieve_all(self.asset_finder.sids)
            )
        return self._backwards_compat_universe

    def _create_blotter(self, blotter_class, delay):
        """
            simulation blotter
            function --- transform order to txn via different ways(capital , amount ,dual)
        """
        if blotter_class is not None:
            simulation_blotter = blotter_class
        else:
            simulation_blotter = SimulationBlotter(self._creator, delay)
        return simulation_blotter

    def _create_broker(self, broker, risk):
        """
            broker --- xtp
        """
        if broker is not None:
            broke_class = broker
        else:
            broke_class = Broker(self.blotter, risk)
        return broke_class

    def init_engine(self, scripts):
        """
        Construct and store a PipelineEngine from loader.

        If get_loader is None, constructs an ExplodingPipelineEngine
        """
        pipelines = []
        for script_file in scripts:
            name = script_file.rsplit('.')[-2]
            with open(script_file, 'r') as f:
                exec(f.read(), self.namespace)
                pipelines.append(self.namespace[name])
        try:
            engine = SimplePipelineEngine(
                                pipelines,
                                self.asset_finder,
                                self.data_portal,
                                self.restrictions,
                                self.alternatives
                                        )
            return engine
        except Exception as e:
            raise ValueError('initialization error %s' % e)

    def _create_metrics_tracker(self):
        return MetricsTracker(
            ledger=self.ledger,
            sim_params=self.sim_params,
            metrics_sets=self._metrics_set
        )

    def _calculate_benchmark_returns(self):
        benchmark = self.sim_params.benchmark
        benchmark_symbols = self.asset_finder.retrieve_index_symbols()
        if benchmark in benchmark_symbols:
            returns = get_benchmark_returns(benchmark)
        else:
            returns = get_alternative_returns(benchmark)
        return returns.loc[self.sim_params.sessions, :]

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        return MinuteSimulationClock(
                        self.sim_params
                        )

    def _create_generator(self, sim_params):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        if not self.initialized:
            self.initialize(**self.initialize_kwargs)
            self.initialized = True
        self.trading_client = AlgorithmSimulator(
            self,
            sim_params
            # universe_func=self._calculate_universe
        )
        return self.trading_client.transform()

    def get_generator(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(self.sim_params)

    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self, *args, **kwargs)

    def run(self):
        """Run the algorithm.
        """
        # Create px_trade and loop through simulated_trading.
        # Each iteration returns a perf dictionary
        try:
            perfs = []
            for perf in self.get_generator():
                perfs.append(perf)
            # convert perf dict to pandas frame
            daily_stats = self._create_daily_stats(perfs)
            self.analyze(daily_stats)
        finally:
            self.data_portal = None
            self.metrics_tracker = None
        return daily_stats

    @staticmethod
    def _create_daily_stats(perfs):
        daily_perfs = []
        # create daily and cumulative stats frame
        for perf in perfs:
            if 'daily_perf' in perf:
                perf['daily_perf'].update(perf['cumulative_risk_metrics'])
                daily_perfs.append(perf['daily_perf'])

        daily_dts = pd.DatetimeIndex(
            [p['period_close'] for p in daily_perfs], tz='UTC'
        )
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
        return daily_stats

    def analyze(self, perf):
        if self._analyze is None:
            return

        with ZiplineAPI(self):
            self._analyze(self, perf)

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
            'arena': 'china',
            # 'data_frequency': self.sim_params.data_frequency,
            'start': self.sim_params.sessions[0],
            'end': self.sim_params.sessions[-1],
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
            Event(rule, callback),
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

    def set_logger(self, logger):
        self.logger = logger

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
        self.sim_params.benchmark = benchmark
        if self.initialized:
            raise SetBenchmarkOutsideInitialize()

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
    def set_commission(self, commission_class):
        """Sets the commission models for the simulation.

        Parameters
        ----------
        commission_class : CommissionModel instance
            The commission model to use for trading US equities.

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

        self.blotter.commission_model = commission_class

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

    def set_restrictions(self, restricted_list):
        """Set a restriction on which asset can be ordered.

        Parameters
        ----------
        restricted_list : Restrictions
            An object providing information about restricted asset.

        See Also
        --------
        zipline.finance.restrictions.Restrictions
        """
        if self.initialized:
            raise SetCancelPolicyPostInit()

        self.pipeline_engine.restricted_rules = restricted_list

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
        max_notional : float, optional
                The maximum value to hold for an asset.
        on_error : int, optional
                The maximum number of shares to hold for an asset.
        """
        control = MaxPositionSize(max_notional=max_notional,
                                  on_error=on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_size(self,
                           max_shares=None,
                           on_error='fail'):
        """Set a limit on the number of shares and/or dollar value of any single
        order placed for sid.  Limits are treated as absolute values and are
        enforced at the time that the algo attempts to place an order for sid.

        If an algorithm attempts to place an order that would result in
        exceeding one of these limits, raise a TradingControlException.

        Parameters
        ----------
        max_shares : int, optional
            The maximum number of shares that can be ordered at one time.
        on_error : fail or log
        """
        control = MaxOrderSize(max_shares=max_shares,
                               on_error=on_error)
        self.register_trading_control(control)

    @api_method
    def set_long_only(self, on_error='fail'):
        """Set a rule specifying that this algorithm cannot take short
        positions.
        """
        self.register_trading_control(LongOnly(on_error))

    ##################
    # End pipe API
    ##################

    @classmethod
    def all_api_methods(cls):
        """
        Return a list of all the TradingAlgorithm API methods.
        """
        return [
            fn for fn in vars(cls).items
            if getattr(fn, 'is_api_method', False)
        ]

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
                   blotter=repr(self.blotter)
                   # recorded_vars=repr(self.recorded_vars))
                   )
