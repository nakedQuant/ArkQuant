# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
# error module
from error.errors import (
    IncompatibleSlippageModel,
    RegisterTradingControlPostInit,
    SetBenchmarkOutsideInitialize,
    SetCancelPolicyPostInit,
    SetCommissionPostInit,
    SetSlippagePostInit,
    UnsupportedCancelPolicy,
    ZeroCapitalError
)
# gateway api
from gateway.asset.assets import Equity
from gateway.asset._finder import init_finder
from gateway.driver.data_portal import portal
from gateway.driver.benchmark import (
    get_benchmark_returns,
    get_foreign_benchmark_returns
)
# finance module
from finance.ledger import Ledger
from finance.slippage import NoSlippage
from finance.execution import MarketOrder
from finance.commission import NoCommission
from finance.restrictions import NoRestrictions, UnionRestrictions
from finance.control import (
    NoControl,
    MaxPositionSize,
    MaxOrderAmount,
    MaxOrderProportion
)
# risk management
from risk.allocation import Equal
from risk.alert import UnionRisk, NoRisk
# pb module
from pb.dist import BaseDist
from pb.broker import Broker
from pb.generator import Generator
from pb.blotter import SimulationBlotter
from pb.division import PositionDivision, CapitalDivision
# pipe engine
from pipe.engine import SimplePipelineEngine
# trade simulation
from trade.clock import MinuteSimulationClock
from trade.tradesimulation import AlgorithmSimulator
# metric module
from metric import default_metrics
from metric.tracker import MetricsTracker
# util api method
from util.wrapper import  api_method
from util.api_support import ZiplineAPI
from util.events import EventManager, Event, Always


class TradingAlgorithm(object):
    """A class that represents a trading strategy and parameters to execute
    the strategy.

    Parameters
    ----------
    initialize : callable[context -> None], optional
        Function that is called at the start to setup the initial context.
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
    create_event_context : callable[BarData -> context manager], optional
        A function used to create a context mananger that wraps the
        execution of all events that are scheduled for a bar.
        This function will be passed the data for the bar and should
        return the actual context manager that will be entered.
    platform : str, optional
        The platform the nakedquant is running on. This can be queried for
        in the nakedquant with ``get_environment``. This allows algorithms
        to conditionally execute code based on platform it is running on.
        default: 'zipline'

    量化交易系统:
        a.策略识别（搜索策略 ， 挖掘优势 ， 交易频率）
        b.回溯测试（获取数据 ， 分析策略性能 ，剔除偏差）
        c.交割系统（经纪商接口 ，交易自动化 ， 交易成本最小化）
        d.风险管理（最优资本配置 ， 最优赌注或者凯利准则 ， 海龟仓位管理）
    """
    def __init__(self,
                 sim_params,
                 on_error='log',
                 # finance module
                 control=None,
                 slippage=None,
                 commission=None,
                 restrictions=None,
                 execution_style=None,
                 # dist
                 dist=None,
                 # pipe API
                 pipelines=None,
                 alternatives=10,
                 disallow_righted=False,
                 disallowed_violation=True,
                 # risk
                 risk_models=None,
                 risk_allocation=None,
                 # metric
                 _analyze=None,
                 metrics_set=None,
                 metrics_tracker=None,
                 # os property
                 namespace=None,
                 platform='px_trader',
                 create_event_context=None,
                 logger=None,
                 **initialize_kwargs):

        assert sim_params.capital_base <= 0, ZeroCapitalError()
        self.sim_params = sim_params
        self.on_error = on_error
        # set benchmark returns
        self.benchmark_returns = self._calculate_benchmark_returns()
        # data interface
        self.data_portal = portal
        self.asset_finder = init_finder()
        # restrictions
        restrictions = restrictions or NoRestrictions()
        # set ledger
        risk_models = risk_models or NoRisk()
        self.ledger = Ledger(sim_params, risk_models)
        # init blotter
        self.slippage = slippage or NoSlippage()
        self.commission = commission or NoCommission()
        self.execution_style = execution_style or MarketOrder()
        self.blotter = self._init_blotter()
        # init generator
        self.generator = self._init_generator(dist)
        # Initialize pipe_engine API
        self.pipelines = pipelines
        self.restrictions = restrictions
        self.pipe_engine = self._init_engine(
                                            alternatives,
                                            disallow_righted,
                                            disallowed_violation)
        # set controls
        self.trading_controls = control or NoControl()
        # set allocation policy
        risk_allocation = risk_allocation or Equal()
        # init broker
        self.broker = Broker(risk_allocation)

        if metrics_set is not None:
            self._metrics_set = metrics_set
        else:
            self._metrics_set = default_metrics()
        if metrics_tracker is not None:
            self.metrics_tracker = metrics_tracker
        else:
            self.metrics_tracker = self._create_metrics_tracker()
        # analyze the metric
        self._analyze = _analyze
        # set event manager
        self.event_manager = EventManager(create_event_context)
        # self.event_manager.add_event(
        #     Event(Always(), self.handle_data.__func__),
        #     prepend=True,
        # )
        self.initialized = False
        # set additional attr
        self.logger = logger
        self._platform = platform
        self.initialize_kwargs = initialize_kwargs or {}
        self._recorded_vars = {}
        self.namespace = namespace or {}

    def _init_blotter(self):
        """
            nakedquant blotter
            function --- transform order to txn via different ways(capital , amount ,dual)
        """
        blotter = SimulationBlotter(self.commission,
                                    self.slippage,
                                    self.execution_style)
        return blotter

    def _init_generator(self, dist):
        # set division
        divisions = [CapitalDivision(dist), PositionDivision(dist)]
        # generator
        generator_class = Generator(self.sim_params.delay,
                                    self.blotter,
                                    divisions)
        return generator_class

    def _init_engine(self,
                     alternatives,
                     righted,
                     violation):
        """
        Construct and store a PipelineEngine from loader.

        If get_loader is None, constructs an ExplodingPipelineEngine
        """
        # pipelines = []
        # for script_file in scripts:
        #     name = script_file.rsplit('.')[-2]
        #     with open(script_file, 'r') as f:
        #         exec(f.read(), self.namespace)
        #         pipelines.append(self.namespace[name])
        try:
            engine = SimplePipelineEngine(self.pipelines,
                                          self.restrictions,
                                          alternatives,
                                          righted,
                                          violation)
            return engine
        except Exception as e:
            raise ValueError('initialization error %s' % e)

    def _init_broker(self, allocation_model):
        """
            broker --- xtp
        """
        broker_class = Broker(self.pipe_engine,
                              self.generator,
                              self.trading_controls,
                              allocation_model)
        return broker_class

    def _create_metrics_tracker(self):
        return MetricsTracker(
            ledger=self.ledger,
            sim_params=self.sim_params,
            metrics_sets=self._metrics_set
        )

    def _calculate_benchmark_returns(self):
        benchmark = self.sim_params.benchmark
        try:
            returns = get_benchmark_returns(benchmark)
        except Exception as e:
            print('error:', e)
            returns = get_foreign_benchmark_returns(benchmark)
        return returns.loc[self.sim_params.sessions, :]

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        return MinuteSimulationClock(
                        self.sim_params
                        )

    def _create_simulation(self, sim_params):
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

    def yield_simulation(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_simulation(self.sim_params)

    def _calculate_universe(self):
        # this exists to provide backwards compatibility for older,
        # deprecated APIs, particularly around the iterability of
        # BarData (ie, 'for sid in data`).
        if self._backwards_compat_universe is None:
            self._backwards_compat_universe = (
                self.asset_finder.retrieve_all(self.asset_finder.sids)
            )
        return self._backwards_compat_universe

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
            for perf in self.yield_simulation():
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
                  The arena from the nakedquant parameters. This will normally
                  be ``'backtest'`` but some systems may use this distinguish
                  live trading from backtesting.
              data_frequency : {'daily', 'minute'}
                  data_frequency tells the algorithm if it is running with
                  daily data or minute data.
              start : datetime
                  The start date for the nakedquant.
              end : datetime
                  The end date for the nakedquant.
              capital_base : float
                  The starting capital for the nakedquant.
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
            'data_frequency': self.sim_params.data_frequency,
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

    ####################
    # PipeEngine Control
    ####################

    def set_pipeline_engine(self,
                            num_choices,
                            righted=True,
                            violated=True):
        if self.initialized:
            raise AttributeError
        self.pipe_engine = self._init_engine(num_choices, righted, violated)

    ####################
    # Finance Controls #
    ####################

    def set_risk_models(self, risk_models):
        self.ledger.risk_alert = UnionRisk(risk_models)

    @api_method
    def set_slippage(self, slippage_class):
        """
        Set the slippage models for the nakedquant.

        Parameters
        ----------
        slippage_class : EquitySlippageModel
            The slippage model to use for trading US equities.

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
        self.slippage = slippage_class

    @api_method
    def set_commission(self, commission_class):
        """Sets the commission models for the trading

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
        self.commission = commission_class

    def set_execution_style(self, execution_model):
        """Sets the execution_style models for the trading
        :param execution_model --- execution_style
        """
        if self.initialized:
            raise AttributeError
        self.execution_style = execution_model

    def set_restrictions(self, restricted_list):
        """Set a restriction on which asset can be ordered.

        Parameters
        ----------
        restricted_list : list of Restrictions
            An object providing information about restricted asset.

        See Also
        --------
        zipline.finance.restrictions.Restrictions
        """
        if self.initialized:
            raise SetCancelPolicyPostInit()
        self.restrictions = restricted_list

    ####################
    # Pb Controls #
    ####################

    def set_dist_func(self, dist_model):
        assert isinstance(dist_model, BaseDist), \
            'dist_model must be subclass of BaseDist'
        self.generator = self._init_generator(dist_model)

    def set_risk_allocation(self, allocation_model):
        if self.initialized:
            raise AttributeError
        self.broker.capital_model = allocation_model

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
                              window,
                              max_notional):
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
        window : measure_window
        max_notional : float, optional
                The maximum value to hold for an asset.
        on_error : int, optional
                The maximum number of shares to hold for an asset.
        """
        control = MaxPositionSize(window=window,
                                  max_notional=max_notional,
                                  on_error=self.on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_amount(self,
                             window,
                             max_notional):
        """Set a limit on the number of shares and/or dollar value of any single
        order placed for sid.  Limits are treated as absolute values and are
        enforced at the time that the algo attempts to place an order for sid.

        If an algorithm attempts to place an order that would result in
        exceeding one of these limits, raise a TradingControlException.

        Parameters
        ----------
        window : int, optional
            window  that measure the max amount can be ordered at one time
        max_notional : float
        """
        control = MaxOrderAmount(window=window,
                                 max_notional=max_notional,
                                 on_error=self.on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_proportion(self, max_notional):
        """Set a rule specifying that this algorithm cannot take short
        positions.
        """
        control = MaxOrderProportion(max_notional=max_notional,
                                     on_error=self.on_error)
        self.register_trading_control(control)

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
    broker={broker})
""".strip().format(class_name=self.__class__.__name__,
                   capital_base=self.sim_params.capital_base,
                   sim_params=repr(self.sim_params),
                   initialized=self.initialized,
                   slippage_models=repr(self.blotter.slippage),
                   commission_models=repr(self.blotter.commission),
                   broker=repr(self.broker)
                   )


__all__ = ['TradingAlgorithm']
