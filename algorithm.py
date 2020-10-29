# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
# error module
from error.errors import (
    RegisterTradingControlPostInit,
    RegisterAccountControlPostInit,
    SetBenchmarkOutsideInitialize,
    SetCancelPolicyPostInit,
    SetCommissionPostInit,
    SetSlippagePostInit,
    ZeroCapitalError
)
# finance module
from finance.ledger import Ledger
from finance.slippage import NoSlippage
from finance.execution import MarketOrder
from finance.commission import NoCommission
from finance.restrictions import NoRestrictions
from finance.control import (
    MaxPositionSize,
    MaxOrderSize,
    LongOnly,
    NoControl,
    NetLeverage
)
# pb module
from pb.broker import Broker
from pb.division import Division
from pb.generator import Generator
from pb.blotter import SimulationBlotter
from pb.underneath import UncoverAlgorithm
# pipe engine
from pipe.final import Final
from pipe.pipeline import Pipeline
from pipe.engine import SimplePipelineEngine
# trade simulation
from trade.clock import MinuteSimulationClock
from trade.tradesimulation import AlgorithmSimulator
# risk management
from risk.allocation import Equal
from risk.alert import UnionRisk, NoRisk
from risk.fuse import Fuse
# metric module
from metric import default_metrics
from metric.tracker import MetricsTracker, _ClassicRiskMetrics
# util api method
from util.wrapper import api_method
from util.api_support import AlgoAPI
from util.events import EventManager, Event, Always
# gateway api
# from gateway.asset.finder import init_finder
# from gateway.driver.data_portal import portal
from gateway.driver.benchmark_source import BenchmarkSource


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
                 pipelines,
                 on_error='log',
                 # finance module
                 slippage=None,
                 commission=None,
                 restrictions=None,
                 execution_style=None,
                 trading_controls=None,
                 account_controls=None,
                 # pd
                 uncover_algo=None,
                 divison_model=None,
                 # pipe API
                 scripts=None,
                 final=None,
                 disallow_righted=False,
                 disallowed_violation=True,
                 # risk
                 risk_models=None,
                 risk_fuse=None,
                 risk_allocation=None,
                 # metric
                 _analyze=None,
                 metrics_set=None,
                 metrics_tracker=None,
                 # extra
                 namespace=None,
                 initialize=None,
                 platform='px_trader',
                 _handle_data=None,
                 logger=None,
                 before_trading_start=None,
                 create_event_context=None,
                 **initialize_kwargs):

        assert sim_params.capital_base > 0, ZeroCapitalError()
        self.sim_params = sim_params
        self.benchmark_returns = self._calculate_benchmark_returns()
        # # data interface
        # self.data_portal = portal
        # self.asset_finder = init_finder()
        # set ledger
        risk_models = risk_models or NoRisk()
        risk_fuse = risk_fuse or Fuse()
        self.ledger = Ledger(sim_params, risk_models, risk_fuse)
        # init blotter module : transform order to txn via slippage commission execution
        self.slippage = slippage or NoSlippage()
        self.commission = commission or NoCommission()
        self.execution_style = execution_style or MarketOrder()
        self.blotter = SimulationBlotter(self.commission,
                                         self.slippage,
                                         self.execution_style)
        # init generator
        self.uncover_algo = uncover_algo or UncoverAlgorithm()
        self.trading_controls = trading_controls or NoControl()
        self.division_model = divison_model or Division(
                                            self.uncover_algo,
                                            self.trading_controls,
                                            sim_params.per_capital)
        self.generator = self._create_generator()
        # Initialize pipe_engine API
        self.final = final or Final()
        # restrictions
        restrictions = restrictions or NoRestrictions()
        self.restrictions = restrictions
        self.pipelines = self.validate_pipeline(pipelines)
        self.pipeline_engine = self._construct_pipeline_engine(
                                                    disallow_righted,
                                                    disallowed_violation)
        # set allocation policy
        risk_allocation = risk_allocation or Equal()
        # init broker
        self.broker = self._initialize_broker(risk_allocation)
        # set account controls
        self.account_controls= account_controls
        # metric tracker
        if metrics_set is not None:
            self._metrics_set = metrics_set
        else:
            self._metrics_set = default_metrics()
        if metrics_tracker is not None:
            self.tracker = metrics_tracker
        else:
            self.tracker = self._create_metrics_tracker()
        # analyse
        self._analyze = _ClassicRiskMetrics()

        # set event manager
        self.event_manager = EventManager(create_event_context)
        self.event_manager.add_event(
            Event(Always(), _handle_data),
            prepend=True,
        )

        self.initialized = False
        self.logger = logger
        self._platform = platform
        self.on_error = on_error
        self.initialize_kwargs = initialize_kwargs or {}
        self._recorded_vars = {}
        self.namespace = namespace or {}

        # set default func
        def noop(*args, **kwargs):
            pass

        self._before_trading_start = before_trading_start or noop
        self._initialize = initialize or noop

    def _create_generator(self):
        """
        :param dist: distribution module (simulate_dist , simulate_ticker)
                    to generate price timeseries or ticker timeseries
        generator --- compute capital or position to transactions
        """
        # generator
        delay = self.sim_params.delay
        generator_class = Generator(delay,
                                    self.blotter,
                                    self.division_model)
        return generator_class

    @staticmethod
    def validate_pipeline(pipelines):
        # pipelines = []
        # for script_file in scripts:
        #     name = script_file.rsplit('.')[-2]
        #     with open(script_file, 'r') as f:
        #         exec(f.read(), self.namespace)
        #         pipelines.append(self.namespace[name])
        assert pipelines is not None, 'pipelines must validate and composed by terms '
        pipes = [pipelines if isinstance(pipelines, Pipeline) else pipelines]
        return pipes

    def _construct_pipeline_engine(self,
                                   righted,
                                   violation):
        """
        Construct and store a PipelineEngine from loader.
        If get_loader is None, constructs an ExplodingPipelineEngine
        """
        try:
            engine = SimplePipelineEngine(self.pipelines,
                                          self.final,
                                          self.restrictions,
                                          righted,
                                          violation)
            return engine
        except Exception as e:
            raise ValueError('initialization error %s' % e)

    def _initialize_broker(self, allocation_model):
        """
        allocation_model : allocate capitals among assets
        broker : combine pipe_engine and generator
        """
        broker_class = Broker(self.pipeline_engine,
                              self.generator,
                              allocation_model)
        return broker_class

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        return MinuteSimulationClock(self.sim_params)

    def before_trading_start(self, data):

        self._before_trading_start(data)

    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with AlgoAPI(self):
            self._initialize(self, *args, **kwargs)

    def _create_simulation(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        if not self.initialized:
            self.initialize(**self.initialize_kwargs)
            self.initialized = True
        clock = self._create_clock()
        # universe_func=self._calculate_universe
        self.trading_client = AlgorithmSimulator(self, clock)
        return self.trading_client.transform()

    def yield_simulation(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_simulation()

    def _create_metrics_tracker(self):
        """
            measure metrics of ledger
        """
        return MetricsTracker(
            # ledger=self.ledger,
            sim_params=self.sim_params,
            benchmark_rets=self.benchmark_returns,
            metrics_sets=self._metrics_set
        )

    def _calculate_benchmark_returns(self):
        """
            benchmark returns
        """
        source = BenchmarkSource(self.sim_params.sessions)
        benchmark = self.sim_params.benchmark
        returns = source.calculate_returns(benchmark)
        return returns

    @staticmethod
    def _create_daily_stats(perfs):
        # create daily and cumulative stats frame
        print('perfs', perfs)
        daily_perfs = []
        for perf in perfs:
            if 'daily_perf' in perf:
                perf['daily_perf'].update(perf['cumulative_risk_metrics'])
                daily_perfs.append(perf['daily_perf'])

        # daily_dts = pd.DatetimeIndex(
        #     [p['period_close'] for p in daily_perfs], tz='UTC'
        # )
        # daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
        daily_stats = pd.DataFrame(daily_perfs)
        print('daily_stats', daily_stats)
        return daily_stats

    def analyse(self, perf):
        with AlgoAPI(self):
            stats = self._analyze.end_of_simulation(
                perf,
                self.ledger,
                self.benchmark_returns)
        return stats

    def run(self):
        """Run the algorithm.
        """
        # Create px_trade and loop through simulated_trading.
        # Each iteration returns a perf dictionary
        perfs = []
        for perf in self.yield_simulation():
            perfs.append(perf)
        # convert perf dict to pandas frame
        daily_stats = self._create_daily_stats(perfs)
        analysis = self.analyse(daily_stats)
        return analysis

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

    ##############
    # Pipeline API
    ##############

    def set_pipeline(self,
                     righted=True,
                     violated=True):
        if self.initialized:
            raise AttributeError
        self.pipeline_engine = self._construct_pipeline_engine(righted, violated)

    def set_pipeline_final(self, final):
        self.final = final

    def attach_pipeline(self, terms, ump_pickers=None):
        """Register a pipeline to be computed at the start of each day.

        Parameters
        ----------
        terms : components of Pipeline
            The pipeline to have computed.

        ump_pickers :  ump_pickers intended for short operation

        Returns
        -------
        pipeline : Pipeline
            Returns the pipeline that was attached unchanged.

        See Also
        --------
        :func:`zipline.api.pipeline_output`
        """
        # Return the pipeline to allow expressions like
        # p = attach_pipeline(Pipeline(), 'name')
        pipeline = Pipeline(terms, ump_pickers)
        self.pipelines.append(pipeline)

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

    def set_uncover_policy(self, uncover_model):
        if self.initialized:
            raise InterruptedError
        self.uncover_algo = uncover_model

    def set_allocation_policy(self, allocation_model):
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
    def set_max_position_size(self, max_notional):
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
        """
        control = MaxPositionSize(max_notional,
                                  on_error=self.on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_size(self,
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
        control = MaxOrderSize(max_notional,
                               sliding_window=window,
                               on_error=self.on_error)
        self.register_trading_control(control)

    @api_method
    def set_long_only(self):
        """Set a rule specifying that this algorithm cannot take short
        positions.
        """
        self.register_trading_control(LongOnly())

    ####################
    # Account Controls #
    ####################

    def register_account_control(self, control):
        """
        Register a new AccountControl to be checked on each bar.
        """
        if self.initialized:
            raise RegisterAccountControlPostInit()
        self.account_controls.append(control)

    @api_method
    def set_net_leverage(self, net_leverage):
        """Set a limit on the maximum leverage of the algorithm.

        Parameters
        ----------
        net_leverage : float
            The net leverage for the algorithm. If not provided there will
            be 1.0
        """
        control = NetLeverage(base_leverage=net_leverage)
        self.register_account_control(control)

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


if __name__ == '__main__':

    from trade.params import create_simulation_parameters
    trading_params = create_simulation_parameters(start='2019-09-01', end='2019-09-20')
    print('trading_params', trading_params)
    # set pipeline term
    from pipe.term import Term
    kw = {'window': (5, 10), 'fields': ['close']}
    cross_term = Term('cross', kw)
    # print('sma_term', cross_term)
    kw = {'fields': ['close'], 'window': 5, 'final': True}
    break_term = Term('break', kw, cross_term)
    # print(break_term.dependencies)
    # set pipeline
    # pipeline = Pipeline([cross_term])
    pipeline = Pipeline([break_term])
    # pipeline = Pipeline([break_term, cross_term])
    # initialize trading algo
    trading = TradingAlgorithm(trading_params, pipeline)
    # run algorithm
    trading.run()

