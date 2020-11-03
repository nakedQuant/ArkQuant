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
)
# finance module
from finance.ledger import Ledger
from finance.slippage import NoSlippage, FixedBasisPointSlippage
from finance.execution import MarketOrder, LimitOrder
from finance.commission import NoCommission, Commission
from finance.restrictions import NoRestrictions, StatusRestrictions, DataBoundsRestrictions
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
from pb.underneath import SimpleUncover
# pipe engine
from pipe.final import Final
from pipe.pipeline import Pipeline
from pipe.engine import SimplePipelineEngine
# trade simulation
from trade.clock import MinuteSimulationClock
from trade.tradesimulation import AlgorithmSimulator
# risk management
from risk.allocation import Equal, Turtle
from risk.alert import Risk, NoRisk, PositionLossRisk
from risk.fuse import BaseFuse, NoFuse
# metric module
from metric import default_metrics
from metric.tracker import MetricsTracker, _ClassicRiskMetrics
# util api method
from util.wrapper import api_method
from util.api_support import AlgoAPI
from util.events import EventManager, Event, Always
# benchmark source
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
        The platform the ArkQuant is running on. This can be queried for
        in the ArkQuant with ``get_environment``. This allows algorithms
        to conditionally execute code based on platform it is running on.
    analyze : callable[(context, pd.DataFrame) -> None], optional
        The analyze function to use for the algorithm. This function is called
        once at the end of the backtest and is passed the context and the
        performance data.

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
                 slippage_model=None,
                 commission_model=None,
                 restrictions=None,
                 execution_style=None,
                 trading_controls=None,
                 account_controls=None,
                 ledger=None,
                 # pd
                 underneath_model=None,
                 broker=None,
                 # pipe API
                 pipelines=None,
                 final=None,
                 disallow_righted=True,
                 disallowed_violation=True,
                 engine=None,
                 # risk
                 risk_fuse=None,
                 risk_models=None,
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
                 analyse=None,
                 **initialize_kwargs):

        self.sim_params = sim_params
        self.benchmark_returns = self._calculate_benchmark_returns()
        # set finance module
        self.slippage = slippage_model or NoSlippage()
        self.commission = commission_model or NoCommission()
        self.execution_style = execution_style or MarketOrder()
        self.restrictions = restrictions or NoRestrictions()
        self.account_controls = account_controls or []
        self.trading_controls = trading_controls or [NoControl()]
        self.ledger = ledger
        # set engine module
        self.violated = disallowed_violation
        self.righted = disallow_righted
        self.final = final or Final()
        self.pipelines = pipelines or []
        self.pipeline_engine = engine
        # set risk module
        self.risk_allocation = risk_allocation or Equal()
        self.risk_models = risk_models or NoRisk()
        self.risk_fuse = risk_fuse or NoFuse()
        # set pd module
        self.underneath_module = underneath_model or SimpleUncover()
        self.broker = broker

        # set metrics module
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

        def noop(*args, **kwargs):
            pass

        self._initialize = initialize or noop
        self._before_trading_start = before_trading_start or noop

    def _calculate_benchmark_returns(self):
        """
            benchmark returns
        """
        source = BenchmarkSource(self.sim_params.sessions)
        benchmark = self.sim_params.benchmark
        returns = source.calculate_returns(benchmark)
        return returns

    def _create_broker(self):
        """
            broker : combine pipe_engine and generator
        """
        division_model = Division(self.underneath_module,
                                  self.trading_controls,
                                  self.sim_params.per_capital)
        blotter = SimulationBlotter(self.commission,
                                    self.slippage,
                                    self.execution_style)
        # generator --- compute capital or position to transactions
        generator = Generator(self.sim_params.delay,
                              blotter,
                              division_model)
        self.broker = Broker(self.pipeline_engine,
                             generator,
                             self.risk_allocation)

    def before_trading_start(self):
        """
            This is called once at the very begining of the backtest and should be used to set up
            any state needed by the algorithm.
        """
        try:
            #  Construct and store a PipelineEngine from loader.
            self.pipeline_engine = SimplePipelineEngine(self.pipelines,
                                                        self.final,
                                                        self.restrictions,
                                                        self.righted,
                                                        self.violated)

            self.ledger = Ledger(self.sim_params, self.risk_models, self.risk_fuse)
            self._create_broker()
        except Exception as e:
            raise ValueError('initialization error %s' % e)
        self._before_trading_start()

    def _create_metrics_tracker(self):
        """
            measure metrics of ledger
        """
        return MetricsTracker(
            sim_params=self.sim_params,
            benchmark_returns=self.benchmark_returns,
            metrics_sets=self._metrics_set
        )

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        return MinuteSimulationClock(self.sim_params)

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
        self.trading_client = AlgorithmSimulator(self, clock)
        return self.trading_client.transform()

    def yield_simulation(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_simulation()

    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with AlgoAPI(self):
            self._initialize(self, *args, **kwargs)

    # @staticmethod
    # def _create_daily_stats(perfs):
    #     # create daily and cumulative stats frame
    #     risk_perf = []
    #     daily_perf = []
    #     cumulative_perf = []
    #     for perf in perfs[:-1]:
    #         daily_perf.append(perf['daily_perf'])
    #         cumulative_perf.append(perf['cumulative_perf'])
    #         risk_perf.append(perf['cumulative_risk_metrics'])
    #     daily_dts = pd.DatetimeIndex(
    #         [p['period_close'] for p in daily_perf], tz='UTC'
    #     )
    #     # aggregate
    #     aggregate_stats = dict()
    #     risk_stats = pd.DataFrame(risk_perf, index=daily_dts)
    #     # print('risk_stats', risk_stats.T)
    #     aggregate_stats['cumulative_risk_metrics'] = risk_stats.T
    #     daily_stats = pd.DataFrame(daily_perf, index=daily_dts)
    #     aggregate_stats['daily_perf'] = daily_stats.T
    #     # print('daily_stats', daily_stats.T)
    #     cumulative_stats = pd.DataFrame(cumulative_perf, index=daily_dts)
    #     aggregate_stats['cumulative_risk_metrics'] = cumulative_stats.T
    #     # print('cumulative_stats', cumulative_stats.T)
    #     aggregate_stats['title'] = perfs[-1]
    #     # print('title', perfs[-1])
    #     print('stats', aggregate_stats)
    #     return aggregate_stats

    @staticmethod
    def _create_daily_stats(perfs):
        # create daily and cumulative stats frame
        risk_perf = []
        daily_perf = []
        cumulative_perf = []
        for perf in perfs[:-1]:
            daily_perf.append(perf['daily_perf'])
            cumulative_perf.append(perf['cumulative_perf'])
            risk_perf.append(perf['cumulative_risk_metrics'])
        daily_dts = pd.DatetimeIndex(
            [p['period_close'] for p in daily_perf], tz='UTC'
        )
        # aggregate
        aggregate_stats = dict()
        aggregate_stats['cumulative_risk_metrics'] = risk_perf
        aggregate_stats['daily_perf'] = daily_perf
        aggregate_stats['cumulative_risk_metrics'] = cumulative_perf
        aggregate_stats['title'] = perfs[-1]
        print('aggregate_stats', aggregate_stats)
        return aggregate_stats

    def run(self):
        """Run the algorithm.
        """
        self.before_trading_start()
        # Create px_trade and loop through simulated_trading.
        # Each iteration returns a perf dictionary
        perfs = []
        for perf in self.yield_simulation():
            print('perf', perf)
            perfs.append(perf)
        # convert perf dict to pandas frame
        analysis = self._create_daily_stats(perfs)
        # analysis = self.analyse(daily_stats)
        return analysis

    # def analyse(self, perf):
    #     with AlgoAPI(self):
    #         stats = self._analyze.end_of_simulation(
    #             perf,
    #             self.ledger,
    #             self.benchmark_returns)
    #     return stats

    @api_method
    def get_environment(self, field='platform'):
        """Query the execution environment.

        Parameters
        ----------
        field : {'platform', 'arena', 'data_frequency',
                 'start', 'end', 'capital_base', 'platform', '*'}
            The field to query. The options have the following meanings:
              arena : str
                  The arena from the ArkQuant parameters. This will normally
                  be ``'backtest'`` but some systems may use this distinguish
                  reality trading from backtesting.
              data_frequency : {'daily', 'minute'}
                  data_frequency tells the algorithm if it is running with
                  daily data or minute data.
              start : datetime
                  The start date for the ArkQuant.
              end : datetime
                  The end date for the ArkQuant.
              capital_base : float
                  The starting capital for the ArkQuant.
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

    def set_final_style(self, final_model):
        """
        :param final_model:  Final model used in pipeline to resolve the output of pipeline
        :return:
        """
        self.final = final_model

    @staticmethod
    def validate_pipeline(pipe_model):
        assert pipe_model is not None and isinstance(pipe_model, Pipeline), \
            'pipelines must validate and subclass of Pipeline'
        return pipe_model

    def attach_pipeline(self, pipeline_model):

        """Register a pipeline to be computed at the start of each day.

        Parameters
        ----------
        pipeline_model : Pipeline objects The pipeline to have computed.

        Returns
        -------
        pipeline : Pipeline
            Returns the pipeline that was attached unchanged.

        See Also
        --------
        :func:`zipline.api.pipeline_output`
        """
        # Return the pipeline to allow expressions like
        if self.initialized:
            raise AttributeError
        new_pipeline = self.validate_pipeline(pipeline_model)
        self.pipelines.append(new_pipeline)

    def create_pipeline(self, terms, ump_pickers=None):
        # for script_file in scripts:
        #     name = script_file.rsplit('.')[-2]
        #     with open(script_file, 'r') as f:
        #         exec(f.read(), self.namespace)
        #         pipelines.append(self.namespace[name])
        try:
            pipeline = Pipeline(terms, ump_pickers)
        except Exception as e:
            print('can not create pipeline due to %s' % e)
        else:
            self.attach_pipeline(pipeline)

    def set_engine_restriction(self,
                               righted=True,
                               violated=True):
        """
        :param righted: bool remove righted position
        :param violated: bool remove violated position
        """
        self.righted = righted
        self.violated = violated

    ####################
    # Finance Controls #
    ####################

    @api_method
    def set_slippage_style(self, slippage_intance):
        """
        Set the slippage models for the ArkQuant.

        Parameters
        ----------
        slippage_intance : EquitySlippageModel
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
        self.slippage = slippage_intance

    @api_method
    def set_commission_style(self, commission_instance):
        """Sets the commission models for the trading

        Parameters
        ----------
        commission_instance : CommissionModel instance
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
        self.commission = commission_instance

    def set_execution_style(self, execution_model):
        """Sets the execution_style models for the trading
        :param execution_model --- execution_style
        """
        if self.initialized:
            raise AttributeError
        self.execution_style = execution_model

    def set_restriction_style(self, restricted_models):
        """Set a restriction on which asset can be ordered.

        Parameters
        ----------
        restricted_models : list of Restrictions model
            An object providing information about restricted asset.

        See Also
        --------
        zipline.finance.restrictions.Restrictions
        """
        if self.initialized:
            raise SetCancelPolicyPostInit()
        self.restrictions = restricted_models

    ####################
    # Pb Controls #
    ####################

    def set_allocation_style(self, allocation_model):
        if self.initialized:
            raise InterruptedError
        self.risk_allocation = allocation_model

    def set_uncover_style(self, uncover_model):
        if self.initialized:
            raise InterruptedError
        self.underneath_module = uncover_model

    ####################
    # Risk Controls #
    ####################

    def set_alert_style(self, alert_model):
        """
        :param alert_model: risk module --- alert.py
        """
        assert isinstance(alert_model, Risk), 'must be subclass of Risk'
        self.risk_models = alert_model

    def set_fuse_style(self, fuse_model):
        """
            :param fuse_model: risk module --- fuse.py
        """
        assert isinstance(fuse_model, BaseFuse), 'must be subclass of fuse'
        self.risk_fuse = fuse_model

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
                           max_notional,
                           window=1):
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
        leverage = total_value / loan
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


# if __name__ == '__main__':
#
#     from trade.params import create_simulation_parameters
#     trading_params = create_simulation_parameters(start='2020-09-01', end='2020-09-20')
#     print('trading_params', trading_params)
#     # initialize trading algorithm
#     trading = TradingAlgorithm(trading_params)
#     # set finance models
#     slippage = FixedBasisPointSlippage()
#     commission = Commission()
#     order_style = LimitOrder(0.08)
#     restrictions_list = [StatusRestrictions(), DataBoundsRestrictions()]
#     trading.set_slippage_style(slippage)
#     # print('slippage', trading.slippage)
#     trading.set_commission_style(commission)
#     # print('commission', trading.commission)
#     trading.set_execution_style(order_style)
#     # print('execution', trading.execution)
#     trading.set_restriction_style(restrictions_list)
#     # set pipeline api scripts = None
#     # trading.set_pipeline_final()
#     from pipe.term import Term
#     kw = {'window': (5, 10), 'fields': ['close']}
#     cross_term = Term('cross', kw)
#     kw = {'fields': ['close'], 'window': 5, 'final': True}
#     break_term = Term('break', kw, cross_term)
#     pipeline = Pipeline([break_term, cross_term])
#     trading.attach_pipeline(pipeline)
#     # set pb models
#     # trading.set_uncover_style(SimpleUncover)
#     # set risk models
#     turtle = Turtle(5)
#     trading.set_allocation_style(turtle)
#     # print('allocation', trading.broker.capital_model)
#     trading.set_alert_style(PositionLossRisk(0.1))
#     trading.set_fuse_style(Fuse(0.85))
#     # set trading control models
#     trading.set_max_position_size(0.8)
#     trading.set_max_order_size(0.1, window=5)
#     trading.set_long_only()
#     # set account models
#     # trading.set_net_leverage(1.3)
#     # run algorithm
#     analysis = trading.run()
#     # analysis_path = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/metric/temp.json'
#     # with open(analysis_path, 'w+') as f:
#     #     json.dump(analysis, f)
