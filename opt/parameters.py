#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""

from collections import Mapping
from functools import reduce , partial
from itertools import product
import operator ,os ,sys,pandas as pd

# __func__ ---指向函数对象
DEFAULT_CAPITAL_BASE = 1e5


class SimulationParameters(object):
    def __init__(self,
                 start_session,
                 end_session,
                 trading_calendar,
                 capital_base=DEFAULT_CAPITAL_BASE,
                 # emission_rate='daily',
                 # arena='backtest',
                 data_frequency='daily'):

        assert type(start_session) == pd.Timestamp
        assert type(end_session) == pd.Timestamp

        assert trading_calendar is not None, \
            "Must pass in trading _calendar!"
        assert start_session <= end_session, \
            "Period start falls after period end."
        assert start_session <= trading_calendar.last_trading_session, \
            "Period start falls after the last known trading day."
        assert end_session >= trading_calendar.first_trading_session, \
            "Period end falls before the first known trading day."

        # chop off any minutes or hours on the given start and end dates,
        # as we only support session labels here (and we represent session
        # labels as midnight UTC).
        # self._start_session = normalize_date(start_session)
        # self._end_session = normalize_date(end_session)

        self._capital_base = capital_base

        # self._emission_rate = emission_rate
        self._data_frequency = data_frequency

        # copied to algorithm's environment for runtime access
        # self._arena = arena

        self._trading_calendar = trading_calendar

        if not trading_calendar.is_session(self._start_session):
            # if the start date is not a valid session in this _calendar,
            # push it forward to the first valid session
            self._start_session = trading_calendar.minute_to_session_label(
                self._start_session
            )

        if not trading_calendar.is_session(self._end_session):
            # if the end date is not a valid session in this _calendar,
            # pull it backward to the last valid session before the given
            # end date.
            self._end_session = trading_calendar.minute_to_session_label(
                self._end_session, direction="previous"
            )

        self._first_open = trading_calendar.open_and_close_for_session(
            self._start_session
        )[0]
        self._last_close = trading_calendar.open_and_close_for_session(
            self._end_session
        )[1]

    @property
    def capital_base(self):
        return self._capital_base

    # @property
    # def emission_rate(self):
    #     return self._emission_rate

    @property
    def data_frequency(self):
        return self._data_frequency

    @data_frequency.setter
    def data_frequency(self, val):
        self._data_frequency = val

    # @property
    # def arena(self):
    #     return self._arena
    #
    # @arena.setter
    # def arena(self, val):
    #     self._arena = val

    @property
    def start_session(self):
        return self._start_session

    @property
    def end_session(self):
        return self._end_session

    @property
    def first_open(self):
        return self._first_open

    @property
    def last_close(self):
        return self._last_close

    @property
    def trading_calendar(self):
        return self._trading_calendar

    @property
    @remember_last #remember_last = weak_lru_cache(1)
    def sessions(self):
        return self._trading_calendar.sessions_in_range(
            self.start_session,
            self.end_session
        )

    def create_new(self, start_session, end_session, data_frequency=None):
        if data_frequency is None:
            data_frequency = self.data_frequency

        return SimulationParameters(
            start_session,
            end_session,
            self._trading_calendar,
            capital_base=self.capital_base,
            emission_rate=self.emission_rate,
            data_frequency=data_frequency,
            arena=self.arena
        )

    def __repr__(self):
        return """
{class_name}(
    start_session={start_session},
    end_session={end_session},
    capital_base={capital_base},
    data_frequency={data_frequency},
    emission_rate={emission_rate},
    first_open={first_open},
    last_close={last_close},
    trading_calendar={trading_calendar}
)\
""".format(class_name=self.__class__.__name__,
           start_session=self.start_session,
           end_session=self.end_session,
           capital_base=self.capital_base,
           data_frequency=self.data_frequency,
           emission_rate=self.emission_rate,
           first_open=self.first_open,
           last_close=self.last_close,
           trading_calendar=self._trading_calendar)


class Trading(object):

    def __init__(self,
                 sim_params,
                 data_portal=None,
                 asset_finder=None,
                 # algorithm API
                 namespace=None,
                 script=None,
                 algo_filename=None,
                 initialize=None,
                 handle_data=None,
                 before_trading_start=None,
                 analyze=None,
                 #
                 trading_calendar=None,
                 metrics_set=None,
                 blotter=None,
                 blotter_class=None,
                 cancel_policy=None,
                 benchmark_sid=None,
                 benchmark_returns=None,
                 platform='zipline',
                 capital_changes=None,
                 get_pipeline_loader=None,
                 create_event_context=None,
                 **initialize_kwargs):

        # List of trading controls to be used to validate orders.
        self.trading_controls = []

        # List of account controls to be checked on each bar.
        self.account_controls = []

        self.restrictions = NoRestrictions()

        self._recorded_vars = {}

        self._platform = platform

        self.data_portal = data_portal

        # Prepare the algo for initialization
        self.initialized = False
        self.benchmark_sid = benchmark_sid

        # if self.data_portal is None:
        #     if asset_finder is None:
        #         raise ValueError(
        #             "Must pass either data_portal or asset_finder "
        #             "to TradingAlgorithm()"
        #         )
        #     self.asset_finder = asset_finder
        # else:
        #     # Raise an error if we were passed two different asset finders.
        #     # There's no world where that's a good idea.
        #     if asset_finder is not None \
        #        and asset_finder is not data_portal.asset_finder:
        #         raise ValueError(
        #             "Inconsistent asset_finders in TradingAlgorithm()"
        #         )
        #     self.asset_finder = data_portal.asset_finder


        self.sim_params = sim_params

        # if trading_calendar is None:
        #     self.trading_calendar = sim_params.trading_calendar
        # elif trading_calendar.name == sim_params.trading_calendar.name:
        #     self.trading_calendar = sim_params.trading_calendar
        # else:
        #     raise ValueError(
        #         "Conflicting trading-calendars: trading_calendar={}, but "
        #         "sim_params.trading_calendar={}".format(
        #             trading_calendar.name,
        #             self.sim_params.trading_calendar.name,
        #         )
        #     )

        self.benchmark_returns = benchmark_returns

        if blotter is not None:
            self.blotter = blotter
        else:
            # or的运算原理：or是从左到右计算表达式，返回第一个为真的值
            cancel_policy = cancel_policy or NeverCancel()
            blotter_class = blotter_class or SimulationBlotter
            self.blotter = blotter_class(cancel_policy=cancel_policy)

        # The symbol lookup date specifies the date to use when resolving
        # symbols to sids, and can be set using set_symbol_lookup_date()
        self._symbol_lookup_date = None

        self.event_manager = EventManager(create_event_context)

        # If string is passed in, execute and get reference to
        # functions.
        self.algoscript = script

        self._initialize = None
        self._before_trading_start = None
        self._handle_data = None
        self._analyze = None

        self._in_before_trading_start = False

        self.namespace = namespace or {}

        if self.algoscript is not None:
            unexpected_api_methods = set()
            if initialize is not None:
                unexpected_api_methods.add('initialize')
            if handle_data is not None:
                unexpected_api_methods.add('handle_data')
            if before_trading_start is not None:
                unexpected_api_methods.add('before_trading_start')
            if analyze is not None:
                unexpected_api_methods.add('analyze')

            if unexpected_api_methods:
                raise ValueError(
                    "TradingAlgorithm received a script and the following API"
                    " methods as functions:\n{funcs}".format(
                        funcs=unexpected_api_methods,
                    )
                )

            if algo_filename is None:
                algo_filename = '<string>'

            #exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object
            # if filename is None ,'<string>' is used
            code = compile(self.algoscript, algo_filename, 'exec')
            #动态执行文件， 相当于import
            exec(code, self.namespace)

            def noop(*args, **kwargs):
                pass

            #dict get参数可以为方法或者默认参数
            self._initialize = self.namespace.get('initialize', noop)
            self._handle_data = self.namespace.get('handle_data', noop)
            self._before_trading_start = self.namespace.get(
                'before_trading_start',
            )
            # Optional analyze function, gets called after run
            self._analyze = self.namespace.get('analyze')

        else:
            self._initialize = initialize or (lambda self: None)
            self._handle_data = handle_data
            self._before_trading_start = before_trading_start
            self._analyze = analyze

        self.event_manager.add_event(
            zipline.utils.events.Event(
                zipline.utils.events.Always(),
                # We pass handle_data.__func__ to get the unbound method.
                # We will explicitly pass the algorithm to bind it again.
                self.handle_data.__func__,
            ),
            prepend=True,
        )

        self._last_sync_time = pd.NaT
        self.metrics_tracker = None
        self._metrics_set = metrics_set
        if self._metrics_set is None:
            self._metrics_set = load_metrics_set('default')

        if self.sim_params.capital_base <= 0:
            raise ZeroCapitalError()

        self.initialize_kwargs = initialize_kwargs or {}

        # A dictionary of capital changes, keyed by timestamp, indicating the
        # target/delta of the capital changes, along with values
        self.capital_changes = capital_changes or {}

        # A dictionary of the actual capital change deltas, keyed by timestamp
        self.capital_change_deltas = {}

        self._backwards_compat_universe = None

        # Initialize pipe API data.
        self.init_engine(get_pipeline_loader)
        self._pipelines = {}

        # Create an already-expired cache so that we compute the first time
        # data is requested.
        self._pipeline_cache = ExpiringCache(
            cleanup=clear_dataframe_indexer_caches
        )

    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self, *args, **kwargs)

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


    def compute_eager_pipelines(self):
        """
        Compute any pipelines attached with eager=True.
        """
        for name, pipe in self._pipelines.items():
            if pipe.eager:
                self.pipeline_output(name)

    def before_trading_start(self,dt):

        self.compute_eager_pipelines()

        assets_we_care = self.metrics_tracker.position.assets
        splits = self.data_portal.get_splits(assets_we_care, dt)
        self.metrics_tracker.process_splits(splits)

        self.compute_eager_pipelines()

        self._in_before_trading_start = True

    def handle_data(self, data):
        if self._handle_data:
            self._handle_data(self, data)

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


class AlgorithmSimulation(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }


    def __init__(self, algo, sim_params, data_portal, clock, benchmark_source,
                 restrictions, universe_func):

        # ==============
        # Simulation
        # Param Setup
        # ==============
        self.sim_params = sim_params
        self.data_portal = data_portal
        self.restrictions = restrictions

        # ==============
        # Algo Setup
        # ==============
        self.algo = algo

        # ==============
        # Snapshot Setup
        # ==============

        # This object is the way that user algorithms interact with OHLCV data,
        # fetcher data, and some API methods like `data.can_trade`.
        self.current_data = self._create_bar_data(universe_func)

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None

        self.clock = clock

        self.benchmark_source = benchmark_source

        # =============
        # Logging Setup
        # =============

        # Processor function for injecting the algo_dt into
        # user prints/logs.
        def inject_algo_dt(record):
            if 'algo_dt' not in record.extra:
                record.extra['algo_dt'] = self.simulation_dt

    def get_simulation_dt(self):
        return self.simulation_dt

    #获取交易日数据，封装为一个API(fetch process flush other api)
    def _create_bar_data(self, universe_func):
        return BarData(
            data_portal=self.data_portal,
            simulation_dt_func=self.get_simulation_dt,
            data_frequency=self.sim_params.data_frequency,
            trading_calendar=self.algo.trading_calendar,
            restrictions=self.restrictions,
            universe_func=get_splits_divdend
        )

    def transfrom(self,dt):
        """
        Main generator work loop.
        """
        algo = self.algo
        metrics_tracker = algo.metrics_tracker
        emission_rate = metrics_tracker.emission_rate
        engine = algo.engine
        handle_data = algo.event_manager.handle_data

        metrics_tracker.handle_market_open(dt, algo.data_portal)

        def process_txn_commission(transactions,commissions):
            for txn in transactions:
                metrics_tracker.process_transaction(txn)

            for commission in commissions:
                metrics_tracker.process_commission(commission)

        @contextlib.contextmanager
        def once_a_day(dt):
            payout = engine.get_payout(dt,metrics_tracker)
            try:
                yield payout
            finally:
                layout = engine.get_layout(dt,metrics_tracker)
                process_txn_commission(*layout)

        def on_exit():
            # Remove references to algo, data portal, et al to break cycles
            # and ensure deterministic cleanup of these objects when the
            # simulation finishes.
            self.algo = None
            self.benchmark_source = self.data_portal = None

        with ExitStack() as stack:
            """
            由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
            这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
            enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
            callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
            """
            stack.callback(on_exit())
            stack.enter_context(ZiplineAPI(self.algo))

            for dt in algo.trading_calendar:

                algo.on_dt_changed(dt)
                algo.before_trading_start(self.current_data(dt))
                with once_a_day(dt) as  action:
                    process_txn_commission(*action)
                yield self._get_daily_message(dt, algo, metrics_tracker)

            risk_message = metrics_tracker.handle_simulation_end(
                self.data_portal,
            )
            yield risk_message

    def _get_daily_message(self, dt, algo, metrics_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = metrics_tracker.handle_market_close(
            dt,
            self.data_portal,
        )
        perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
        return perf_message



def run_algorithm(start,
                  end,
                  initialize,
                  capital_base,
                  handle_data=None,
                  before_trading_start=None,
                  analyze=None,
                  data_frequency='daily',
                  bundle='quantopian-quandl',
                  bundle_timestamp=None,
                  trading_calendar=None,
                  metrics_set='default',
                  benchmark_returns=None,
                  default_extension=True,
                  extensions=(),
                  strict_extensions=True,
                  environ=os.environ,
                  blotter='default'):
    """
    Run a trading algorithm.

    Parameters
    ----------
    start : datetime
        The start date of the backtest.
    end : datetime
        The end date of the backtest..
    initialize : callable[context -> None]
        The initialize function to use for the algorithm. This is called once
        at the very begining of the backtest and should be used to set up
        any state needed by the algorithm.
    capital_base : float
        The starting capital for the backtest.
    handle_data : callable[(context, BarData) -> None], optional
        The handle_data function to use for the algorithm. This is called
        every minute when ``data_frequency == 'minute'`` or every day
        when ``data_frequency == 'daily'``.
    before_trading_start : callable[(context, BarData) -> None], optional
        The before_trading_start function for the algorithm. This is called
        once before each trading day (after initialize on the first day).
    analyze : callable[(context, pd.DataFrame) -> None], optional
        The analyze function to use for the algorithm. This function is called
        once at the end of the backtest and is passed the context and the
        performance data.
    data_frequency : {'daily', 'minute'}, optional
        The data frequency to run the algorithm at.
    bundle : str, optional
        The name of the data bundle to use to load the data to run the backtest
        with. This defaults to 'quantopian-quandl'.
    bundle_timestamp : datetime, optional
        The datetime to lookup the bundle data for. This defaults to the
        current time.
    trading_calendar : TradingCalendar, optional
        The trading _calendar to use for your backtest.
    metrics_set : iterable[Metric] or str, optional
        The set of metrics to compute in the simulation. If a string is passed,
        resolve the set with :func:`zipline.finance.metrics.load`.
    default_extension : bool, optional
        Should the default zipline extension be loaded. This is found at
        ``$ZIPLINE_ROOT/extension.py``
    extensions : iterable[str], optional
        The names of any other extensions to load. Each element may either be
        a dotted module path like ``a.b.c`` or a path to a python file ending
        in ``.py`` like ``a/b/c.py``.
    strict_extensions : bool, optional
        Should the run fail if any extensions fail to load. If this is false,
        a warning will be raised instead.
    environ : mapping[str -> str], optional
        The os environment to use. Many extensions use this to get parameters.
        This defaults to ``os.environ``.
    blotter : str or zipline.finance.oms.Blotter, optional
        Blotter to use with this algorithm. If passed as a string, we look for
        a oms construction function registered with
        ``zipline.extensions.register`` and call it with no parameters.
        Default is a :class:`zipline.finance.oms.SimulationBlotter` that
        never cancels orders.

    Returns
    -------
    perf : pd.DataFrame
        The daily performance of the algorithm.

    See Also
    --------
    zipline.data.bundles.bundles : The available data bundles.
    """
    load_extensions(default_extension, extensions, strict_extensions, environ)

    return _run(
        handle_data=handle_data,
        initialize=initialize,
        before_trading_start=before_trading_start,
        analyze=analyze,
        algofile=None,
        algotext=None,
        defines=(),
        data_frequency=data_frequency,
        capital_base=capital_base,
        bundle=bundle,
        bundle_timestamp=bundle_timestamp,
        start=start,
        end=end,
        output=os.devnull,
        trading_calendar=trading_calendar,
        print_algo=False,
        metrics_set=metrics_set,
        local_namespace=False,
        environ=environ,
        blotter=blotter,
        benchmark_returns=benchmark_returns,
    )


def _run(handle_data,
         initialize,
         before_trading_start,
         analyze,
         algofile,
         algotext,
         defines,
         data_frequency,
         capital_base,
         bundle,
         bundle_timestamp,
         start,
         end,
         output,
         trading_calendar,
         print_algo,
         metrics_set,
         local_namespace,
         environ,
         blotter,
         benchmark_returns):
    """Run a backtest for the given algorithm.

    This is shared between the cli and :func:`zipline.run_algo`.
    """
    if benchmark_returns is None:
        benchmark_returns, _ = load_market_data(environ=environ)

    if algotext is not None:
        if local_namespace:
            ip = get_ipython()  # noqa
            namespace = ip.user_ns
        else:
            namespace = {}

        for assign in defines:
            try:
                name, value = assign.split('=', 2)
            except ValueError:
                raise ValueError(
                    'invalid define %r, should be of the form name=value' %
                    assign,
                )
            try:
                # evaluate in the same namespace so names may refer to
                # eachother
                namespace[name] = eval(value, namespace)
            except Exception as e:
                raise ValueError(
                    'failed to execute definition for name %r: %s' % (name, e),
                )
    elif defines:
        raise _RunAlgoError(
            'cannot pass define without `algotext`',
            "cannot pass '-D' / '--define' without '-t' / '--algotext'",
        )
    else:
        namespace = {}
        if algofile is not None:
            algotext = algofile.read()

    if print_algo:
        if PYGMENTS:
            highlight(
                algotext,
                PythonLexer(),
                TerminalFormatter(),
                outfile=sys.stdout,
            )
        else:
            click.echo(algotext)

    if trading_calendar is None:
        trading_calendar = get_calendar('XNYS')

    # date parameter validation
    if trading_calendar.session_distance(start, end) < 1:
        raise _RunAlgoError(
            'There are no trading days between %s and %s' % (
                start.date(),
                end.date(),
            ),
        )

    bundle_data = bundles.load(
        bundle,
        environ,
        bundle_timestamp,
    )

    first_trading_day = \
        bundle_data.equity_minute_bar_reader.first_trading_day

    data = DataPortal(
        bundle_data.asset_finder,
        trading_calendar=trading_calendar,
        first_trading_day=first_trading_day,
        equity_minute_reader=bundle_data.equity_minute_bar_reader,
        equity_daily_reader=bundle_data.equity_daily_bar_reader,
        adjustment_reader=bundle_data.adjustment_reader,
    )

    pipeline_loader = USEquityPricingLoader(
        bundle_data.equity_daily_bar_reader,
        bundle_data.adjustment_reader,
    )

    def choose_loader(column):
        if column in USEquityPricing.columns:
            return pipeline_loader
        raise ValueError(
            "No pipe registered for column %s." % column
        )

    if isinstance(metrics_set, six.string_types):
        try:
            metrics_set = metrics.load(metrics_set)
        except ValueError as e:
            raise _RunAlgoError(str(e))

    if isinstance(blotter, six.string_types):
        try:
            blotter = load(Blotter, blotter)
        except ValueError as e:
            raise _RunAlgoError(str(e))

    perf = TradingAlgorithm(
        namespace=namespace,
        data_portal=data,
        get_pipeline_loader=choose_loader,
        trading_calendar=trading_calendar,
        sim_params=SimulationParameters(
            start_session=start,
            end_session=end,
            trading_calendar=trading_calendar,
            capital_base=capital_base,
            data_frequency=data_frequency,
        ),
        metrics_set=metrics_set,
        blotter=blotter,
        benchmark_returns=benchmark_returns,
        **{
            'initialize': initialize,
            'handle_data': handle_data,
            'before_trading_start': before_trading_start,
            'analyze': analyze,
        } if algotext is None else {
            'algo_filename': getattr(algofile, 'name', '<algorithm>'),
            'script': algotext,
        }
    ).run()

    if output == '-':
        click.echo(str(perf))
    elif output != os.devnull:  # make the zipline magic not write any data
        perf.to_pickle(output)

    return perf



def create_simulation_parameters(year=2006,
                                 start=None,
                                 end=None,
                                 capital_base=float("1.0e5"),
                                 num_days=None,
                                 data_frequency='daily',
                                 emission_rate='daily',
                                 trading_calendar=None):

    if not trading_calendar:
        trading_calendar = get_calendar("NYSE")

    if start is None:
        start = pd.Timestamp("{0}-01-01".format(year), tz='UTC')
    elif type(start) == datetime:
        start = pd.Timestamp(start)

    if end is None:
        if num_days:
            start_index = trading_calendar.all_sessions.searchsorted(start)
            end = trading_calendar.all_sessions[start_index + num_days - 1]
        else:
            end = pd.Timestamp("{0}-12-31".format(year), tz='UTC')
    elif type(end) == datetime:
        end = pd.Timestamp(end)

    sim_params = SimulationParameters(
        start_session=start,
        end_session=end,
        capital_base=capital_base,
        data_frequency=data_frequency,
        emission_rate=emission_rate,
        trading_calendar=trading_calendar,
    )

    return sim_params


class ParameterGrid(object):
    '''
      scipy.optimize.min(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None,
                         constraints=(), tol=None, callback=None, options=None)
      method: str or callable, optional, Nelder - Mead, (see here)
      Powell,, CG, BFGS, Newton - CG, L - BFGS - B, TNC, COBYLA, SLSQP, dogleg, trust - ncg,
               options: dict, optional
      maxiter: int.Maximum number of iterations to perform. disp: bool Constraints definition(only for COBYLA and SLSQP)
      type: eq for equality, ineq for inequality.fun: callable.jac: optional(only for SLSQP)
      args: sequence, optional
    '''
    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        """迭代参数组合实现"""
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """参数组合长度实现"""
        product_mul = partial(reduce, operator.mul)
        return sum(product_mul(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)


class TradingAlgorithm(object):
    """
        分步建仓 --- 重复行为（因为对应的标的不变）
        扩张增加特征收集模块

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
        platform : str, optional
            The platform the simulation is running on. This can be queried for
            in the simulation with ``get_environment``. This allows algorithms
            to conditionally execute code based on platform it is running on.
            default: 'zipline'
        adjustment_reader : AdjustmentReader
            The interface to the adjustments.

        history_container_class : type, optional
            The type of history container to use. default: HistoryContainer
    """
    def __init__(self,
                 sim_params,
                 data_portal=None,
                 asset_finder=None,
                 # algorithm API
                 namespace=None,
                 script=None,
                 algo_filename=None,
                 initialize=None,
                 handle_data=None,
                 before_trading_start=None,
                 analyze=None,
                 #
                 trading_calendar=None,
                 metrics_set=None,
                 blotter=None,
                 blotter_class=None,
                 cancel_policy=None,
                 benchmark_sid=None,
                 benchmark_returns=None,
                 platform='zipline',
                 capital_changes=None,
                 get_pipeline_loader=None,
                 create_event_context=None,
                 **initialize_kwargs):

        # List of trading controls to be used to validate orders.
        self.trading_controls = []

        # List of account controls to be checked on each bar.
        self.account_controls = []

        self._recorded_vars = {}
        self.namespace = namespace or {}

        self._platform = platform
        self.logger = None

        self.data_portal = data_portal

        if self.data_portal is None:
            if asset_finder is None:
                raise ValueError(
                    "Must pass either data_portal or asset_finder "
                    "to TradingAlgorithm()"
                )
            self.asset_finder = asset_finder
        else:
            # Raise an error if we were passed two different asset finders.
            # There's no world where that's a good idea.
            if asset_finder is not None \
               and asset_finder is not data_portal.asset_finder:
                raise ValueError(
                    "Inconsistent asset_finders in TradingAlgorithm()"
                )
            self.asset_finder = data_portal.asset_finder


        self.sim_params = sim_params
        if trading_calendar is None:
            self.trading_calendar = sim_params.trading_calendar
        elif trading_calendar.name == sim_params.trading_calendar.name:
            self.trading_calendar = sim_params.trading_calendar
        else:
            raise ValueError(
                "Conflicting trading-calendars: trading_calendar={}, but "
                "sim_params.trading_calendar={}".format(
                    trading_calendar.name,
                    self.sim_params.trading_calendar.name,
                )
            )

        self.benchmark_returns = benchmark_returns

        self._last_sync_time = pd.NaT
        self.metrics_tracker = None
        self._metrics_set = metrics_set
        if self._metrics_set is None:
            self._metrics_set = load_metrics_set('default')

        if blotter is not None:
            self.blotter = blotter
        else:
            cancel_policy = cancel_policy or NeverCancel()
            blotter_class = blotter_class or SimulationBlotter
            self.blotter = blotter_class(cancel_policy=cancel_policy)

        # The symbol lookup date specifies the date to use when resolving
        # symbols to sids, and can be set using set_symbol_lookup_date()
        self._symbol_lookup_date = None

        self.event_manager = EventManager(create_event_context)

        # If string is passed in, execute and get reference to
        # functions.
        self.algoscript = script

        self._handle_data = None

        if self.algoscript is not None:
            unexpected_api_methods = set()
            if initialize is not None:
                unexpected_api_methods.add('initialize')
            if handle_data is not None:
                unexpected_api_methods.add('handle_data')
            if before_trading_start is not None:
                unexpected_api_methods.add('before_trading_start')
            if analyze is not None:
                unexpected_api_methods.add('analyze')

            if unexpected_api_methods:
                raise ValueError(
                    "TradingAlgorithm received a script and the following API"
                    " methods as functions:\n{funcs}".format(
                        funcs=unexpected_api_methods,
                    )
                )

            if algo_filename is None:
                algo_filename = '<string>'

            #exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
            code = compile(self.algoscript, algo_filename, 'exec')
            #动态执行文件， 相当于import
            exec_(code, self.namespace)

            def noop(*args, **kwargs):
                pass

            #dict get参数可以为方法或者默认参数
            self._initialize = self.namespace.get('initialize', noop)
            self._handle_data = self.namespace.get('handle_data', noop)
            self._before_trading_start = self.namespace.get(
                'before_trading_start',
            )
            # Optional analyze function, gets called after run
            self._analyze = self.namespace.get('analyze')

        else:
            self._initialize = initialize or (lambda self: None)
            self._handle_data = handle_data
            self._before_trading_start = before_trading_start
            self._analyze = analyze

        self.event_manager.add_event(
            zipline.utils.events.Event(
                zipline.utils.events.Always(),
                # We pass handle_data.__func__ to get the unbound method.
                # We will explicitly pass the algorithm to bind it again.
                self.handle_data.__func__,
            ),
            prepend=True,
        )

        if self.sim_params.capital_base <= 0:
            raise ZeroCapitalError()

        # Prepare the algo for initialization
        self.initialized = False

        self.initialize_kwargs = initialize_kwargs or {}

        self.benchmark_sid = benchmark_sid

        # A dictionary of capital changes, keyed by timestamp, indicating the
        # target/delta of the capital changes, along with values
        self.capital_changes = capital_changes or {}

        # A dictionary of the actual capital change deltas, keyed by timestamp
        self.capital_change_deltas = {}

        self.restrictions = NoRestrictions()

        self._backwards_compat_universe = None

        # Initialize pipe API data.
        self.init_engine(get_pipeline_loader)
        self._pipelines = {}

        # Create an already-expired cache so that we compute the first time
        # data is requested.
        self._pipeline_cache = ExpiringCache(
            cleanup=clear_dataframe_indexer_caches
        )

        self._initialize = None
        self._before_trading_start = None
        self._analyze = None

        self._in_before_trading_start = False


    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self, *args, **kwargs)

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


    def compute_eager_pipelines(self):
        """
        Compute any pipelines attached with eager=True.
        """
        for name, pipe in self._pipelines.items():
            if pipe.eager:
                self.pipeline_output(name)

    def before_trading_start(self,dt):

        self.compute_eager_pipelines()

        assets_we_care = self.metrics_tracker.position.assets
        splits = self.data_portal.get_splits(assets_we_care, dt)
        self.metrics_tracker.process_splits(splits)

        self.compute_eager_pipelines()

        self._in_before_trading_start = True

    def handle_data(self, data):
        if self._handle_data:
            self._handle_data(self, data)

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
        # This is where a scheduled function's rule is associated with a _calendar.
        opd.cal = cal
        return opd

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
        for name, value in chain(positionals, iteritems(kwargs)):
            self._recorded_vars[name] = value

    @api_method
    def set_benchmark(self, benchmark):
        """Set the benchmark asset.

        Parameters
        ----------
        benchmark : zipline.assets.Asset
            The asset to set as the new benchmark.

        Notes
        -----
        Any dividends payed out for that new benchmark asset will be
        automatically reinvested.
        """
        if self.initialized:
            raise SetBenchmarkOutsideInitialize()

        self.benchmark_sid = benchmark

    @api_method
    @preprocess(
        symbol_str=ensure_upper_case,
        country_code=optionally(ensure_upper_case),
    )
    def symbol(self, symbol_str, country_code=None):
        """Lookup an Equity by its ticker symbol.

        Parameters
        ----------
        symbol_str : str
            The ticker symbol for the equity to lookup.
        country_code : str or None, optional
            A country to limit symbol searches to.

        Returns
        -------
        equity : zipline.assets.Equity
            The equity that held the ticker symbol on the current
            symbol lookup date.

        Raises
        ------
        SymbolNotFound
            Raised when the symbols was not held on the current lookup date.

        See Also
        --------
        :func:`zipline.api.set_symbol_lookup_date`
        """
        # If the user has not set the symbol lookup date,
        # use the end_session as the date for symbol->sid resolution.
        # self.asset_finder.retrieve_asset(sid)
        _lookup_date = self._symbol_lookup_date \
            if self._symbol_lookup_date is not None \
            else self.sim_params.end_session

        return self.asset_finder.lookup_symbol(
            symbol_str,
            as_of_date=_lookup_date,
            country_code=country_code,
        )

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
    def set_slippage(self, us_equities=None, us_futures=None):
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

        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleSlippageModel(
                    asset_type='futures',
                    given_model=us_futures,
                    supported_asset_types=us_futures.allowed_asset_types,
                )
            self.blotter.slippage_models[Future] = us_futures

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

        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleCommissionModel(
                    asset_type='futures',
                    given_model=us_futures,
                    supported_asset_types=us_futures.allowed_asset_types,
                )
            self.blotter.commission_models[Future] = us_futures

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

    @api_method
    def set_symbol_lookup_date(self, dt):
        """Set the date for which symbols will be resolved to their asset
        (symbols may map to different firms or underlying asset at
        different times)

        Parameters
        ----------
        dt : datetime
            The new symbol lookup date.
        """
        try:
            self._symbol_lookup_date = pd.Timestamp(dt, tz='UTC')
        except ValueError:
            raise UnsupportedDatetimeFormat(input=dt,
                                            method='set_symbol_lookup_date')

    # Remain backwards compatibility
    @property
    def data_frequency(self):
        return self.sim_params.data_frequency

    @data_frequency.setter
    def data_frequency(self, value):
        assert value in ('daily', 'minute')
        self.sim_params.data_frequency = value

    @api_method
    @require_initialized(HistoryInInitialize())
    def history(self, bar_count, frequency, field, ffill=True):
        """DEPRECATED: use ``data.history`` instead.
        """
        warnings.warn(
            "The `history` method is deprecated.  Use `data.history` instead.",
            category=ZiplineDeprecationWarning,
            stacklevel=4
        )

        return self.get_history_window(
            bar_count,
            frequency,
            self._calculate_universe(),
            field,
            ffill
        )

    def get_history_window(self, bar_count, frequency, assets, field, ffill):
        if not self._in_before_trading_start:
            return self.data_portal.get_history_window(
                assets,
                self.datetime,
                bar_count,
                frequency,
                field,
                self.data_frequency,
                ffill,
            )
        else:
            # If we are in before_trading_start, we need to get the window
            # as of the previous market minute
            adjusted_dt = \
                self.trading_calendar.previous_minute(
                    self.datetime
                )

            window = self.data_portal.get_history_window(
                assets,
                adjusted_dt,
                bar_count,
                frequency,
                field,
                self.data_frequency,
                ffill,
            )

            # Get the adjustments between the last market minute and the
            # current before_trading_start dt and apply to the window
            adjs = self.data_portal.get_adjustments(
                assets,
                field,
                adjusted_dt,
                self.datetime
            )
            window = window * adjs

            return window

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

    def validate_account_controls(self):
        for control in self.account_controls:
            control.validate(self.portfolio,
                             self.account,
                             self.get_datetime(),
                             self.trading_client.current_data)

    @api_method
    def set_max_leverage(self, max_leverage):
        """Set a limit on the maximum leverage of the algorithm.

        Parameters
        ----------
        max_leverage : float
            The maximum leverage for the algorithm. If not provided there will
            be no maximum.
        """
        control = MaxLeverage(max_leverage)
        self.register_account_control(control)

    @api_method
    def set_min_leverage(self, min_leverage, grace_period):
        """Set a limit on the minimum leverage of the algorithm.

        Parameters
        ----------
        min_leverage : float
            The minimum leverage for the algorithm.
        grace_period : pd.Timedelta
            The offset from the start date used to enforce a minimum leverage.
        """
        deadline = self.sim_params.start_session + grace_period
        control = MinLeverage(min_leverage, deadline)
        self.register_account_control(control)

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
    def set_max_order_count(self, max_count, on_error='fail'):
        """Set a limit on the number of orders that can be placed in a single
        day.

        Parameters
        ----------
        max_count : int
            The maximum number of orders that can be placed on any single day.
        """
        control = MaxOrderCount(on_error, max_count)
        self.register_trading_control(control)

    @api_method
    def set_do_not_order_list(self, restricted_list, on_error='fail'):
        """Set a restriction on which asset can be ordered.

        Parameters
        ----------
        restricted_list : container[Asset], SecurityList
            The asset that cannot be ordered.
        """
        if isinstance(restricted_list, SecurityList):
            warnings.warn(
                "`set_do_not_order_list(security_lists.leveraged_etf_list)` "
                "is deprecated. Use `set_asset_restrictions("
                "security_lists.restrict_leveraged_etfs)` instead.",
                category=ZiplineDeprecationWarning,
                stacklevel=2
            )
            restrictions = SecurityListRestrictions(restricted_list)
        else:
            warnings.warn(
                "`set_do_not_order_list(container_of_assets)` is deprecated. "
                "Create a zipline.finance.asset_restrictions."
                "StaticRestrictions object with a container of asset and use "
                "`set_asset_restrictions(StaticRestrictions("
                "container_of_assets))` instead.",
                category=ZiplineDeprecationWarning,
                stacklevel=2
            )
            restrictions = StaticRestrictions(restricted_list)

        self.set_asset_restrictions(restrictions, on_error)

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


