#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import os
from .algorithm import TradingAlgorithm
from trade.params import create_simulation_parameters


def run_algorithm(start,
                  end,
                  delay,
                  capital_base,
                  loan_base,
                  per_capital,
                  data_frequency,
                  benchmark,
                  initialize,
                  handle_data=None,
                  before_trading_start=None,
                  analyze=None,
                  trading_calendar=None,
                  metrics_set='default',
                  benchmark_returns=None,
                  default_extension=True,
                  extensions=(),
                  strict_extensions=True,
                  environ=os.environ,
                  blotter='default'):
    """
    Run a backtest for the given algorithm.This is shared between the cli and :func:`zipline.run_algo`.

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
        The set of metric to compute in the ArkQuant. If a string is passed,
        resolve the set with :func:`zipline.finance.metric.load`.
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
    blotter : str or zipline.finance.broker.Blotter, optional
        Blotter to use with this algorithm. If passed as a string, we look for
        a broker construction function registered with
        ``zipline.extensions.register`` and call it with no parameters.
        Default is a :class:`zipline.finance.broker.SimulationBlotter` that
        never cancels orders.

    Returns
    -------
    perf : pd.DataFrame
        The daily performance of the algorithm.

    See Also
    --------
    zipline.data.bundles.bundles : The available data bundles.
    """
    # load_extensions(default_extension, extensions, strict_extensions, environ)
    sim_params = create_simulation_parameters(
                                            start=start,
                                            end=end,
                                            delay=delay,
                                            capital_base=capital_base,
                                            loan_base=loan_base,
                                            per_capital=per_capital,
                                            data_frequency=data_frequency,
                                            benchmark=benchmark)

    # set module which needed
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

    def choose_loader(column):
        if column in USEquityPricing.columns:
            return pipeline_loader
        raise ValueError(
            "No pipe registered for column %s." % column
        )

    if isinstance(metrics_set, six.string_types):
        try:
            metrics_set = metric.load(metrics_set)
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
