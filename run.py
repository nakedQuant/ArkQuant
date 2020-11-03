#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import pandas as pd, os
# finance module
from finance.slippage import FixedBasisPointSlippage
from finance.execution import LimitOrder
from finance.commission import Commission
from finance.restrictions import StatusRestrictions, DataBoundsRestrictions
from finance.control import NetLeverage
# pipe engine
from pipe.pipeline import Pipeline
# risk management
from risk.allocation import Turtle
from risk.alert import PositionLossRisk
from risk.fuse import Fuse
# metric module
from algorithm import TradingAlgorithm
from trade.params import create_simulation_parameters


def run_algorithm(start=None,
                  end=None,
                  delay=None,
                  loan_base=None,
                  per_capital=None,
                  capital_base=None,
                  benchmark=None,
                  data_frequency='daily',
                  slippage=FixedBasisPointSlippage(),
                  commission=Commission(),
                  execution=LimitOrder(0.08),
                  position_control_threshold=0.8,
                  order_control_threshold=0.05,
                  allocation_policy=Turtle(5),
                  restricted_rules=[StatusRestrictions(), DataBoundsRestrictions()],
                  risk_alert_policy=PositionLossRisk(0.1),
                  risk_fuse_policy=Fuse(0.85),
                  metrics_set=None
                  ):
    """
        Run a backtest for the given algorithm

    Parameters
    ----------
    start : datetime
        The start date of the backtest.
    end : datetime
        The end date of the backtest.
    delay : int
        delay used in pb to simulate dual transaction
    capital_base : float
        the initial capital
    per_capital : float
        the min_capital to carry call action
    loan_base : float
        use to calculate leverage
    data_frequency : {'daily', 'minute'}, optional
        The data frequency to run the algorithm at.
    benchmark : string
        e.g. 000001
    slippage : finance Slipppage model
        used to simulate order price into transaction price
    commission : finance Commission
        used to calculate cost of order into transaction
    execution : finance Execution
        used to specify order type
    position_control_threshold : float
        used to control the solo positon proportion
    order_control_threshold :
        used to control the order amount
    allocation_policy : func to distribute capital
        e.g. equal , delta
    restricted_rules : finance restrictions
        used to filter assets from market before run pipeline engine
    risk_alert_policy : risk  alert
        used to control position by measure the value of position or returns
    risk_fuse_policy : risk fuse
        used to handle portfolio when portfolio value is less than threshold
    metrics_set : iterable[Metric] or str, optional
        The set of metric to compute in the ArkQuant. If a string is passed,
    # default_extension : bool, optional
    #     Should the default zipline extension be loaded. This is found at
    #     ``$ZIPLINE_ROOT/extension.py``
    # extensions : iterable[str], optional
    #     The names of any other extensions to load. Each element may either be
    #     a dotted module path like ``a.b.c`` or a path to a python file ending
    #     in ``.py`` like ``a/b/c.py``.
    # strict_extensions : bool, optional
    #     Should the run fail if any extensions fail to load. If this is false,
    #     a warning will be raised instead.
    # environ : mapping[str -> str], optional
    #     The os environment to use. Many extensions use this to get parameters.
    #     This defaults to ``os.environ``.

    Returns
    -------
    perf : pd.DataFrame
        The daily performance of the algorithm.

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

    # initialize trading algorithm
    trading = TradingAlgorithm(sim_params,
                               slippage_model=slippage,
                               commission_model=commission,
                               execution_style=execution,
                               risk_allocation=allocation_policy,
                               restrictions=restricted_rules,
                               risk_models=risk_alert_policy,
                               risk_fuse=risk_fuse_policy,
                               metrics_set=metrics_set
                               )
    # trading.set_pipeline_final()
    # if algotext is None else {
    #         'algo_filename': getattr(algofile, 'name', '<algorithm>'),
    #         'script': algotext,
    #     }
    from pipe.term import Term
    kw = {'window': (5, 10), 'fields': ['close']}
    cross_term = Term('cross', kw)
    kw = {'fields': ['close'], 'window': 5, 'final': True}
    break_term = Term('break', kw, cross_term)
    pipeline = Pipeline([break_term, cross_term])
    trading.attach_pipeline(pipeline)
    # set trading control models
    trading.set_max_position_size(position_control_threshold)
    trading.set_max_order_size(order_control_threshold)
    trading.set_long_only()
    # set account models
    # trading.set_net_leverage(1.3)
    # run algorithm
    analysis = trading.run()
    print('analysis', analysis)
    # analysis_path = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/metric/temp.json'
    # with open(analysis_path, 'w+') as f:
    #     json.dump(analysis, f)
    # analysis.to_pickle(output)


if __name__ == '__main__':

    run_algorithm()

