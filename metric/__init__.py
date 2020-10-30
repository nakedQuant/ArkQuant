#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from metric.tracker import _ClassicRiskMetrics
from metric.metrics import (
    NumTradingDays,
    SessionField,
    DailyLedgerField,
    PNL,
    Returns,
    HitRate,
    Transactions,
    BenchmarkReturnsAndVolatility,
    AlphaBeta,
    ReturnsStatistic,
    _ConstantCumulativeRiskMetric,
)
from metric.analyzers import (
                        sortino_ratio,
                        sharpe_ratio,
                        annual_volatility,
                        max_drawdown,
                        )


def default_metrics():
    return {
        SessionField(),
        NumTradingDays(),
        _ConstantCumulativeRiskMetric('project', 'ArkQuant'),
        _ConstantCumulativeRiskMetric('treasury_period_return', '0.0'),
        DailyLedgerField('portfolio.portfolio_value'),
        DailyLedgerField('portfolio.positions_values'),
        DailyLedgerField('portfolio.portfolio_cash'),
        DailyLedgerField('portfolio.utility'),
        DailyLedgerField('portfolio.positions'),
        DailyLedgerField('portfolio.current_portfolio_weights'),
        DailyLedgerField('positions'),
        PNL(),
        Returns(),
        Transactions(),
        HitRate(),
        BenchmarkReturnsAndVolatility(),
        AlphaBeta(),
        # ReturnsStatistic(sharpe_ratio, 'sharpe'),
        # ReturnsStatistic(sortino_ratio, 'sortino'),
        # ReturnsStatistic(max_drawdown, 'max_down'),
        # ReturnsStatistic(annual_volatility, 'algorithm_volatility'),
    }


def classic_metrics():
    metrics_set = default_metrics()
    metrics_set.add(_ClassicRiskMetrics())
    return metrics
