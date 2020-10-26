#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from metric.tracker import _ClassicRiskMetrics
from metric.metrics import (
    SessionField,
    DailyLedgerField,
    PNL,
    CashFlow,
    Returns,
    Profits,
    Weights,
    Utility,
    Cushion,
    Proportion,
    HitRate,
    Positions,
    Transactions,
    PeriodLabel,
    NumTradingDays,
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
        DailyLedgerField('portfolio.portfolio_value'),
        DailyLedgerField('portfolio.position_values'),
        DailyLedgerField('portfolio.start_cash', 'cash'),
        DailyLedgerField('portfolio.utility', 'capital_utility'),
        DailyLedgerField('portfolio.positions', 'ending_exposure'),
        DailyLedgerField('portfolio.current_portfolio_weights', 'portfolio_weights'),
        PNL(),
        CashFlow(),
        Transactions(),
        Returns(),
        Profits(),
        Weights(),
        Utility(),
        Cushion(),
        Proportion(),
        HitRate(),
        Positions(),
        BenchmarkReturnsAndVolatility(),
        AlphaBeta(),
        ReturnsStatistic(sharpe_ratio, 'sharpe'),
        ReturnsStatistic(sortino_ratio, 'sortino'),
        ReturnsStatistic(max_drawdown, 'max_down'),
        ReturnsStatistic(annual_volatility, 'algorithm_volatility'),
        NumTradingDays(),
        PeriodLabel(),
        _ConstantCumulativeRiskMetric('base_capital', 0.0),
        _ConstantCumulativeRiskMetric('treasury_period_return', 0.0),
    }


def classic_metrics():
    metrics = default_metrics()
    metrics.add(_ClassicRiskMetrics())
    return metrics
