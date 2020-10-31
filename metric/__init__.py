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
    SymbolPNL,
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
                        max_duration,
                        max_succession
                        )


def default_metrics():
    return {
        SessionField(),
        NumTradingDays(),
        _ConstantCumulativeRiskMetric('project', 'ArkQuant'),
        _ConstantCumulativeRiskMetric('treasury_period_return', '0.0'),
        # DailyLedgerField('positions'),
        DailyLedgerField('portfolio.utility'),
        DailyLedgerField('portfolio.portfolio_cash'),
        DailyLedgerField('portfolio.portfolio_value'),
        DailyLedgerField('portfolio.positions_values'),
        DailyLedgerField('portfolio.current_portfolio_weights'),
        PNL(),
        SymbolPNL(),
        Returns(),
        # Transactions(),
        HitRate(),
        BenchmarkReturnsAndVolatility(),
        AlphaBeta(),
        ReturnsStatistic(sharpe_ratio, 'sharpe'),
        ReturnsStatistic(sortino_ratio, 'sortino'),
        ReturnsStatistic(max_drawdown, 'max_drawdown'),
        ReturnsStatistic(max_duration, 'max_duration'),
        ReturnsStatistic(max_succession, 'max_succession'),
        ReturnsStatistic(annual_volatility, 'annual_volatility'),
    }


def classic_metrics():
    metrics_set = default_metrics()
    metrics_set.add(_ClassicRiskMetrics())
    return metrics
