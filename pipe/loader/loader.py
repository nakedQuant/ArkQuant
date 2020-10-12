# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from pipe.loader.base import PipelineLoader
from _calendar.trading_calendar import calendar
from pipe.loader import EVENT
from gateway.driver.data_portal import DataPortal
from pipe.term import Term
from gateway.asset.assets import Equity

__all__ = ['PricingLoader', 'EventLoader']

# init portal
portal = DataPortal()


class PricingLoader(PipelineLoader):

    def __init__(self, terms):
        """
            A pipe for loading daily adjusted qfq live OHLCV data.
            terms --- pipe terms and ump terms
        """
        domains = [term.domain for term in terms]
        self.pipeline_domain = self._resolve_domains(domains)

    def load_pipeline_arrays(self, dts, assets, data_frequency):
        fields = list(self.pipeline_domain.domain_field)
        print('loader fields', fields)
        window = - abs(self.pipeline_domain.domain_window)
        print('loader window', window)
        adjust_kline = portal.get_history_window(assets,
                                                 dts,
                                                 window,
                                                 fields,
                                                 data_frequency
                                                 )
        return adjust_kline


class EventLoader(PipelineLoader):
    """
        release massive  holder
        Base class for PipelineLoaders that supports loading the next and previous
        value of an event field.
        columns = ['sid','release_date','release_type','cjeltszb']
        --- 根据解禁类型：首发原股东限售股份，股权激励限售股份，定向增发机构配售股份分析解禁时点前的收益情况，与之后的收益率的关系
        before_event after_event

        Base class for PipelineLoaders that supports loading the next and previous
        value of an event field.
        columns = ['sid','declared_date','股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例']
        --- 方式:增持与减持 股东

        Base class for PipelineLoaders that supports loading the next and previous
        value of an event field.
        columns = ['declared_date','sid','bid_price','discount','bid_volume','buyer','seller','cleltszb']
        --- 主要折价率和每天大宗交易暂流通市值的比例 ，第一种清仓式出货， 第二种资金对导接力
    """
    def __init__(self, terms):
        domains = [term.domain for term in terms]
        self.pipeline_domain = self._resolve_domains(domains, True)

    def load_pipeline_arrays(self, dt, assets, data_frequency='daily'):
        assert data_frequency == 'daily', 'event has not minutes'
        event_mappings = dict()
        fields = list(self.pipeline_domain.domain_field)
        window = self.pipeline_domain.domain_window
        sdate = calendar.dt_window_size(dt, - window)
        for field in fields:
            raw =  EVENT[field].load_raw_arrays([sdate, dt], assets)
            event_mappings[field] = raw
        return event_mappings


if __name__ == '__main__':

    asset = [Equity('600000')]
    date = '2020-09-30'
    kw_p = {'window': (5, 10), 'fields': ['close']}
    cross_term_p = Term('cross', kw_p)
    kw_p_1 = {'window': 10, 'fast': 12, 'slow': 26, 'period': 9, 'fields': ['high', 'amount']}
    break_term_p = Term('break', kw_p_1, cross_term_p)
    terms = [break_term_p, cross_term_p]
    pricing = PricingLoader(terms)
    kline = pricing.load_pipeline_arrays(date, asset, 'daily')
    print('kline', kline)
    kw = {'window': (5, 10), 'fields': ['massive', 'release']}
    cross_term = Term('cross', kw)
    kw = {'window': 10, 'fast': 12, 'slow': 26, 'period': 9, 'fields': ['ownership', 'holder']}
    break_term = Term('break', kw, cross_term)
    terms = [break_term, cross_term]
    event = EventLoader(terms)
    event_kline = event.load_pipeline_arrays(date, asset)
    print('event_kline', event_kline)

