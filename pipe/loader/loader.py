# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from .base import PipelineLoader
from _calendar.trading_calendar import calendar
from pipe.loader import EVENT


class PricingLoader(PipelineLoader):

    def __init__(self, terms, data_portal):
        """
            A pipe for loading daily adjusted qfq live OHLCV data.
            terms --- pipe terms and ump terms
        """
        domains = [term.domain for term in terms]
        self.pipeline_domain = self._resolve_domains(domains)
        self._data_portal = data_portal

    def load_pipeline_arrays(self, dts, assets, data_frequency):
        fields = self.pipeline_domain.fields
        window = self.pipeline_domain.window
        adjust_kline = self._data_portal.history(self,
                                                 assets,
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

    def load_pipeline_arrays(self, dts, assets, data_frequency='daily'):
        if data_frequency == 'minute':
            raise ValueError('event data only be daily frequency')
        event_mappings = dict()
        fields = self.pipeline_domain.fields
        window = self.pipeline_domain.window
        assert set(fields).issubset(set(EVENT)), ValueError('unknown event')
        sessions = calendar.session_in_window(dts, window, include=False)
        for field in fields:
            raw = EVENT[field].load_raw_arrays(self, sessions, assets)
            event_mappings[field] = raw
        return event_mappings
