# -*- coding : utf-8 -*-

from .base import PipelineLoader


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
    def __init__(self,terms):
        domains = [term.domain for term in terms]
        self.pipeline_domain = self._resolve_domains(domains,True)

    def load_pipeline_arrays(self,edate):
        fields = self.pipeline_domain.fields
        window = self.pipeline_domain.window
        event_raw = dict()
        for event_type in fields:
            method = 'load_%s_kline'%event_type
            raw = self.event_loader.__getattribute__(method)(edate,window)
            event_raw[event_type] = raw.set_index(['sid','declared_date'],inplace = True)
        return event_raw
