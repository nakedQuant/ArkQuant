# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .base import PipelineLoader

EVENT = frozenset(['massive','release','holder','structure','gross','margin'])


class PricingLoader(PipelineLoader):

    def __init__(self,terms,reader):
        """
            A pipeline for loading daily adjusted qfq live OHLCV data.
            terms --- pipeline terms and ump terms
        """
        self._pricing_reader = reader
        domains = [term.domain for term in terms]
        self.pipeline_domain = self._resolve_domains(domains)

    def load_pipeline_arrays(self,dts,sids):
        fields = self.pipeline_domain.fields
        window = self.pipeline_domain.window
        adjust_kline = self._pricing_reader.history(self,
                                                    sids,
                                                    fields,
                                                    dts,
                                                    window)
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
    def __init__(self,terms,event_reader):
        self._reader_dct = event_reader
        domains = [term.domain for term in terms]
        self.pipeline_domain = self._resolve_domains(domains,True)

    def load_pipeline_arrays(self,dts,sids,op_name):
        fields = self.pipeline_domain.fields
        window = self.pipeline_domain.window
        assert op_name in EVENT , ValueError('unidentified event')
        raw = self._reader_dct[op_name].load_raw_arrays(self,
                                                                dts,
                                                                window,
                                                                sids
                                                                )
        kline = raw.loc[:,fields]
        return kline