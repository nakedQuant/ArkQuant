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
from numpy import iinfo, uint32
from .base import PipelineLoader

UINT32_MAX = iinfo(uint32).max


class PricingLoader(PipelineLoader):

    def __init__(self,terms):
        """
            A pipeline for loading daily adjusted qfq live OHLCV data.
            terms --- pipeline terms and ump terms
        """
        domains = [term.domain for term in terms]
        self.pipeline_domain = self._resolve_domains(domains)

    def load_pipeline_arrays(self,edate,_types):
        fields = self.pipeline_domain.fields
        window = self.pipeline_domain.window
        if 'symbol' in _types:
            adjust_kline = self.adjusted_loader.load_adjusted_array(
                edate,
                window,
                fields,
            )
        extra_types = set(_types) - set('symbol')
        raw = self.reader.load_asset_kline(
            edate,
            window,
            fields,
            extra_types
        )
        adjust_kline.update(raw)
        return adjust_kline