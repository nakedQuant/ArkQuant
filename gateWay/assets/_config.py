# Copyright 2016 Quantopian, Inc.
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

equity_url = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&' \
             'fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'

dual_url = 'http://19.push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=20&po=1&np=1&invt=2&fid=f3&' \
           'fs=b:DLMK0101&fields=f12,f191,f193'

bond_url = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=KZZ_LB2.0' \
           '&token=70f12f2f4f091e459a279469fe49eca5&cmd=&sr=-1&p=%d&ps=50&js={"pages":(tp),"data":(x)} '

fund_url = 'http://fund.eastmoney.com/cnjy_jzzzl.html'

ASSERT_URL_MAPPING = {
                    'equity':equity_url,
                    'bond':bond_url,
                    'fund':fund_url,
                    'dual':dual_url
                    }

equity_supplement_url = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/%s.phtml'

bond_supplement_url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'

ASSET_SUPPLEMENT_URL = {
                    'equity_supplement':equity_supplement_url,
                    'convertible_supplement':bond_supplement_url
                        }


__all__ = [ASSERT_URL_MAPPING,ASSET_SUPPLEMENT_URL]