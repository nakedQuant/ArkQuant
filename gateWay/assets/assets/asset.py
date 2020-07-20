# test_cython: embedsignature=True
#
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

import numpy as np, sqlalchemy as sa
from abc import ABC,abstractmethod
from gateWay.driver.db_schema import engine,metadata
from gateWay.spider.engine import TsClient
from gateWay.driver.trading_calendar import TradingCalendar

""" 
    is_alive_for_session
    auto_close_date
"""
#sidview
class SidView:
    """
    This class exists to temporarily support the deprecated data[sid(N)] API.
    """
    def __init__(self, asset, data_portal, simulation_dt_func, data_frequency):
        """
        Parameters
        ---------
        asset : Asset
            The asset for which the instance retrieves data.

        data_portal : DataPortal
            Provider for bar pricing data.

        simulation_dt_func: function
            Function which returns the current simulation time.
            This is usually bound to a method of TradingSimulation.

        data_frequency: string
            The frequency of the bar data; i.e. whether the data is
            'daily' or 'minute' bars
        """
        self.asset = asset
        self.data_portal = data_portal
        self.simulation_dt_func = simulation_dt_func
        self.data_frequency = data_frequency

    def __getattr__(self, column):
        # backwards compatibility code for Q1 API
        if column == "close_price":
            column = "close"
        elif column == "open_price":
            column = "open"
        elif column == "dt":
            return self.dt
        elif column == "datetime":
            return self.datetime
        elif column == "sid":
            return self.sid

        return self.data_portal.get_spot_value(
            self.asset,
            column,
            self.simulation_dt_func(),
            self.data_frequency
        )

    def __contains__(self, column):
        return self.data_portal.contains(self.asset, column)

    def __getitem__(self, column):
        return self.__getattr__(column)

    @property
    def sid(self):
        return self.asset

    @property
    def dt(self):
        return self.datetime

    @property
    def datetime(self):
        return self.data_portal.get_last_traded_dt(
            self.asset,
            self.data_frequency)

    @property
    def current_dt(self):
        return self.simulation_dt_func()

    def mavg(self, num_minutes):
        self._warn_deprecated("The `mavg` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "mavg", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def stddev(self, num_minutes):
        self._warn_deprecated("The `stddev` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "stddev", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def vwap(self, num_minutes):
        self._warn_deprecated("The `vwap` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "vwap", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def returns(self):
        self._warn_deprecated("The `returns` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "returns", self.simulation_dt_func(),
            self.data_frequency
        )


class Asset(ABC):
    """
    Base class for entities that can be owned by a trading algorithm.

    Attributes
    ----------
    sid : int
        Persistent unique identifier assigned to the asset.
    symbol : str
        Most recent ticker under which the asset traded. This field can change
        without warning if the asset changes tickers. Use ``sid`` if you need a
        persistent identifier.
    asset_name : str
        Full name of the asset.
    exchange : str
        Canonical short name of the exchange on which the asset trades (e.g.,
        'NYSE').
    country_code : str
        Two character code indicating the country in which the asset trades.
    last_traded : pd.Timestamp
        Date on which the asset first traded.
    last_traded : pd.Timestamp
        Last date on which the asset traded. On Quantopian, this value is set
        to the current (real time) date for assets that are still trading.
    tick_size : float
        Minimum amount that the price can change for this asset.

    """
    trading_calendar = TradingCalendar()

    @abstractmethod
    def _retrieve_basics(self):
        """
            bacis information for asset
        """
        raise NotImplementedError

    @property
    def country_code(self):
        return 'China'

    @property
    def origin(self):
        """
            --- specify algorithm which simulate the asset
        """
        return None

    @origin.setattr
    def origin(self,value):
        return value

    @property
    def tick_size(self):
        return 100

    @property
    def multiple(self):
        """是tick_size的倍数"""
        return True

    @property
    def interday(self):
        """
            T + 1交易
        """
        return False

    @abstractmethod
    def price_limit(self,dt):
        raise NotImplementedError

    @property
    def bid_rule(self):
        return None

    @property
    def last_traded(self):
        return None

    def is_alive(self, session_label):
        """
        Returns whether the asset is alive at the given dt.

        Parameters
        ----------
        session_label: pd.Timestamp
            The desired session label to check. (midnight UTC)

        Returns
        -------
        boolean: whether the asset is alive at the given dt.
        """
        if self.last_traded and self.last_traded > session_label:
            return True
        elif not self.last_traded:
            return True
        else:
            return False

    @abstractmethod
    def supplement_for_asset(self):

        raise NotImplementedError

    def __repr__(self):

        return '%s(%d [%s])' % (type(self).__name__, self.sid)

    def __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.asset_name,
                                 self.first_traded,
                                 self.last_traded,
                                 self.tick_size,
                                 self.exchange
                                 ))

    def to_dict(self):
        """Convert to a python dict containing all attributes of the asset.

        This is often useful for debugging.

        Returns
        -------
        as_dict : dict
        """
        return {
            'sid': self.sid,
            'tick_size': self.tick_size,
            'asset_name': self.asset_name,
            'first_traded': self.first_traded,
            'last_traded': self.last_traded,
            'exchange': self.exchange,
            'country': self.country_code,
        }


class Equity(Asset):
    """
    Asset subclass representing partial ownership of a company, trust, or
    partnership.
    """
    def __init__(self,sid):

        self.sid = sid
        for key,value in self._retrieve_basics():
            self.__setattr__(key,value)

        if sid.startwith('688'):
            self.check_point = self.trading_calendar._roll_forward(self.first_traded,6)
            self.tick_size = 200
            # 200以上增加个数 --- 201
            self.multiple = False

    @property
    def asset_type(self):
        return 'symbol'

    def price_limit(self,dt):
        """
            科创板股票上市后的前5个交易日不设涨跌幅限制，从第六个交易日开始设置20%涨跌幅限制
            前5个交易日，科创板还设置了临时停牌制度，当盘中股价较开盘价上涨或下跌幅度首次达到30%、60%时，都分别进行一次临时停牌
            单次盘中临时停牌的持续时间为10分钟。每个交易日单涨跌方向只能触发两次临时停牌，最多可以触发四次共计40分钟临时停牌。
        """

        if dt == self.first_traded:
            price_limit = np.inf if self.sid.startwith('688') else 0.44
        elif dt < self.check_point:
            price_limit = np.inf if self.sid.startwith('688') else 0.1
        elif dt > self.check_point:
            price_limit = 0.2 if self.sid.startwith('688') else 0.1
        return price_limit

    @property
    def bid_rule(self):
        """在临时停牌阶段，投资者可以继续申报也可以撤销申报，并且申报价格不受2%的报价限制。复牌时，对已经接受的申报实行集合竞价撮合交易"""
        return 0.02 if self.sid.startwith('688') else None

    @property
    def last_traded(self):
        tbl = metadata.tables['symbol_span']
        ins = sa.select([tbl.delist_date]).where(tbl.c.sid == self.sid)
        rp = self.engine.execute(ins)
        return rp.scalar()

    def _retrieve_basics(self):
        """
            bacis information for asset
        """
        tbl = metadata.tables['symbol_basics']
        ins = sa.select([tbl.c.ipo_date,
                         tbl.c.initial_price,
                         tbl.c.name,
                         tbl.c.broker,
                         tbl.c.district]).where(tbl.c.sid == self.sid)
        rp = engine.execute(ins)
        info = pd.DataFrame(rp.fetchall(),columns = ['first_traded',
                                                    'initial_price',
                                                    'asset_name',
                                                    'broker',
                                                    'area'])
        basics = info.iloc[0,:].to_dict()
        return basics

    def supplement_for_asset(self):
        """
            extra information about asset --- equity structure
            股票的总股本、流通股本，公告日期,变动日期结构
            Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
            Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
            --- 将 -- 变为0
        """
        table = metadata.tables['symbol_equity_basics']
        ins = sa.select([table.c.declared_date, table.c.effective_day,
                         sa.cast(table.c.general, sa.Numeric(20, 3)),
                         sa.cast(table.c.float, sa.Numeric(20, 3)),
                         sa.cast(table.c.strict, sa.Numeric(20, 3))]).where(
            table.c.sid == self.sid)
        rp = self.engine.execute(ins)
        raw = rp.fetchall()
        equity = pd.DataFrame(raw,columns = ['declared_date', 'effective_day', 'general', 'float', 'strict'])
        return equity


class Convertible(Asset):
    """
       我国《上市公司证券发行管理办法》规定，可转换公司债券的期限最短为1年，最长为6年，自发行结束之日起6个月方可转换为公司股票
       回售条款 --- 最后两年
       1.强制赎回 --- 股票在任何连续三十个交易日中至少十五个交易日的收盘价格不低于当期转股价格的125%(含 125%)
       2.回售 --- 公司股票在最后两个计息年度任何连续三十个交易日的收盘价格低于当期转股价格的70%时
       3. first_traded --- 可转摘转股日期
       限制条件:
       1.可转换公司债券流通面bai值少于3000万元时，交易所立即公告并在三个交易日后停止交易
       2.可转换公司债券转换期结束前的10个交易日停止交易
       3.中国证监会和交易所认为必须停止交易
    """
    def __init__(self,bond):

        self.sid = bond
        self.price_limit = None
        self.interday = True
        for key, value in self._retrieve_basics():
            self.__setattr__(key, value)

    @property
    def asset_type(self):
        return 'bond'

    def price_limit(self,dt):
        return np.inf

    def _retrieve_basics(self):
        """
            bacis information for asset
        """
        tbl = metadata.tables['bond_basics']
        ins = sa.select([tbl.c.sid,
                         tbl.c.name,
                         tbl.c.put_price,
                         tbl.c.convert_price,
                         tbl.c.convert_dt,
                         tbl.c.maturity_dt,
                         tbl.c.force_redeem_price,
                         tbl.c.put_convert_price,
                         tbl.c.guarantor]).\
            where(tbl.c.sid == self.sid)
        rp = self.engine.execute(ins)
        bond_info = pd.DataFrame(rp.fetchall(),columns = [
                                                        'sid',
                                                        'asset_name',
                                                        'put_price',
                                                        'convert_price',
                                                        'first_traded',
                                                        'last_traded',
                                                        'force_redeem_price',
                                                        'put_convert_price',
                                                        'guarantor'])
        basics = bond_info.iloc[0,:].to_dict()
        return basics

    def supplement_for_asset(self):
        """
            nothing for supplement
        """


class Fund(Asset):
    """
    ETF --- exchange trade fund
    目前不是所有的ETF都是t+0的，只有跨境ETF、债券ETF、黄金ETF、货币ETF实行的是t+0，境内A股ETF暂不支持t+0
    10%
    """
    def __init__(self,sid):
        self.sid = sid

    @property
    def asset_type(self):
        return 'fund'

    @property
    def first_traded(self):
        tbl = metadata.tables['fund_price']
        ins = sa.select([sa.func.min(tbl.c.trade_dt)]).\
            where(tbl.c.sid == self.sid)
        rp = engine.execute(ins)
        init_date = rp.fetchall()[0]
        return init_date

    def price_limit(self,dt):
        return 0.1

    def supplement_for_asset(self):
        """
            nothing for supplement
        """

import json, re, pandas as pd,datetime
from weakref import WeakValueDictionary
from functools import lru_cache

from gateWay.tools import _parse_url
from gateWay.driver.trading_calendar import TradingCalendar


ASSERT_URL_MAPPING = {
    'stock': 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&'
             'fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12',
    'bond': 'https://www.jisilu.cn/data/cbnew/cb_list/?',
    'etf': "http://vip.stock.finance.sina.com.cn/quotes_service/api/jsonp.php/IO.XSRV2.CallbackList"
           "['v7BkqPkwcfhoO1XH']/Market_Center.getHQNodeDataSimple?page=%d&num=80&sort=symbol&asc=0&node=etf_hq_fund",
    'dual_stock': 'http://19.push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=20&po=1&np=1&invt=2&fid=f3&' \
                  'fs=b:DLMK0101&fields=f12,f191,f193',
    'suspend':'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=1&ps=500&'
              'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=%s'
                    }


class AssetFinder(object):
    """
        AssetFinder is an interface to a database of Asset metadata written by
        an AssetDBWriter
        Asset is mainly concentrated on a_stock which relates with corresponding h_stock and convertible_bond;
        besides etf , benchmark
        基于 上市时间 注册地 主承商 ，对应H股, 可转债
    """
    _cache = WeakValueDictionary()

    def __new__(cls,
                engine,
                metadata):
        identity = (engine,metadata)
        try:
            instance = cls._cache[identity]
        except KeyError:
            instance = cls._cache[identity] = super(AssetFinder,cls).__new__(cls)._initialize(*identity)
        return instance

    def _initialize(self,engine,metadata):
        self.engine = engine
        #反射
        for table_name,table_object in metadata.tables.items():
            setattr(self,table_name,table_object)
        self.ts = TsClient()
        self.trading_calendar = TradingCalendar()
        self.adjust_array = AdjustArray()
        return self

    @lru_cache(maxsize= 8)
    def lookup_assets(self,_type):
        if _type == 'symbol':
            # 获取存量股票包括退市
            raw = json.loads(_parse_url(ASSERT_URL_MAPPING['stock'], bs=False))
            assets = [item['f12'] for item in raw['data']['diff']]
            # sci_tech = list(filter(lambda x: x.startswith('688'), q))
            # tradtional = set(q) - set(sci_tech)
        elif _type == 'etf':
            """获取ETF列表 page num --- 20 40 80"""
            assets = []
            page = 1
            while True:
                url = ASSERT_URL_MAPPING['etf'] % page
                obj = _parse_url(url)
                text = obj.find('p').get_text()
                mid = re.findall('s[z|h][0-9]{6}', text)
                if len(mid) > 0:
                    assets.extend(mid)
                else:
                    break
                page = page + 1
        elif _type == 'bond':
            """可转债  --- item['cell']['stock_id'] item['id'] """
            text = _parse_url(ASSERT_URL_MAPPING['bond'], bs=False, encoding=None)
            text = json.loads(text)
            assets = text['rows']
        else:
            raise ValueError()
        return assets

    def retrieve_asset(self,asset_id):
        raise NotImplementedError()

    def fuzzy_symbol_ownership_by_district(self,area):
        """
            基于区域地址找到对应的股票代码
        """
        rp = sa.select(self.symbol_basics.c.code).\
                         where(self.symbol_basics.c.district == area)

        assets = [r[0] for r in self.engine.execute(rp).fetchall()]
        return assets

    def fuzzy_symbol_ownership_by_broker(self,broker):
        """
            基于主承商找到对应的股票代码
        """
        rp = sa.select(self.symbol_basics.c.code).\
                      where(self.symbol_basics.c.broker == broker)
        assets = [r[0] for r in self.engine.execute(rp).fetchall()]
        return assets

    def fuzzy_bond_ownership_by_symbol(self,code):
        """
            基于A股代码找到对应的可转债数据
        """
        rp = sa.select(self.bond_basics.bond_id).\
                  where(self.bond_basics.stock_id == code)
        assets = [r[0] for r in self.engine.execute(rp).fetchall()]
        return assets

    def fuzzy_Hsymbol_ownership_by_A(self,sid):
        """
            基于A股代码找到对应的H股代码
        """
        hstock= sa.select(self.dual_symbol.hk).\
                  where(self.dual_symbol.c.sid == sid).\
                  execute().scalar()
        return hstock

    def fuzzy_symbol_ownership_by_exchange(self, exchange, flag=1):
        """获取沪港通、深港通股票 , exchange 交易所 ; flag :1 最新的， 0 为历史的已经踢出的"""
        assets = self.ts.to_ts_con(exchange, flag)
        return assets

    def fuzzy_symbol_ownership_by_hk(self):
        """ 获取AH,A股与H股同时上市的股票 """
        dual_assets = pd.DataFrame()
        page = 1
        while True:
            url = ASSERT_URL_MAPPING['dual_stock'] % page
            raw = _parse_url(url, bs=False, encoding=None)
            raw = json.loads(raw)
            diff = raw['data']
            if diff and len(diff['diff']):
                diff = [[item['f12'], item['f191'], item['f193']] for item in diff['diff']]
                raw = pd.DataFrame(diff, columns=['h_code', 'code', 'name'])
                assets = dual_assets.append(raw)
                page = page + 1
            else:
                break
        return dual_assets

    def fuzzy_symbol_ownership_by_ipodate(self,date,window):
        """
            基于上市时间筛选出 --- 上市时间不满足一定条件的标的
        """
        shift_date = self.trading_calendar._roll_forward(date,window)

        rp = sa.select(self.symbol_basics.c.code).\
                      where(self.symbol_basics.c.initial_date < shift_date)

        assets = [r[0] for r in self.engine.execute(rp).fetchall()]
        return assets

    def fuzzy_symbols_ownership_by_suspend(self,date):
        # 停盘的股票 --- 获取实时  --- 当天早上执行
        if date == datetime.datetime.strftime(datetime.datetime.now(),'%y%m%d'):
            html_path = ASSERT_URL_MAPPING['suspend']%date
            raw = _parse_url(html_path, 'utf-8')
            text = json.loads(raw.get_text())
            data = [item.split(',') for item in text['data']]
            df = pd.DataFrame(data)
            suspend = df[not df.iloc[:,-1] & df.iloc[:,-1] > date]
            supsend_assets = suspend.iloc[:,0].values.tolist()
        else:
            #历史停盘的股票 --- 停盘的没有数据
            assets = self.lookup_assets('symbol')
            daily = self.adjust_array.load_raw_array(date,0,['code'],['symbol'])
            supsend_assets = assets - set(daily.keys())
        return supsend_assets

    def fuzzy_symbol_ownership_by_delist(self,date,window = 30):
        #剔除处于退市整理期的股票，一般是30个交易日  --- 收到退市决定后申请复合，最后一步进入退市整理期30个交易日
        assets = self.lookup_assets('symbol')
        shift_date = self.trading_calendar._roll_forward(date,-window)
        _symbols = [assets for asset in assets if asset.is_alive(date) and date > shift_date]
        return _symbols
