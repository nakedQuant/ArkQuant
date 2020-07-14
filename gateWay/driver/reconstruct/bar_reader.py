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
from abc import ABC, abstractmethod

OHLCV = ('open', 'high', 'low', 'close', 'volume')


class BarReader(ABC):

    @property
    def data_frequency(self):
        return 'daily'

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        return self._trading_calendar

    def _window_size_to_dt(self,date,window):
        shift_date = self.trading_calendar.shift_calendar(date, window)
        return shift_date

    @abstractmethod
    def get_value(self, asset, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        asset : Asset
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_raw_arrays(self, date, window,columns,assets):
        """
        Parameters
        ----------
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        raise NotImplementedError()


#reader
class AssetSessionReader(BarReader):
    """
    Reader for raw pricing data from mysql.
    return different asset types : etf bond symbol
    """
    AssetsTypes = frozenset(['symbol','bond','etf','dual'])

    def __init__(self,
                 metadata,
                 engine,
                 trading_calendar,
                ):
        self.metadata = metadata
        self.engine = engine
        self._trading_calendar = trading_calendar

    def get_value(self, asset,dt):
        table_name = '%s_price'%asset.asset_type
        tbl = self.metadata[table_name]
        orm = select([cast(tbl.c.open, Numeric(10, 2)).label('open'),
                      cast(tbl.c.close, Numeric(12, 2)).label('close'),
                      cast(tbl.c.high, Numeric(10, 2)).label('high'),
                      cast(tbl.c.low, Numeric(10, 3)).label('low'),
                      cast(tbl.c.volume, Numeric(15, 0)).label('volume'),
                      cast(tbl.c.amount, Numeric(15, 2)).label('amount')])\
            .where(and_(tbl.c.trade_dt == dt,tbl.c.sid == asset.sid))
        rp = self.engine.execute(orm)
        arrays = [[r.trade_dt, r.code, r.open, r.close, r.high, r.low, r.volume] for r in
                  rp.fetchall()]
        kline = pd.DataFrame(arrays,
                             columns=['open', 'close', 'high', 'low', 'volume',
                                      'amount'])
        return kline

    def _retrieve_asset_type(self,table,sids,fields,start_date,end_date):
        tbl = self.metadata['%s_price' & table]
        orm = select([tbl.c.trade_dt, tbl.c.sid,
                      cast(tbl.c.open, Numeric(10, 2)).label('open'),
                      cast(tbl.c.close, Numeric(12, 2)).label('close'),
                      cast(tbl.c.high, Numeric(10, 2)).label('high'),
                      cast(tbl.c.low, Numeric(10, 3)).label('low'),
                      cast(tbl.c.volume, Numeric(15, 0)).label('volume'),
                      cast(tbl.c.amount, Numeric(15, 2)).label('amount')]). \
            where(tbl.c.trade_dt.between(start_date, end_date))
        rp = self.engine.execute(orm)
        arrays = [[r.trade_dt, r.code, r.open, r.close, r.high, r.low, r.volume] for r in
                  rp.fetchall()]
        raw = pd.DataFrame(arrays,
                             columns=['trade_dt', 'code', 'open', 'close', 'high', 'low', 'volume',
                                      'amount'])
        raw.set_index('code', inplace=True)
        # 基于code
        _slice = raw.loc[sids]
        # 基于fields 获取数据
        kline = _slice.loc[:, fields]
        unpack_kline = unpack_df_to_component_dict(kline)
        return unpack_kline

    def load_raw_arrays(self,end_date,window,columns,assets):
        start_date = self._window_size_to_dt(end_date,window)
        _get_source = partial(self._retrieve_asset_type(fields= columns,
                                                           start_date= start_date,
                                                           end_date = end_date
                                                           ))
        sid_groups = {}
        for asset in assets:
            sid_groups[asset.asset_type].append(asset.sid)
        #验证
        assert set(sid_groups) in self.AssetsTypes,('extra asset types %r'%
                                                    (set(sid_groups) - self.AssetsTypes))
        #获取数据
        batch_arrays = {}
        for _name,sids in sid_groups.items():
            raw = _get_source(table = _name,sids = sids)
            batch_arrays.update(raw)
        return batch_arrays


class MassiveSessionReader(BarReader):

    def __init__(self,
                 metadata,
                 engine,
                 trading_calenar):
        self.metadata = metadata
        self.engine = engine
        self._trading_calenar = trading_calenar

    def get_value(self, asset, dt):
        table = self.metadata['massive']
        sql = select([cast(table.c.bid_price, Numeric(10,2)),
                      cast(table.c.discount, Numeric(10,5)),
                      cast(table.c.bid_volume, Integer),
                      table.c.buyer,
                      table.c.seller,
                      table.c.cleltszb]).where(and_(table.c.trade_dt == dt,table.c.sid == asset.sid))
        raw = self.engine.execute(sql).fetchall()
        share_massive = pd.DataFrame(raw,columns = ['bid_price','discount','bid_volume','buyer','seller','cleltszb'])
        return share_massive

    def load_raw_arrays(self, edate, window,assets):
        sdate = self._window_size_to_dt(edate,window)
        sids = [asset.sid for asset in assets]
        #获取数据
        table = self.metadata['massive']
        sql = select([table.c.trade_dt,
                      table.c.sid,
                      cast(table.c.bid_price, Numeric(10,2)),
                      cast(table.c.discount, Numeric(10,5)),
                      cast(table.c.bid_volume, Integer),
                      table.c.buyer,
                      table.c.seller,
                      table.c.cleltszb]).where(table.c.trade_dt.between(sdate,edate))
        raw = self.engine.execute(sql).fetchall()
        df = pd.DataFrame(raw,columns = ['trade_dt','code','bid_price','discount','bid_volume','buyer','seller','cleltszb'])
        df.set_index('code',inplace= True)
        massive = df.loc[sids]
        return massive


class ReleaseSessionReader(BarReader):

    def __init__(self,
                 metadata,
                 engine,
                 trading_calendar):
        self.metadata = metadata
        self.engine = engine
        self._trading_calendar = trading_calendar

    def get_value(self, asset,dt):
        table = self.metadata['release']
        sql = select([cast(table.c.release_type, Numeric(10, 2)),
                      cast(table.c.cjeltszb, Numeric(10, 5)), ]).\
            where(and_(table.c.release_date == dt,table.c.sid == asset.sid))
        raw = self.engine.execute(sql).fetchall()
        release = pd.DataFrame(raw, columns=['release_type', 'cjeltszb'])
        return release

    def load_raw_arrays(self, edate, window,assets):
        sdate = self._window_size_to_dt(edate,window)
        sids = [asset.sid for asset in assets]
        table = self.metadata['release']
        sql = select([table.c.sid,
                      table.c.release_date,
                      cast(table.c.release_type, Numeric(10, 2)),
                      cast(table.c.cjeltszb, Numeric(10, 5)), ]).where \
            (table.c.release_date.between(sdate, edate))
        raw = self.engine.execute(sql).fetchall()
        df = pd.DataFrame(raw, columns=['code', 'release_date', 'release_type', 'cjeltszb'])
        df.set_index('code',inplace= True)
        releases = df.loc[sids]
        return releases


class ShareHolderSessionReader(BarReader):

    def __init__(self,
                 metadata,
                 engine,
                 trading_calendar):
        self.metadata = metadata
        self.engine = engine
        self._trading_calendar = trading_calendar

    def get_value(self, asset,dt):
        """股东持仓变动"""
        table = self.metadata['shareholder']
        sql = select([table.c.股东,
                      table.c.方式,
                      cast(table.c.变动股本, Numeric(10,2)),
                      cast(table.c.总持仓, Integer),
                      cast(table.c.占总股本比例, Numeric(10, 5)),
                      cast(table.c.总流通股, Integer),
                      cast(table.c.占总流通比例, Numeric(10, 5))]).where(and_(table.c.公告日 == dt,table.c.sid == asset.sid))
        raw = self.engine.execute(sql).fetchall()
        share_tracker = pd.DataFrame(raw,columns = ['股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例'])
        return share_tracker

    def load_raw_arrays(self, edate, window,assets):
        sdate = self._window_size_to_dt(edate,window)
        sids = [asset.sid for asset in assets]
        """股东持仓变动"""
        table = self.metadata['shareholder']
        sql = select([table.c.sid,
                      table.c.公告日,
                      table.c.股东,
                      table.c.方式,
                      cast(table.c.变动股本, Numeric(10,2)),
                      cast(table.c.总持仓, Integer),
                      cast(table.c.占总股本比例, Numeric(10, 5)),
                      cast(table.c.总流通股, Integer),
                      cast(table.c.占总流通比例, Numeric(10, 5))]).where(
                    table.c.公告日.between(sdate,edate))
        raw = self.engine.execute(sql).fetchall()
        df = pd.DataFrame(raw,columns = ['code','公告日','股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例'])
        df.set_index('code',inplace= True)
        trackers = df.loc[sids]
        return trackers


class GrossSessionReader(BarReader):

    GdpPath = 'http://data.eastmoney.com/cjsj/grossdomesticproduct.aspx?p=%d'

    def __init__(self,
                 trading_calendar,
                 url = None):
        self._trading_calendar = trading_calendar
        self._url = url if url else self.GdpPath

    def get_value(self, asset, dt, field):
        print('get_values is deprescated by gpd ,use load_raw_arrays method')

    def load_raw_arrays(self,edate,window):
        sdate = self._window_size_to_dt(edate,window)
        """获取GDP数据"""
        page = 1
        gross_value = pd.DataFrame()
        while True:
            req_url = self._url%page
            obj = _parse_url(req_url)
            raw = obj.findAll('div', {'class': 'Content'})
            text = [t.get_text() for t in raw[1].findAll('td')]
            text = [item.strip() for item in text]
            data = zip(text[::9], text[1::9])
            data = pd.DataFrame(data, columns=['季度', '总值'])
            gross_value = gross_value.append(data)
            if len(gross_value) != len(gross_value.drop_duplicates(ignore_index=True)):
                gross_value.drop_duplicates(inplace=True, ignore_index=True)
                return gross_value
            page = page + 1
        #截取
        start_idx = gross_value.index(sdate)
        end_idx = gross_value.index(edate)
        return gross_value.iloc[start_idx:end_idx +1,:]


class MarginSessionReader(BarReader):

    MarginPath = 'http://api.dataide.eastmoney.com/data/get_rzrq_lshj?' \
                 'orderby=dim_date&order=desc&pageindex=%d&pagesize=50'

    def __init__(self,
                 trading_calendar,
                 _url):
        self._trading_calendar = trading_calendar
        self._url = _url if _url else self.MarginPath

    def get_value(self, asset, dt, field):
        raise NotImplementedError('get_values is deprescated ,use load_raw_arrays method')

    def load_raw_arrays(self, edate, window):
        sdate = self._window_size_to_dt(edate,window)
        """获取市场全量融资融券"""
        page = 1
        margin = pd.DataFrame()
        while True:
            req_url = self._url% page
            raw = _parse_url(req_url, bs=False)
            raw = json.loads(raw)
            raw = [
                [item['dim_date'], item['rzye'], item['rqye'], item['rzrqye'], item['rzrqyecz'], item['new'],
                 item['zdf']]
                for item in raw['data']]
            data = pd.DataFrame(raw, columns=['trade_dt', 'rzye', 'rqye', 'rzrqze', 'rzrqce', 'hs300', 'pct'])
            data.loc[:, 'trade_dt'] = [datetime.datetime.fromtimestamp(dt / 1000) for dt in data['trade_dt']]
            data.loc[:, 'trade_dt'] = [datetime.datetime.strftime(t, '%Y-%m-%d') for t in data['trade_dt']]
            if len(data) == 0:
                break
            margin = margin.append(data)
            page = page + 1
        margin.set_index('trade_dt', inplace=True)
        #
        start_idx = margin.index(sdate)
        end_idx = margin.index(edate)
        return margin.iloc[start_idx:end_idx +1,:]


#Adjustments
import numpy as np

ADJUSTMENT_COLUMNS_DTYPE = {
                'sid_bonus':int,
                'sid_transfer':int,
                'bonus':np.float64,
                'right_price':np.float64
                            }


class SQLiteAdjustmentReader(object):
    """
        1 获取所有的分红 配股 数据用于pipeloader
        2.形成特定的格式的dataframe
    """

    # @preprocess(conn=coerce_string_to_conn(require_exists=True))
    def __init__(self,engine):
        self.conn = engine.conncect()

    def __enter__(self):
        return self

    @property
    def _calendar(self):
        return trading_calendar

    def _offset_session(self,date,window):
        sessions = self._calendar.sessions_in_range(date,window)
        return sessions

    def _get_divdends_with_ex_date(self,session):
        table = metadata.tables['equity_divdends']
        sql_dialect = sa.select([table.c.sid,
                                table.c.ex_date,
                                sa.cast(table.c.sid_bonus,sa.Numeric(5,2)),
                                sa.cast(table.c.sid_transfer,sa.Numeric(5,2)),
                                sa.cast(table.c.bonus,sa.Numeric(5,2))]).\
                                where(and_(table.c.pay_date.between(session[0],session[1]), table.c.progress.like('实施')))
        rp = self.conn.execute(sql_dialect)
        divdends = pd.DataFrame(rp.fetchall(),
                                     columns = ['code','ex_date','sid_bonus','sid_transfer','bonus'])
        adjust_divdends = self._generate_dict_from_dataframe(divdends)
        return adjust_divdends

    def _get_rights_with_ex_date(self,session):
        table = metadata.tables['equity_rights']
        sql = sa.select([table.c.sid,
                         table.c.ex_date,
                         sa.cast(table.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.right_price,sa.Numeric(5,2))]).\
                        where(sa.table.c.pay_date.between(session[0],session[1]))
        rp = self.conn.execute(sql)
        rights = pd.DataFrame(rp.fetchall(),
                              columns=['code','ex_date','right_bonus', 'right_price'])
        adjust_rights = self._generate_dict_from_dataframe(rights)
        return adjust_rights

    @staticmethod
    def _generate_dict_from_dataframe(df):
        df.set_index('sid',inplace = True)
        for col,col_type in ADJUSTMENT_COLUMNS_DTYPE.items():
            try:
                df[col] = df[col].astype(col_type)
            except KeyError:
                raise TypeError('%s cannot mutate into %s'%(col,col_type))
        #asset : splits or divdend or rights
        unpack_df = unpack_df_to_component_dict(df)
        return unpack_df

    def _load_adjustments_from_sqlite(self,sessions,
         should_include_dividends,
         should_include_rights,
        ):
        adjustments = {}
        if should_include_dividends:
            adjustments['divdends'] =  self._get_divdends_with_ex_date(sessions)
        elif should_include_rights:
            adjustments['rights'] =  self._get_rights_with_ex_date(sessions)
        else:
            adjustments = None
        return adjustments

    def load_pricing_adjustments(self,date,window,
                                 should_include_dividends=True,
                                 should_include_rights=True,
                                 ):
        sessions = self._offset_session(date,window)
        pricing_adjustments = self._load_adjustments_from_sqlite(sessions,
                                should_include_dividends,
                                should_include_rights)
        return pricing_adjustments

    def load_divdends_for_sid(self,sid,date):
        table = metadata.tables['symbol_divdends']
        sql_dialect = sa.select([table.c.ex_date,
                         sa.cast(table.c.sid_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.sid_transfer,sa.Numeric(5,2)),
                         sa.cast(table.c.bonus,sa.Numeric(5,2))]).\
                        where(sa.and_(table.c.sid == sid,table.c.progress.like('实施'),table.c.ex_date < date))
        rp = self.conn.execute(sql_dialect)
        divdends = pd.DataFrame(rp.fetchall(),
                                     columns = ['ex_date','sid_bonus','sid_transfer','bonus'])
        return divdends

    def load_rights_for_sid(self,sid,date):
        table = metadata.tables['symbol_rights']
        sql = sa.select([table.c.ex_date,
                         sa.cast(table.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.right_price,sa.Numeric(5,2))]).\
                        where(sa.and_(table.c.sid == sid, table.c.ex_date < date))
        rp = self.conn.execute(sql)
        rights = pd.DataFrame(rp.fetchall(),
                              columns=['ex_date','right_bonus', 'right_price'])
        return rights

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        return self.conn.close()


#计算系数
from functools import partial
import pandas as pd


class HistoryCompatibleAdjustments(object):
    """
        calculate adjustments coef
    """
    def __init__(self,
                 _reader,
                 adjustment_reader
                 ):
        self._adjustments_reader = adjustment_reader
        self._reader = _reader

    def _init_raw_array(self,assets,edate,window):
        if self._reader.data_frequency == 'daily':
            bars = self._reader.load_raw_arrays(edate, window, ['close'], assets)
        elif self._reader.data_frequency == 'minute':
            bars = self._reader.load_ticker_arrays(edate,window,assets,['close'],'15:00')
        return bars

    @staticmethod
    def _calculate_divdends_for_sid(adjustment,close,sid):
        """
           股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
           前复权：复权后价格=(复权前价格-现金红利)/(1+流通股份变动比例)
           后复权：复权后价格=复权前价格×(1+流通股份变动比例)+现金红利
        """
        bar = close[sid]
        try:
            divdend_fq = adjustment['divdends'][sid]
            exdate_close = bar['close'][bar['trade_dt'] == divdend_fq['ex_date']]
            qfq = (1- divdend_fq['bonus'] /
                          (10 * exdate_close)) / \
                         (1 + (divdend_fq['sid_bonus'] +
                               divdend_fq['sid_transfer']) / 10)
        except KeyError:
            qfq = pd.Series(1.0,index = bar['trade_dt'])
        return qfq

    @staticmethod
    def _calculate_rights_for_sid(adjustment,close,sid):
        """
           配股除权价=（除权登记日收盘价+配股价*每股配股比例）/（1+每股配股比例）
        """
        bar = close[sid]
        try:
            rights_fq = adjustment['rights'][sid]
            exdate_close = bar['close'][bar['trade_dt'] == rights_fq['ex_date']]
            qfq = (exdate_close + (rights_fq['rights_price'] *
                       rights_fq['rights_bonus']) / 10) \
                 / (1 + rights_fq['rights_bonus']/10)
        except KeyError:
            qfq = pd.Series(1.0,index = bar['trade_dt'])
        return qfq

    def _calculate_adjustments_for_sid(self,adjustment,close,sid):
        fq = self._calculate_divdends_for_sid(adjustment,close,sid)
        fq_rights = self._calculate_rights_for_sid(adjustment,close,sid)
        fq.append(fq_rights)
        fq.sort_index(ascending= False,inplace = True)
        qfq = 1 / fq.cumprod()
        return qfq

    def calculate_adjustments_in_sessions(self,edate,window,assets):
        """
        Returns
        -------
        adjustments : list[dict[int -> Adjustment]]
            A list, where each element corresponds to the `columns`, of
            mappings from index to adjustment objects to apply at that index.
        """
        adjs = {}
        #获取全部的分红除权配股数据
        adjustments = self._adjustments_reader.load_pricing_adjustments(edate,window)
        #获取对应的收盘价数据
        close_bars = self._init_raw_array(edate,window,assets)
        #计算前复权系数
        _calculate = partial(self._calculate_adjustments_for_sid,adjustments = adjustments,close = close_bars)
        for asset in assets:
            adjs[asset] = _calculate(sid = asset.sid)
        return adjs


class SlidingWindow(ABC):

    @property
    def frequency(self):
        return None

    @abstractmethod
    def _array(self):
        raise NotImplementedError()

    @abstractmethod
    def _window_arrays(self):
        raise NotImplementedError()


#sliding window
class AdjustedDailyWindow(SlidingWindow):
    """
    Wrapper around an AdjustedArrayWindow which supports monotonically
    increasing (by datetime) requests for a sized window of data.

    Parameters
    ----------
    window : AdjustedArrayWindow
       Window of pricing data with prefetched values beyond the current
       simulation dt.
    cal_start : int
       Index in the overall calendar at which the window starts.
    """
    FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume'])

    def __init__(self,
                trading_calendar,
                bar_reader,
                equity_adjustment_reader):
        self._trading_calendar = trading_calendar
        self._adjustment = HistoryCompatibleAdjustments(
                                    equity_adjustment_reader,
                                    bar_reader)

    @property
    def frequency(self):
        return 'daily'

    def _array(self, dts, assets, field):
        _reader =  self._adjustment._reader
        raw = _reader.load_raw_arrays(
            [field],
            dts[0],
            dts[-1],
            assets,
        )
        return raw

    def _window_arrays(self,edate,window,assets,field):
        """基于固定的fields才需要adjust"""
        #获取时间区间
        session = self._trading_calendar.sessions_in_range(edate,window)
        # 获取原始数据
        raw_arrays = self._array(session,assets, field)
        #需要调整的
        adjusted_fields = set(field) & self.FIELDS
        if adjusted_fields:
            #调整系数
            adjustments = self._adjustment.calculate_adjustments_in_sessions(edate,window,assets)
            #计算调整数据
            adjust_arrays = {}
            for asset in assets:
                sid = asset.sid
                qfq = adjustments[sid]
                raw = raw_arrays[sid]
                try:
                    qfq = qfq.reindex(session)
                    qfq.fillna(method = 'bfill',inplace = True)
                    qfq.fillna(1.0,inplace = True)
                    raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
                except Exception as e:
                    print(e,asset)
                adjust_arrays[sid] = raw
        else:
            adjust_arrays = raw_arrays
        return adjust_arrays


#cache value dt
class CachedObject(object):
    """
    A simple struct for maintaining a cached object with an expiration date.

    Parameters
    ----------
    value : object
        The object to cache.
    expires : datetime-like
        Expiration date of `value`. The cache is considered invalid for dates
        **strictly greater** than `expires`.
    """
    def __init__(self, value, expires):
        self._value = value
        self._expires = expires

    @classmethod
    def expired(cls):
        """Construct a CachedObject that's expired at any time.
        """
        return cls(ExpiredCachedObject, expires=AlwaysExpired)

    def unwrap(self, dts):
        """
        Get the cached value.

        Returns
        -------
        value : object
            The cached value.

        Raises
        ------
        Expired
            Raised when `dt` is greater than self.expires.
        """
        # expires = self._expires
        # if expires is AlwaysExpired or expires < dt:
        #     raise Expired(self._expires)
        expires = self._expires
        if not set(dts).issubset(set(expires)):
            raise Expiried(self._expired)
        return self._value

    def _unsafe_get_value(self):
        """You almost certainly shouldn't use this."""
        return self._value


class ExpiriedCache(object):
    """
    A cache of multiple CachedObjects, which returns the wrapped the value
    or raises and deletes the CachedObject if the value has expired.

    Parameters
    ----------
    cache : dict-like, optional
        An instance of a dict-like object which needs to support at least:
        `__del__`, `__getitem__`, `__setitem__`
        If `None`, than a dict is used as a default.

    cleanup : callable, optional
        A method that takes a single argument, a cached object, and is called
        upon expiry of the cached object, prior to deleting the object. If not
        provided, defaults to a no-op.

    """
    def __init__(self, cache=None, cleanup=lambda value_to_clean: None):
        if cache is not None:
            self._cache = cache
        else:
            self._cache = {}

        self.cleanup = cleanup

    def get(self, key, dt):
        """Get the value of a cached object.

        Parameters
        ----------
        key : any
            The key to lookup.
        dt : datetime
            The time of the lookup.

        Returns
        -------
        result : any
            The value for ``key``.

        Raises
        ------
        KeyError
            Raised if the key is not in the cache or the value for the key
            has expired.
        """
        try:
            return self._cache[key].unwrap(dt)
        except Expired:
            self.cleanup(self._cache[key]._unsafe_get_value())
            del self._cache[key]
            raise KeyError(key)

    def set(self, key, value, expiration_dt):
        """Adds a new key value pair to the cache.

        Parameters
        ----------
        key : any
            The key to use for the pair.
        value : any
            The value to store under the name ``key``.
        expiration_dt : datetime
            When should this mapping expire? The cache is considered invalid
            for dates **strictly greater** than ``expiration_dt``.
        """
        self._cache[key] = CachedObject(value, expiration_dt)


from collections import defaultdict


class HistoryLoader(ABC):

    @property
    def _frequency(self):
        raise NotImplementedError()

    @abstractmethod
    def _compute_slice_window(self,data,dt,window):
        raise NotImplementedError

    def _ensure_adjust_windows(self, edate, window,field,assets):
        """
        Ensure that there is a Float64Multiply window for each asset that can
        provide data for the given parameters.
        If the corresponding window for the (assets, len(dts), field) does not
        exist, then create a new one.
        If a corresponding window does exist for (assets, len(dts), field), but
        can not provide data for the current dts range, then create a new
        one and replace the expired window.

        Parameters
        ----------
        assets : iterable of Assets
            The assets in the window
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str or list
            The OHLCV field for which to retrieve data.
        is_perspective_after : bool
            see: `PricingHistoryLoader.history`

        Returns
        -------
        out : list of Float64Window with sufficient data so that each asset's
        window can provide `get` for the index corresponding with the last
        value in `dts`
        """
        dts = self._calendar.sessions_in_range(edate,window)
        #设立参数
        asset_windows = {}
        needed_assets = []
        #默认获取OHLCV数据
        for asset in assets:
            try:
                _window = self._window_blocks[asset].get(
                    field, dts)
            except KeyError:
                needed_assets.append(asset)
            else:
                _slice = self._compute_slice_window(_window,dts)
                asset_windows[asset] = _slice

        if needed_assets:
            for i, asset in enumerate(needed_assets):
                sliding_window = self.adjust_window._window_arrays(
                        edate,
                        window,
                        asset,
                        field
                            )
                asset_windows[asset] = sliding_window
                #设置ExpiriedCache
                self._window_blocks[asset].set(
                    field,
                    sliding_window)
        return [asset_windows[asset] for asset in assets]

    def history(self,assets,field,dts,window = 0):
        """
        A window of pricing data with adjustments applied assuming that the
        end of the window is the day before the current simulation time.
        default fields --- OHLCV

        Parameters
        ----------
        assets : iterable of Assets
            The assets in the window.
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str or list
            The OHLCV field for which to retrieve data.
        window : int
            The length of window
        Returns
        -------
        out : np.ndarray with shape(len(days between start, end), len(assets))
        """
        if window:
            block_arrays = self._ensure_sliding_windows(
                                            dts,
                                            window,
                                            field,
                                            assets
                                             )
        else:
            block_arrays = self.adjust_window._array([dts,dts],assets,field)
        return block_arrays


class HistoryDailyLoader(HistoryLoader):
    """
        生成调整后的序列
        优化 --- 缓存
    """

    def __init__(self,
                _daily_reader,
                equity_adjustment_reader,
                trading_calendar,
    ):
        self.adjust_window = AdjustedDailyWindow(trading_calendar,
                                            _daily_reader,
                                            equity_adjustment_reader)
        self._trading_calendar = trading_calendar
        self._window_blocks = defaultdict(ExpiriedCache())

    @property
    def _frequency(self):
        return 'daily'

    @property
    def _calendar(self):
        return self._trading_calendar

    def _compute_slice_window(self,_window,sessions):
        _slice_window = _window.reindex(sessions)
        return _slice_window


class AdjustedMinuteWindow(SlidingWindow):
    """
    Wrapper around an AdjustedArrayWindow which supports monotonically
    increasing (by datetime) requests for a sized window of data.

    Parameters
    ----------
    window : AdjustedArrayWindow
       Window of pricing data with prefetched values beyond the current
       simulation dt.
    cal_start : int
       Index in the overall calendar at which the window starts.
    """
    FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume'])

    def __init__(self,
                _minute_reader,
                equity_adjustment_reader,
                trading_calendar):
        self._trading_calendar = trading_calendar
        self._adjustment = HistoryCompatibleAdjustments(
                                    equity_adjustment_reader,
                                    _minute_reader)

    @property
    def frequency(self):
        return 'minute'

    def _array(self, dts, assets, field):
        _reader = self._adjustment._reader
        raw = _reader.load_raw_arrays(
                                    dts[0],
                                    dts[-1],
                                    assets,
                                    field
        )
        return raw

    def _window_arrays(self,edate,window,assets,field):
        """基于固定的fields才需要adjust"""
        #获取时间区间
        session = self._trading_calendar.sessions_in_range(edate,window)
        # 获取原始数据
        raw_arrays = self._array(session,assets,field)
        #需要调整的
        adjusted_fields = set(field) & self.FIELDS
        if adjusted_fields:
            #调整系数
            adjustments = self._adjustment.calculate_adjustments_in_sessions(edate,window,assets)
            #计算调整数据
            adjust_arrays = {}
            for asset in assets:
                sid = asset.sid
                #调整index
                qfq = adjustments[sid]
                qfq.index = [ pd.Timestamp(inx).timestamp() + 15 * 60 * 60  for inx in qfq.index]
                raw = raw_arrays[sid]
                try:
                    qfq = qfq.reindex(session)
                    qfq.fillna(method = 'bfill',inplace = True)
                    qfq.fillna(1.0,inplace = True)
                    raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
                except Exception as e:
                    print(e,asset)
                adjust_arrays[sid] = raw
        else:
            adjust_arrays = raw_arrays
        return adjust_arrays



class HistoryMinuteLoader(HistoryLoader):

    def __init__(self,
                _minute_reader,
                 equity_adjustment_reader,
                trading_calendar):
        self.adjust_minute_window = AdjustedMinuteWindow(
                                            trading_calendar,
                                            _minute_reader,
                                            equity_adjustment_reader)
        self._trading_calendar = trading_calendar
        self._cache = {}

    def _compute_slice_window(self,raw,dts):
        # 时间区间为子集，需要过滤
        dts_minutes = self._calendar.minutes_in_window(dts)
        _slice_window = raw.reindex(dts_minutes)
        return _slice_window


from toolz import keyfilter, valmap


class DataPortal(object):
    """Interface to all of the data that a simulation needs.

    This is used by the simulation runner to answer questions about the data,
    like getting the prices of assets on a given day or to service history
    calls.

    Parameters
    ----------
    asset_finder : zipline.assets.assets.AssetFinder
        The AssetFinder instance used to resolve assets.
    trading_calendar: zipline.utils.calendar.exchange_calendar.TradingCalendar
        The calendar instance used to provide minute->session information.
    first_trading_day : pd.Timestamp
        The first trading day for the simulation.
    equity_daily_reader : BcolzDailyBarReader, optional
        The daily bar reader for equities. This will be used to service
        daily data backtests or daily history calls in a minute backetest.
        If a daily bar reader is not provided but a minute bar reader is,
        the minutes will be rolled up to serve the daily requests.
    equity_minute_reader : BcolzMinuteBarReader, optional
        The minute bar reader for equities. This will be used to service
        minute data backtests or minute history calls. This can be used
        to serve daily calls if no daily bar reader is provided.
    adjustment_reader : SQLiteAdjustmentWriter, optional
        The adjustment reader. This is used to apply splits, dividends, and
        other adjustment data to the raw data from the readers.
    """

    OHLCV_FIELDS = frozenset(["open", "high", "low", "close", "volume"])

    Asset_Type = frozenset(['symbol','etf','bond'])

    def __init__(self,
                asset_finder,
                trading_calendar,
                first_trading_day,
                _dispatch_session_reader,
                _dispatch_minute_reader,
                adjustment_reader,
                 ):
        self.asset_finder = asset_finder

        self.trading_calendar = trading_calendar

        self._first_trading_day = first_trading_day

        self._adjustment_reader = adjustment_reader

        self._pricing_readers = {
            'minute': _dispatch_minute_reader,
            'daily': _dispatch_session_reader,
        }

        _history_daily_loader = HistoryDailyLoader(
            _dispatch_minute_reader,
            self._adjustment_reader,
            trading_calendar,
        )
        _history_minute_loader = HistoryMinuteLoader(
            _dispatch_session_reader,
            self._adjustment_reader,
            trading_calendar,

        )
        self._history_loader = {
            'daily':_history_daily_loader,
            'minute':_history_minute_loader,
        }

        # Get the first trading minute
        self._first_trading_minute, _ = (
            self.trading_calendar.open_and_close_for_session(
                [self._first_trading_day]
            )
            if self._first_trading_day is not None else (None, None)
        )

        # Store the locs of the first day and first minute
        self._first_trading_day_loc = (
            self.trading_calendar.all_sessions.get_loc(self._first_trading_day)
            if self._first_trading_day is not None else None
        )
        self._extra_source = None

    @property
    def adjustment_reader(self):
        return self._adjustment_reader

    def _get_pricing_reader(self, data_frequency):
        return self._pricing_readers[data_frequency]

    def get_fetcher_assets(self, _typ):
        """
        Returns a list of assets for the current date, as defined by the
        fetcher data.

        Returns
        -------
        list: a list of Asset objects.
        """
        # return a list of assets for the current date, as defined by the
        # fetcher source
        assets = self.asset_finder.lookup_assets(_typ)
        return assets

    def get_dividends(self, sids, trading_days):
        """
        splits --- divdends

        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_days: pd.DatetimeIndex
            The trading range.

        Returns
        -------
        list: A list of objects with all relevant attributes populated.
        All timestamp fields are converted to pd.Timestamps.
        """
        extra = set(sids) - set(self._divdends_cache)
        if extra:
            for sid in extra:
                divdends = self.adjustment_reader.load_splits_for_sid(sid)
                self._divdends_cache[sid] = divdends
        #
        from toolz import keyfilter,valmap
        cache  = keyfilter(lambda x : x in sids,self._splits_cache)
        out = valmap(lambda x : x[x['pay_date'].isin(trading_days)] if x else x ,cache)
        return out

    def get_stock_rights(self, sids, trading_days):
        """
        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_days: pd.DatetimeIndex
            The trading range.

        Returns
        -------
        list: A list of objects with all relevant attributes populated.
        All timestamp fields are converted to pd.Timestamps.
        """
        extra = set(sids) - set(self._rights_cache)
        if extra:
            for sid in extra:
                rights = self.adjustment_reader.load_splits_for_sid(sid)
                self._rights_cache[sid] = rights
        #
        cache  = keyfilter(lambda x : x in sids,self._rights_cache)
        out = valmap(lambda x : x[x['pay_date'].isin(trading_days)] if x else x ,cache)
        return out

    def _get_history_sliding_window(self,assets,
                                    end_dt,
                                    fields,
                                    bar_count,
                                    frequency
                                   ):
        """
        Internal method that returns a dataframe containing history bars
        of minute frequency for the given sids.
        """
        history = self._history_daily_loader[frequency]
        history_arrays = history.history(assets,fields,end_dt,window = bar_count)
        return history_arrays

    def get_history_window(self,
                           assets,
                           end_dt,
                           bar_count,
                           field,
                           data_frequency):
        """
        Public API method that returns a dataframe containing the requested
        history window.  Data is fully adjusted.

        Parameters
        ----------
        assets : list of zipline.data.Asset objects
            The assets whose data is desired.

        bar_count: int
            The number of bars desired.

        frequency: string
            "1d" or "1m"

        field: string
            The desired field of the asset.

        data_frequency: string
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars.

        ex: boolean
            raw or adjusted array

        Returns
        -------
        A dataframe containing the requested data.
        """
        if field not in self.OHLCV_FIELDS:
            raise ValueError("Invalid field: {0}".format(field))

        if bar_count < 1:
            raise ValueError(
                "bar_count must be >= 1, but got {}".format(bar_count)
            )
        history_window_arrays = self._get_history_sliding_window(assets,
                                                             end_dt,
                                                             field,
                                                             bar_count,
                                                             data_frequency)
        return history_window_arrays

    def get_window_data(self,
                         assets,
                         dt,
                         field,
                         days_in_window,
                         frequency):
        """
        Internal method that gets a window of adjusted daily data for a sid
        and specified date range.  Used to support the history API method for
        daily bars.

        Parameters
        ----------
        asset : Asset
            The asset whose data is desired.

        dt: pandas.Timestamp
            The end of the desired window of data.

        field: string
            The specific field to return.  "open", "high", "close_price", etc.

        bar_count: int
            The number of days of data to return.

        data_frequency : minute or daily

        Returns
        -------
        A numpy array with requested values.  Any missing slots filled with
        nan.
        """
        _reader = self._get_pricing_readers[frequency]
        window_array = _reader.load_raw_arrays(dt, days_in_window, field, assets)
        return window_array

    def _get_resized_minutes(self,dts,sids,field,_ticker):
        """
            Internal method that resample
            api : groups.keys() , get_group()
        """
        _minutes_reader = self._pricing_readers['minute']
        resamples = _minutes_reader.reindex_minutes_ticker(dts,sids,field,_ticker)
        return resamples

    def get_resample_minutes(self,sessions,sids,field,frequency):
        reindex_minutes = self._get_resized_minutes(sessions,sids,field,frequency)
        return reindex_minutes

    def get_current(self,sid):
        """
            return current live tickers data
        """
        _url = 'http://push2.eastmoney.com/api/qt/stock/trends2/get?fields1=f1&' \
               'fields2=f51,f52,f53,f54,f55,f56,f57,f58&iscr=0&secid={}'
        # 处理数据
        req_sid = '0.' + sid if sid.startswith('6') else '1.' + sid
        req_url = _url.format(req_sid)
        obj = _parse_url(req_url, bs=False)
        d = json.loads(obj)
        raw_array = [item.split(',') for item in d['data']['trends']]
        minutes = pd.DataFrame(raw_array,
                          columns=['ticker', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'avg'])
        return minutes

    def handle_extra_source(self,source_df):
        """
            Internal method that determines if this asset/field combination
            represents a fetcher value or a regular OHLCVP lookup.
            Extra sources always have a sid column.
            We expand the given data (by forward filling) to the full range of
            the simulation dates, so that lookup is fast during simulation.
        """
        raise NotImplementedError()


class BarData:
    """
    Provides methods for accessing minutely and daily price/volume data from
    Algorithm API functions.

    Also provides utility methods to determine if an asset is alive, and if it
    has recent trade data.

    An instance of this object is passed as ``data`` to
    :func:`~zipline.api.handle_data` and
    :func:`~zipline.api.before_trading_start`.

    Parameters
    ----------
    data_portal : DataPortal
        Provider for bar pricing data.
    simulation_dt_func : callable
        Function which returns the current simulation time.
        This is usually bound to a method of TradingSimulation.
    data_frequency : {'minute', 'daily'}
        The frequency of the bar data; i.e. whether the data is
        daily or minute bars
    restrictions : zipline.finance.asset_restrictions.Restrictions
        Object that combines and returns restricted list information from
        multiple sources
    universe_func : callable, optional
        Function which returns the current 'universe'.  This is for
        backwards compatibility with older API concepts.
    """

    def __init__(self, data_portal, data_frequency,
                 trading_calendar, restrictions, universe_func=None):
        self.data_portal = data_portal
        self.data_frequency = data_frequency
        self._universe_func = universe_func
        self._trading_calendar = trading_calendar
        self._is_restricted = restrictions.is_restricted

    def get_current_ticker(self,assets,fields):
        """
        Returns the "current" value of the given fields for the given assets
        at the current simulation time.
        :param assets: asset_type
        :param fields: OHLCTV
        :return: dict asset -> ticker
        intended to return current ticker
        """
        cur = {}
        for asset in assets:
            ticker = self.data_portal.get_current(asset)
            cur[asset] = ticker.loc[:,fields]
        return cur

    def history(self, assets, end_dt,bar_count, fields,frequency):
        """
        Returns a trailing window of length ``bar_count`` containing data for
        the given assets, fields, and frequency.

        Returned data is adjusted for splits, dividends, and mergers as of the
        current simulation time.

        The semantics for missing data are identical to the ones described in
        the notes for :meth:`current`.

        Parameters
        ----------
        assets: zipline.assets.Asset or iterable of zipline.assets.Asset
            The asset(s) for which data is requested.
        fields: string or iterable of string.
            Requested data field(s). Valid field names are: "price",
            "last_traded", "open", "high", "low", "close", and "volume".
        bar_count: int
            Number of data observations requested.
        frequency: str
            String indicating whether to load daily or minutely data
            observations. Pass '1m' for minutely data, '1d' for daily data.

        Returns
        -------
        history : pd.Series or pd.DataFrame or pd.Panel
            See notes below.

        Notes
        ------
        returned panel has:
        items: fields
        major axis: dt
        minor axis: assets
        return pd.Panel(df_dict)
        """
        sliding_window = self.data_portal.get_history_window(assets,
                                                             end_dt,
                                                             bar_count,
                                                             fields,
                                                             frequency)
        return sliding_window

    def window_data(self,assets,end_dt,bar_count,fields,frequency):
        window_array = self.data_portal.get_window_data(assets,
                                                        end_dt,
                                                        bar_count,
                                                        fields,
                                                        frequency)
        return window_array

    def _get_equity_price_view(self, asset):
        """
        Returns a DataPortalSidView for the given asset.  Used to support the
        data[sid(N)] public API.  Not needed if DataPortal is used standalone.

        Parameters
        ----------
        asset : Asset
            Asset that is being queried.

        Returns
        -------
        SidView : Accessor into the given asset's data.
        """
        try:
            self._warn_deprecated("`data[sid(N)]` is deprecated. Use "
                            "`data.current`.")
            view = self._views[asset]
        except KeyError:
            try:
                asset = self.data_portal.asset_finder.retrieve_asset(asset)
            except ValueError:
                # assume fetcher
                pass
            view = self._views[asset] = self._create_sid_view(asset)

        return view

    def _create_sid_view(self, asset):
        return SidView(
            asset,
            self.data_portal,
            self.simulation_dt_func,
            self.data_frequency
        )