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
import sqlalchemy as sa
import pandas as pd
from functools import partial
from sqlalchemy import MetaData
from toolz import groupby
from gateway.database import engine
from gateway.driver.tools import unpack_df_to_component_dict


class BarReader(ABC):

    @property
    def data_frequency(self):
        return None

    @property
    def metadata(self):
        return MetaData(bind=engine)

    @abstractmethod
    def get_spot_value(self, asset, dt, fields):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        asset : Asset
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        fields : string or list
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
    def load_raw_arrays(self, sessions, assets, columns):
        """
        Parameters
        ----------
        sessions: list --- element:str
           Beginning of the window range.
        assets : list of int
           The asset identifiers in the window.
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        raise NotImplementedError()


class AssetSessionReader(BarReader):
    """
    Reader for raw pricing data from mysql.
    return different asset types : etf bond symbol
    """
    def __init__(self):
        self.engine = engine

    @property
    def data_frequency(self):
        return 'daily'

    def get_equity_pct(self, dt):
        tbl = self.metadata['equity_price']
        sql = sa.select([tbl.c.sid, sa.cast(tbl.c.pct, sa.Numeric(10, 2)).label('pct')])\
            .where(tbl.c.trade_dt == dt)
        rp = self.engine.execute(sql)
        data = pd.DataFrame(rp.fetchall(), columns=['sid', 'pct'])
        data.set_index('sid', inplace=True)
        return data

    def get_spot_value(self, dt, asset, fields):
        """
            retrieve asset data  on dt
        """
        table_name = '%s_price' % asset.asset_type
        tbl = self.metadata[table_name]
        orm = sa.select([
                    sa.cast(tbl.c.open, sa.Numeric(10, 2)).label('open'),
                    sa.cast(tbl.c.close, sa.Numeric(12, 2)).label('close'),
                    sa.cast(tbl.c.high, sa.Numeric(10, 2)).label('high'),
                    sa.cast(tbl.c.low, sa.Numeric(10, 3)).label('low'),
                    sa.cast(tbl.c.volume, sa.Numeric(15, 0)).label('volume'),
                    sa.cast(tbl.c.amount, sa.Numeric(15, 2)).label('amount')])\
            .where(sa.and_(tbl.c.trade_dt == dt, tbl.c.sid == asset.sid))
        rp = self.engine.execute(orm)
        arrays = [[r.trade_dt, r.code, r.open, r.close, r.high, r.low, r.volume] for r in
                  rp.fetchall()]
        kline = pd.DataFrame(arrays, columns=['open', 'close', 'high',
                                              'low', 'volume', 'amount'])
        return kline.loc[:, fields]

    def _retrieve_assets(self, table, sids, fields, start_date, end_date):
        """
            retrieve specific categroy asset
        """
        tbl = self.metadata['%s_price' % table]
        orm = sa.select([tbl.c.trade_dt, tbl.c.sid,
                         sa.cast(tbl.c.open,sa.Numeric(10, 2)).label('open'),
                         sa.cast(tbl.c.high, sa.Numeric(10, 2)).label('high'),
                         sa.cast(tbl.c.low, sa.Numeric(10, 3)).label('low'),
                         sa.cast(tbl.c.close, sa.Numeric(12, 2)).label('close'),
                         sa.cast(tbl.c.volume, sa.Numeric(15, 0)).label('volume'),
                         sa.cast(tbl.c.amount, sa.Numeric(15, 2)).label('amount')]). \
            where(tbl.c.trade_dt.between(start_date, end_date))
        rp = self.engine.execute(orm)
        arrays = [[r.trade_dt, r.code, r.open, r.close, r.high, r.low, r.volume] for r in
                  rp.fetchall()]
        raw = pd.DataFrame(arrays, columns=['trade_dt', 'code', 'open','high',
                                           'low', 'close', 'volume', 'amount'])
        raw.set_index('code', inplace=True)
        # 基于code
        _slice = raw.loc[sids]
        # 基于fields 获取数据
        kline = _slice.loc[:, fields]
        unpack_kline = unpack_df_to_component_dict(kline)
        return unpack_kline

    def load_raw_arrays(self, session_labels, assets, columns):
        start_date, end_date = session_labels
        columns = set(columns + ['trade_dt'])
        func = partial(self._retrieve_assets(fields=columns,
                                             start_date=start_date,
                                             end_date=end_date)
                              )
        sid_groups = groupby(lambda x: x.asset_type, assets)
        #获取数据
        batch_arrays = {}
        for name, sids in sid_groups.items():
            raw = func(table=name, sids=sids)
            batch_arrays.update(raw)
        return batch_arrays
