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
import sqlalchemy as sa, pandas as pd, numpy as np
from functools import partial
from toolz import groupby, valmap
from gateway.database import engine, metadata
from gateway.driver.tools import unpack_df_to_component_dict
from gateway.asset.assets import Equity, Convertible, Fund

KLINE_COLUMNS_TYPE = {
            'open': np.double,
            'high': np.double,
            'low': np.double,
            'close': np.double,
            'volume': np.int,
            'amount': np.double,
            'pct': np.double
                    }


class BarReader(ABC):

    @property
    def data_frequency(self):
        return None

    @property
    def metadata(self):
        metadata.reflect(bind=engine)
        return metadata

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

    def get_mkv_value(self, sessions, assets, fields):
        tbl = self.metadata.tables['m_cap']
        sdate, edate = sessions
        mkv_dct = {}
        for asset in assets:
            orm = sa.select([tbl.c.trade_dt, tbl.c.sid, tbl.c.mkv, tbl.c.mkv_cap, tbl.c.mkv_strict]).\
                where(sa.and_(tbl.c.trade_dt.between(sdate, edate), tbl.c.sid == asset.sid))
            rp = self.engine.execute(orm)
            frame = pd.DataFrame([[r.trade_dt, r.sid, r.mkv, r.mkv_cap, r.mkv_strict] for r in rp.fetchall()],
                                 columns=['trade_dt', 'sid', 'mkv', 'mkv_cap', 'mkv_strict'])
            frame = frame.loc[:, fields] if fields else frame
            mkv_dct[asset.sid] = frame
        return mkv_dct

    def get_spot_value(self, dt, asset, fields):
        """
            retrieve asset data  on dt
        """
        table_name = '%s_price' % asset.asset_type if asset.asset_type in ['equity', 'convertible'] else 'fund_price'
        tbl = self.metadata.tables[table_name]
        if asset.asset_type == 'equity':
            orm = sa.select([
                        tbl.c.trade_dt,
                        sa.cast(tbl.c.open, sa.Numeric(10, 2)).label('open'),
                        sa.cast(tbl.c.close, sa.Numeric(12, 2)).label('close'),
                        sa.cast(tbl.c.high, sa.Numeric(10, 2)).label('high'),
                        sa.cast(tbl.c.low, sa.Numeric(10, 3)).label('low'),
                        sa.cast(tbl.c.volume, sa.Numeric(15, 0)).label('volume'),
                        sa.cast(tbl.c.amount, sa.Numeric(15, 2)).label('amount'),
                        sa.cast(tbl.c.pct, sa.Numeric(15, 2)).label('pct')])\
                .where(sa.and_(tbl.c.trade_dt == dt, tbl.c.sid == asset.sid))
            rp = self.engine.execute(orm)
            arrays = [[r.trade_dt, r.open, r.close, r.high, r.low,
                       r.volume, r.amount, r.pct] for r in rp.fetchall()]
            kline = pd.DataFrame(arrays, columns=['trade_dt', 'open', 'close', 'high',
                                                  'low', 'volume', 'amount', 'pct'])
        else:
            orm = sa.select([
                        tbl.c.trade_dt,
                        sa.cast(tbl.c.open, sa.Numeric(10, 2)).label('open'),
                        sa.cast(tbl.c.close, sa.Numeric(12, 2)).label('close'),
                        sa.cast(tbl.c.high, sa.Numeric(10, 2)).label('high'),
                        sa.cast(tbl.c.low, sa.Numeric(10, 3)).label('low'),
                        sa.cast(tbl.c.volume, sa.Numeric(15, 0)).label('volume'),
                        sa.cast(tbl.c.amount, sa.Numeric(15, 2)).label('amount')])\
                .where(sa.and_(tbl.c.trade_dt == dt, tbl.c.sid == asset.sid))
            rp = self.engine.execute(orm)
            arrays = [[r.trade_dt, r.open, r.close, r.high, r.low,
                       r.volume, r.amount] for r in rp.fetchall()]
            kline = pd.DataFrame(arrays, columns=['trade_dt', 'open', 'close', 'high',
                                                  'low', 'volume', 'amount'])
        if not kline.empty:
            frame = self._adjust_frame_type(kline)
            return frame.loc[0, fields]
        return kline

    def get_stack_value(self, tbl_name, sessions):
        """
            intend to calculate market index
        """
        tbl = self.metadata.tables['%s_price' % tbl_name]
        start_date, end_date = sessions
        orm = sa.select([
                    tbl.c.trade_dt, tbl.c.sid,
                    sa.cast(tbl.c.open, sa.Numeric(10, 2)).label('open'),
                    sa.cast(tbl.c.close, sa.Numeric(12, 2)).label('close'),
                    sa.cast(tbl.c.high, sa.Numeric(10, 2)).label('high'),
                    sa.cast(tbl.c.low, sa.Numeric(10, 3)).label('low'),
                    sa.cast(tbl.c.volume, sa.Numeric(15, 0)).label('volume'),
                    sa.cast(tbl.c.amount, sa.Numeric(15, 2)).label('amount')])\
            .where(tbl.c.trade_dt.between(start_date, end_date))
        rp = self.engine.execute(orm)
        arrays = [[r.trade_dt, r.sid, r.open, r.close, r.high, r.low, r.volume, r.amount] for r in
                  rp.fetchall()]
        kline = pd.DataFrame(arrays, columns=['trade_dt', 'sid', 'open', 'close',
                                              'high', 'low', 'volume', 'amount'])
        kline.set_index('trade_dt', inplace=True)
        return kline

    @staticmethod
    def _adjust_frame_type(df):
        for col, col_type in KLINE_COLUMNS_TYPE.items():
            try:
                df[col] = df[col].astype(col_type)
            except KeyError:
                pass
            except TypeError:
                raise TypeError('%s cannot mutate into %s' % (col, col_type))
        return df

    def _retrieve_kline(self, table, sids, fields, start_date, end_date):
        """
            retrieve specific categroy asset
        """
        tbl = self.metadata.tables['%s_price' % table]
        orm = sa.select([tbl.c.trade_dt, tbl.c.sid,
                         sa.cast(tbl.c.open, sa.Numeric(10, 2)).label('open'),
                         sa.cast(tbl.c.high, sa.Numeric(10, 2)).label('high'),
                         sa.cast(tbl.c.low, sa.Numeric(10, 3)).label('low'),
                         sa.cast(tbl.c.close, sa.Numeric(12, 2)).label('close'),
                         sa.cast(tbl.c.volume, sa.Numeric(15, 0)).label('volume'),
                         sa.cast(tbl.c.amount, sa.Numeric(15, 2)).label('amount')]). \
            where(tbl.c.trade_dt.between(start_date, end_date))
        rp = self.engine.execute(orm)
        arrays = [[r.trade_dt, r.sid, r.open, r.high, r.low, r.close, r.volume, r.amount] for r in
                  rp.fetchall()]
        frame = pd.DataFrame(arrays, columns=['trade_dt', 'sid', 'open', 'high',
                                              'low', 'close', 'volume', 'amount'])
        frame.drop_duplicates(ignore_index=True, inplace=True)
        frame = frame[frame['sid'].isin(sids)]
        frame.set_index('sid', inplace=True)
        kline = self._adjust_frame_type(frame)
        unpack_kline = unpack_df_to_component_dict(kline.loc[:, fields], 'trade_dt')
        return unpack_kline

    def load_raw_arrays(self, session_labels, asset_objs, columns):
        start_date, end_date = session_labels
        columns = set(columns + ['trade_dt'])
        func = partial(self._retrieve_kline,
                       fields=columns,
                       start_date=start_date,
                       end_date=end_date)
        # adjust
        groups = groupby(lambda x: x.asset_type if x.asset_type in ['equity', 'convertible'] else 'fund', asset_objs)
        sid_groups = valmap(lambda x: [a.sid for a in x], groups)
        batch_arrays = {}
        for name, sids in sid_groups.items():
            data = func(table=name, sids=sids)
            batch_arrays.update(data)
        return batch_arrays


if __name__ == '__main__':

    reader = AssetSessionReader()
    asset = Equity('603612')
    sessions = ['2020-08-10', '2020-09-04']
    # pct = reader.get_equity_pctchange('2020-08-25')
    # print('equity pct', pct)
    # spot_value = reader.get_spot_value('2020-08-25', asset, ['open', 'high', 'low', 'close'])
    # print('spot_value', spot_value)
    # stack_value = reader.get_stack_value('equity', sessions)
    # print('stack value', stack_value)
    # his = reader.load_raw_arrays(sessions, [asset], ['open', 'high', 'low', 'close', 'volume', 'amount'])
    # print('his array', his)
    mkv = reader.get_mkv_value(sessions, [asset])
    print('mkv', mkv)
