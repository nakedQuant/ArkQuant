#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019
@author: python
"""
import pandas as pd, numpy as np, sqlalchemy as sa, datetime
from gateway.database import engine, metadata
from gateway.database.db_writer import db
from gateway.driver.tools import unpack_df_to_component_dict

OWNERSHIP_TYPE = {'general': np.double,
                  'float': np.double,
                  'close': np.double}

RENAME_COLUMNS = {'general': 'mkv',
                  'float': 'mkv_cap',
                  'strict': 'mkv_strict'}


class MarketValue:

    def __init__(self, daily=True):
        self.mode = daily

    @staticmethod
    def _adjust_frame_type(df):
        for col, col_type in OWNERSHIP_TYPE.items():
            try:
                df[col] = df[col].astype(col_type)
            except KeyError:
                pass
            except TypeError:
                raise TypeError('%s cannot mutate into %s' % (col, col_type))
        return df

    def _retrieve_ownership(self):
        tbl = metadata.tables['ownership']
        sql = sa.select([tbl.c.sid, tbl.c.ex_date, tbl.c.general, tbl.c.float])
        # sql = sa.select([tbl.c.sid, tbl.c.ex_date, tbl.c.general, tbl.c.float]).where(tbl.c.sid == '000002')
        rp = engine.execute(sql)
        frame = pd.DataFrame([[r.sid, r.ex_date, r.general, r.float] for r in rp.fetchall()],
                             columns=['sid', 'date', 'general', 'float'])
        frame.set_index('sid', inplace=True)
        frame.replace('--', 0.0, inplace=True)
        frame = self._adjust_frame_type(frame)
        unpack_frame = unpack_df_to_component_dict(frame)
        return unpack_frame

    # def _retrieve_array(self, sid):
    #     edate = datetime.datetime.now().strftime('%Y-%m-%d')
    #     sdate = edate if self.mode else '1990-01-01'
    #     tbl = metadata.tables['equity_price']
    #     sql = sa.select([tbl.c.trade_dt, tbl.c.sid, tbl.c.close]).\
    #          where(tbl.c.trade_dt.between(sdate, edate))
    #     rp = engine.execute(sql)
    #     frame = pd.DataFrame([[r.trade_dt, r.sid, r.close] for r in rp.fetchall()],
    #                          columns=['date', 'sid', 'close'])
    #     frame = self._adjust_frame_type(frame)
    #     frame.set_index('sid', inplace=True)
    #     unpack_frame = unpack_df_to_component_dict(frame, 'date')
    #     return unpack_frame

    def _retrieve_array(self, sid):
        edate = datetime.datetime.now().strftime('%Y-%m-%d')
        sdate = edate if self.mode else '1990-01-01'
        tbl = metadata.tables['equity_price']
        sql = sa.select([tbl.c.trade_dt, tbl.c.close]).\
            where(sa.and_(tbl.c.trade_dt.between(sdate, edate), tbl.c.sid == sid))
        rp = engine.execute(sql)
        frame = pd.DataFrame([[r.trade_dt, r.close] for r in rp.fetchall()],
                             columns=['date', 'close'])
        frame = self._adjust_frame_type(frame)
        frame.set_index('date', inplace=True)
        return frame.iloc[:, 0]

    def calculate_mcap(self):
        """由于存在一个变动时点出现多条记录，保留最大total_assets的记录,先按照最大股本降序，保留第一个记录"""
        ownership = self._retrieve_ownership()
        print('ownership', ownership)
        for sid in set(ownership):
            owner = ownership[sid]
            owner.sort_values(by='general', ascending=False, inplace=True)
            owner.drop_duplicates(subset='date', keep='first', inplace=True)
            owner.set_index('date', inplace=True)
            close = self._retrieve_array(sid)
            print('close', close)
            if close.empty:
                print('%s close is empty' % sid)
            else:
                owner = owner.reindex(index=close.index)
                owner.fillna(method='ffill', inplace=True)
                owner.fillna(method='bfill', inplace=True)
                mcap = owner.apply(lambda x: x * close)
                mcap.loc[:, 'trade_dt'] = mcap.index
                mcap.loc[:, 'sid'] = sid
                mcap.loc[:, 'strict'] = mcap['general'] - mcap['float']
                mcap.rename(columns=RENAME_COLUMNS, inplace=True)
                print('mcap', mcap)
                db.writer('m_cap', mcap)


if __name__ == '__main__':

    m = MarketValue(False)
    m.calculate_mcap()
