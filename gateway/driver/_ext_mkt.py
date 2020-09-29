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

__all__ = ['MarketValue']

OWNERSHIP_TYPE = {'general': np.double,
                  'float': np.double,
                  'close': np.double}

RENAME_COLUMNS = {'general': 'mkv',
                  'float': 'mkv_cap',
                  'strict': 'mkv_strict'}


class MarketValue:

    def __init__(self, daily=True):
        self._init = daily

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

    def _retrieve_array(self, sid):
        edate = datetime.datetime.now().strftime('%Y-%m-%d')
        sdate = '1990-01-01' if self._init else edate
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
        for sid in set(ownership):
            print('sid', sid)
            owner = ownership[sid]
            owner.sort_values(by='general', ascending=False, inplace=True)
            owner.drop_duplicates(subset='date', keep='first', inplace=True)
            owner.set_index('date', inplace=True)
            close = self._retrieve_array(sid)
            print('close', close)
            if close.empty:
                print('%s close is empty' % sid)
            else:
                re_owner = owner.reindex(index=close.index)
                re_owner.sort_index(inplace=True)
                re_owner.fillna(method='bfill', inplace=True)
                re_owner.fillna(method='ffill', inplace=True)
                # 当每日更新的时候re_owner(reindex 为None) --- 需要通过最近的日期来填充
                re_owner = re_owner.fillna({'float': owner['float'][0], 'general': owner['general'][0]})
                print('adjust owner', re_owner)
                mcap = re_owner.apply(lambda x: x * close)
                mcap.loc[:, 'trade_dt'] = mcap.index
                mcap.loc[:, 'sid'] = sid
                mcap.loc[:, 'strict'] = mcap['general'] - mcap['float']
                mcap.rename(columns=RENAME_COLUMNS, inplace=True)
                print('mcap', mcap)
                db.writer('m_cap', mcap)


# if __name__ == '__main__':
#
#     m = MarketValue(False)
#     m.calculate_mcap()
