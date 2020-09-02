# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd,sqlalchemy as sa, numpy as np
from sqlalchemy import inspect, create_engine, update
from contextlib import ExitStack
from weakref import WeakValueDictionary
from gateway.database.db_schema import asset_db_table_names
from gateway.database import metadata, engine_path, SQLITE_MAX_VARIABLE_NUMBER
from gateway.driver.client import tsclient


__all__ = ['db']


class DBWriter(object):

    _cache = WeakValueDictionary()

    def __new__(cls, root_path, level="READ COMMITTED"):
        try:
            return cls._cache[root_path]
        except KeyError:
            engine = create_engine(root_path, isolation_level=level)
            instance = object().__new__(cls)
            instance._init_db(engine)
            cls._cache[engine] = instance
            instance.engine = engine
            return cls._cache[engine]

    def __enter__(self):
        return self

    @staticmethod
    def _all_tables_present(txn):
        """
        Checks if any tables are present in the current asset database.

        Parameters
        ----------
        txn : Transaction
            The open transaction to check in.

        Returns
        -------
        has_tables : bool
            True if any tables are present, otherwise False.
        """
        conn = txn.connect()
        present = np.all([txn.dialect.has_table(conn, t)
                         for t in asset_db_table_names])
        return present

    def _init_db(self, engine):
        """Connect to database and create tables.

        Parameters
        ----------
        txn : sa.engine.Connection, optional
            The transaction to execute in. If this is not provided, a new
            transaction will be started with the engine provided.

        Returns
        -------
        metadata : sa.MetaData
            The metadata that describes the new asset db.
        """
        with ExitStack() as stack:
            # if txn is None:
            txn = stack.enter_context(engine.connect())
            tables_already_exist = self._all_tables_present(txn)
            # Create the SQL tables if they do not already exist.
            if not tables_already_exist:
                metadata.create_all(txn, checkfirst=True)
            # 将table
            metadata.reflect(only=asset_db_table_names)
            for table_name in asset_db_table_names:
                setattr(self, table_name, metadata.tables[table_name])

    def _write_df_to_table(self, conn, tbl, frame, chunksize=SQLITE_MAX_VARIABLE_NUMBER):
        inspection = inspect(self.engine)
        expected_cols = [item['name'] for item in inspection.get_columns(tbl)]
        if frozenset(frame.columns) != frozenset(expected_cols):
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    set(expected_cols),
                    frame.columns.tolist(),
                )
            )
        frame.to_sql(
            tbl,
            conn,
            flavor='mysql',
            index=False,
            if_exists='append',
            chunksize=chunksize,
        )

    @staticmethod
    def _writer_direct(conn, tbl, data):
        ins = metadata.tables[tbl].insert()
        if isinstance(data, pd.DataFrame):
            formatted = list(data.T.to_dict().values())
        elif isinstance(data, pd.Series):
            formatted = data.to_dict()
        else:
            raise ValueError('must be frame or series')
        conn.execute(ins, formatted)

    def writer(self, tbl, df, direct=True):
        # with open('db_schema.py', 'r') as f:
        #     string_obj = f.read()
        # exec(string_obj)
        # 每个线程单独 --- cursor conn
        if not df.empty:
            with self.engine.begin() as conn:
                # Create SQL tables if they do not exist.
                # self.metadata.create_all(bind=engine)
                if direct:
                    self._writer_direct(conn, tbl, df)
                else:
                    self._write_df_to_table(conn, tbl, df)

    def _update(self):
        with self.engine.connect() as conn:
            tbl_a = metadata.tables['asset_router']
            tbl_b = metadata.tables['equity_status']
            ins = update(tbl_a).where(tbl_a.c.sid == tbl_b.c.sid)
            ins = ins.values(last_traded=tbl_b.c.last_traded)
            print('ins', ins)
            conn.execute(ins)

    def update(self):
        # append status 由于吸收合并代码可能会消失但是主体继续上市存在 e.g. T00018
        # 对于暂停上市 delist_date 为None(作为一种长期停盘的情况来考虑，由于我们不能存在后视误差不清楚是否能重新上市）
        # 基于算法发出信号操作暂时上市的标的, 为了避免前视误差，过滤筛选距离已经退市标的而不是暂停上市  e.g. 5个交易日的股票；
        # 对于暂停上市的股票可能还存在重新上市的可能性也可能存在退市的可能性 --- None ,状态不断的更新
        self.clear('equity_status')
        df = tsclient.to_ts_status()
        self.writer('equity_status', df)
        self._update()

    @classmethod
    def reset(cls):
        metadata.drop_all()

    def close(self):
        self.conn.close()

    def clear(self, tbl=None):
        if tbl:
            self.engine.execute(metadata.tables[tbl].delete())
        else:
            for tbl in metadata.tables:
                self.engine.execute(tbl.delete())

    def __exit__(self, * exc_info):
        self.close()


db = DBWriter(engine_path)


if __name__ == '__main__':

    db.update()
