# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd,sqlalchemy as sa, numpy as np
from sqlalchemy import inspect, create_engine
from contextlib import ExitStack
from weakref import WeakValueDictionary
from gateway.database.db_schema import asset_db_table_names
from gateway.database import metadata, engine_path

__all__ = ['db']


class DBWriter(object):

    _cache = WeakValueDictionary()

    def __new__(cls, root_path):
        try:
            return cls._cache[root_path]
        except KeyError:
            engine = create_engine(root_path)
            instance = object().__new__(cls)
            instance._init_db(engine)
            cls._cache[engine] = instance
            return cls._cache[engine]

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

    def __enter__(self):
        return self

    @staticmethod
    def set_isolation_level(conn, level="READ COMMITTED"):
        connection = conn.execution_options(
            isolation_level=level
        )
        return connection

    def _write_df_to_table(self, conn, tbl, frame, chunksize=5000):
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
            index=True,
            index_label=None,
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
        with open('db_schema.py', 'r') as f:
            string_obj = f.read()
        exec(string_obj)

        with self.engine.begin() as conn:
            # Create SQL tables if they do not exist.
            # self.metadata.create_all(bind=engine)
            con = self.set_isolation_level(conn)
            if direct:
                self._writer_direct(con, tbl, df)
            else:
                self._write_df_to_table(con, tbl, df)

    def reset(self, overwriter=False):
        if overwriter:
            metadata.drop_all()
        else:
            for tbl in metadata.tables:
                self.engine.execute(tbl.delete())

    def close(self):
        self.conn.close()

    def __exit__(self, * exc_info):
        self.close()


db = DBWriter(engine_path)
