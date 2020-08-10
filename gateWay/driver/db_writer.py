# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd,sqlalchemy as sa
from sqlalchemy import inspect
from contextlib import ExitStack
from weakref import WeakValueDictionary
from .db_schema import engine, metadata, asset_db_table_names


class DBWriter(object):

    _cache = WeakValueDictionary()

    def __new__(cls):
        try:
            return cls._cache[engine]
        except KeyError:
            txn = engine.connect()
            instance = object().__new__(cls)._init_db(txn)
            cls._cache[engine] = instance
            return cls._cache[engine]

    def _init_db(self, txn=None):
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
            if txn is None:
                txn = stack.enter_context(self.engine.begin())

            tables_already_exist = self._all_tables_present(txn)

            # Create the SQL tables if they do not already exist.
            if not tables_already_exist:
                metadata.create_all(txn, checkfirst=True)
            #将table
            metadata.reflect(only=asset_db_table_names)
            for table_name in asset_db_table_names:
                setattr(self, table_name, metadata.tables[table_name])

    def _all_tables_present(self, txn):
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
        for table_name in asset_db_table_names:
            if txn.dialect.has_table(conn, table_name):
                return True
        return False

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

    def _writer_direct(self, con, tbl, data):
        ins = metadata.tables[tbl].insert()
        if isinstance(data, pd.DataFrame):
            formatted = list(data.T.to_dict().values())
        elif isinstance(data, pd.Series):
            formatted = data.to_dict()
        else:
            raise ValueError('must be dataframe or series')
        con.execute(ins, formatted)

    def writer(self, tbl, df, direct = True):
        with open('db_schema.py','r') as f:
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


db = DBWriter()

__all__ = [db]
