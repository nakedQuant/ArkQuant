# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from contextlib import ExitStack
import sqlalchemy as sa
from gateWay.assets.asset_spider import AssetSpider
from gateWay.assets.asset_db_schema import (
    metadata,
    ASSET_DB_VERSION,
    asset_db_table_names,
    equity_supplementary,
    convertible_supplementary,
    asset_router,
    version_info
)

SQLITE_MAX_VARIABLE_NUMBER = 999

_rename_router_cols = frozenset(['sid',
                                 'asset_name',
                                 'asset_type',
                                 'exchange',
                                 'first_traded',
                                 'last_traded'])

_rename_equity_cols = {
    '代码': 'sid',
    '港股': 'dual',
    '上市市场': 'exchange',
    '上市日期': 'first_traded',
    '发行价格': 'initial_price',
    '主承销商': 'broker',
    '公司名称': 'company_symbol',
    '成立日期': 'establish_date',
    '注册资本': 'register_capital',
    '组织形式': 'organization',
    '邮政编码': 'district',
    '经营范围': 'business_scope',
    '公司简介': 'brief',
    '证券简称更名历史': 'history_name',
    '注册地址': 'register_area',
    '办公地址': 'office_area'
}

_rename_convertible_cols = {
    'BONDCODE': 'sid',
    'SWAPSCODE': 'swap_code',
    'SNAME': 'asset_name',
    'LISTDATE': 'first_traded',
    'maturity_dt': 'last_traded'
}

_rename_fund_cols = {
    '基金代码': 'sid',
    '基金简称': 'asset_name',
    '类型': 'asset_type'
}

spider = AssetSpider()


def check_version_info(conn, version_table, expected_version):
    """
    Checks for a version value in the version table.

    Parameters
    ----------
    conn : sa.Connection
        The connection to use to perform the check.
    version_table : sa.Table
        The version table of the asset database
    expected_version : int
        The expected version of the asset database

    Raises
    ------
    AssetDBVersionError
        If the version is in the table and not equal to ASSET_DB_VERSION.
    """

    # Read the version out of the table
    version_from_table = conn.execute(
        sa.select((version_table.c.version,)),
    ).scalar()

    # A db without a version is considered v0
    if version_from_table is None:
        version_from_table = 0

    # Raise an error if the versions do not match
    if (version_from_table != expected_version):
        # raise AssetDBVersionError(db_version=version_from_table,
        #                           expected_version=expected_version)
        raise ValueError('db_version != version_from_table')


def write_version_info(conn, version_table, version_value):
    """
    Inserts the version value in to the version table.

    Parameters
    ----------
    conn : sa.Connection
        The connection to use to execute the insert.
    version_table : sa.Table
        The version table of the asset database
    version_value : int
        The version to write in to the database

    """
    conn.execute(sa.insert(version_table, values={'version': version_value}))


class AssetWriter(object):
    """Class used to write data to an assets db.

    Parameters
    ----------
    engine : Engine or str
        An SQLAlchemy engine or path to a SQL database.
    """
    DEFAULT_CHUNK_SIZE = SQLITE_MAX_VARIABLE_NUMBER

    # @preprocess(engine=coerce_string_to_eng(require_exists=False))
    def __init__(self, engine):
        self.engine = engine
        self._asset_spider = spider

    def _all_tables_present(self, txn):
        """
        Checks if any tables are present in the current assets database.

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

    def init_db(self, txn=None):
        """Connect to database and create tables.

        Parameters
        ----------
        txn : sa.engine.Connection, optional
            The transaction to execute in. If this is not provided, a new
            transaction will be started with the engine provided.

        Returns
        -------
        metadata : sa.MetaData
            The metadata that describes the new assets db.
        """
        with ExitStack() as stack:
            if txn is None:
                txn = stack.enter_context(self.engine.begin())

            tables_already_exist = self._all_tables_present(txn)

            # Create the SQL tables if they do not already exist.
            metadata.create_all(txn, checkfirst=True)

            if tables_already_exist:
                check_version_info(txn, version_info, ASSET_DB_VERSION)
            else:
                write_version_info(txn, version_info, ASSET_DB_VERSION)

    def _read_writer(self,
                     equity_supplementary_mappings = None,
                     convertible_supplementary_mappings = None,
                     fund_mappings = None,
                     chunk_size=DEFAULT_CHUNK_SIZE):
        with self.engine.begin() as conn:
            # Create SQL tables if they do not exist.
            self.init_db(conn)

            if equity_supplementary_mappings is not None:
                self._write_df_to_table(
                    equity_supplementary,
                    equity_supplementary_mappings,
                    conn,
                    chunk_size,
                )

            if convertible_supplementary_mappings is not None:
                self._write_df_to_table(
                    convertible_supplementary,
                    convertible_supplementary_mappings,
                    conn,
                    chunk_size,
                )

            if fund_mappings is not None:
                self._write_assets(
                    fund_mappings,
                    conn,
                    chunk_size
                )

    def _write_df_to_table(self, tbl, df, txn, chunk_size, include=True):
        df = df.copy()
        if include:
            self._write_assets(df,txn, chunk_size)
        # asset supplement to db
        supplement = df.loc[:, ]
        df.to_sql(
            tbl.name,
            txn.connection,
            # index=True,
            # index_label=first(tbl.primary_key.columns).name,
            index=False,
            if_exists='append',
            chunksize=chunk_size,
        )

    def _write_assets(self,
                      mapping_data,
                      txn,
                      chunk_size):
        symbols_mapping = mapping_data.loc[:, _rename_router_cols]

        symbols_mapping.to_sql(
            asset_router.name,
            txn.connection,
            if_exists='append',
            index=False,
            chunksize=chunk_size
        )

    def _generate_output_dataframe(self, data_set):
        """
        Generates an output dataframe from the given subset of user-provided
        data, the given column names, and the given default values.

        Parameters
        ----------
        data_set :
            A DataFrame, usually from an AssetData object,
            that contains the user's input metadata for the asset type being
            processed
        default_cols : dict
            A dict where the keys are the names of the columns of the desired
            output DataFrame and the values are a function from dataframe and
            column name to the default values to insert in the DataFrame if no user
            data is provided

        Returns
        -------
        DataFrame
            A DataFrame containing all user-provided metadata, and default values
            wherever user-provided metadata was missing
        """
        data_set.equities.rename(columns=_rename_equity_cols, inplace=True)
        data_set.convertibles.rename(columns=_rename_convertible_cols, inplace =True)
        data_set.funds.rename(columns=_rename_fund_cols, inplace =True)
        return data_set


    def write(self, chunk_size=DEFAULT_CHUNK_SIZE):
        """Write asset metadata to a sqlite database.
        """
        assetData = self._asset_spider.load_data()
        reformat_data = self._generate_output_dataframe(assetData)

        self._real_write(
            equity_supplementary_mappings=reformat_data.equities,
            convertible_supplementary_mappings=reformat_data.converibles,
            fund_mappings=reformat_data.funds,
            chunk_size=chunk_size,
        )

    def write_direct(self,
                     root_symbols=None,
                     equity_supplementary_mappings=None,
                     convertible_supplement_mappings=None,
                     chunk_size=DEFAULT_CHUNK_SIZE):
        """Write asset metadata to a sqlite database in the format that it is
        stored in the assets db.
        """
        raise NotImplementedError('not allowed to write metadata into db directly')
