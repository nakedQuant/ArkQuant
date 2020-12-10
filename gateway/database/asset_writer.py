# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import sqlalchemy as sa, numpy as np, pandas as pd
from contextlib import ExitStack
from sqlalchemy import create_engine
from gateway.database.db_schema import asset_db_table_names
from gateway.database.db_writer import init_writer
from gateway.database import (
    engine,
    metadata,
)

db = init_writer()

# 确保过滤还未上市的标的 , 由于退市的股票数据不全只能取关键的上市日字段
EquityNullFields = ['sid', 'first_traded']

ConvertibleNullFields = ['sid', 'swap_code', 'put_price', 'convert_price', 'convert_dt']

# rename cols --- 入库
_rename_router_cols = frozenset(['sid',
                                 'asset_name',
                                 'asset_type',
                                 'first_traded',
                                 'last_traded',
                                 # 'status',
                                 'exchange'])

_rename_equity_cols = {
    '代码': 'sid',
    '港股': 'dual_sid',
    '上市市场': 'exchange',
    '上市日期': 'first_traded',
    '发行价格': 'initial_price',
    '主承销商': 'broker',
    # '公司名称': 'company_symbol',
    '公司名称': 'asset_name',
    '成立日期': 'establish_date',
    '注册资本': 'register_capital',
    '组织形式': 'organization',
    '邮政编码': 'district',
    '经营范围': 'business_scope',
    '公司简介': 'brief',
    '证券简称更名历史': 'history_name',
    '注册地址': 'register_area',
    '办公地址': 'office_area',
    # 'delist_date': 'last_traded'
}

_rename_convertible_cols = {
    'BONDCODE': 'sid',
    'SWAPSCODE': 'swap_code',
    'SNAME': 'asset_name',
    'LISTDATE': 'first_traded',
    # 'maturity_dt': 'last_traded'
}

# fund -- first_traded
_rename_fund_cols = {
    '基金代码': 'sid',
    '基金简称': 'asset_name',
    '类型': 'asset_type'
}


class AssetWriter(object):
    """Class used to write data to an asset db.

    Parameters
    ----------
    engine : Engine or str
        An SQLAlchemy engine or path to a SQL database.
    """
    # @preprocess(engine=coerce_string_to_eng(require_exists=False))
    def __init__(self, engine_path):
        self.engine = create_engine(engine_path) if engine_path else engine
        self._init_db()

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
        # for table_name in asset_db_table_names:
        #     if txn.dialect.has_table(conn, table_name):
        #         return True
        # return False
        present = np.all([txn.dialect.has_table(conn, t)
                         for t in asset_db_table_names])
        return present

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
            if not tables_already_exist:
                # Create the SQL tables if they do not already exist.
                metadata.create_all(txn, checkfirst=True)

    @staticmethod
    def _write_assets(frame):
        # symbols_mapping = frame.loc[:, _rename_router_cols]
        renamed_frame = frame.reindex(columns=_rename_router_cols, fill_value='')
        db.writer('asset_router', renamed_frame)

    def _write_df_to_table(self, tbl, df, include=True):
        df = df.copy()
        if include:
            self._write_assets(df)
        db.writer(tbl, df)

    def _real_write(self,
                    equity_frame=None,
                    convertible_frame=None,
                    fund_frame=None):
        if equity_frame is not None:
            self._write_df_to_table(
                'equity_basics',
                equity_frame,
            )
        print('equity successfully')

        if convertible_frame is not None:
            self._write_df_to_table(
                'convertible_basics',
                convertible_frame,
            )
        print('convertible successfully')

        if fund_frame is not None:
            self._write_assets(
                fund_frame,
            )
        print('fund_name successfully')

    @staticmethod
    def _reformat_frame(data_set):
        """
        Generates an output dataframe from the given subset of user-provided
        data, the given column names, and the given default values.

        --- fillna('')  --- rename cols --- dropna by null field --- add fields(e.g. asset_type exchange)

        Parameters
        ----------
        data_set :
            A DataFrame, usually from an AssetData object,
            that contains the user's input metadata for the asset type being
            processed

        Returns
        -------
        DataFrame
            A DataFrame containing all user-provided metadata, and default values
            wherever user-provided metadata was missing

        Empty DataFrame --- rename Empty DataFrame
        """
        # equity replace null -- Nan remove equities which are not on market
        if not data_set.equities.empty:
            # # replace null -- Nan
            data_set.equities.replace(to_replace='null', value=pd.NA, inplace=True)
            # replace(to_value, value) to_replace is a scalar, list or tuple and value is None. ,method = pad(default)
            # data_set.equities['上市日期'].replace('--', None, inplace=True)
            data_set.equities['上市日期'].replace('--', pd.NA, inplace=True)
            data_set.equities['发行价格'].replace('', 0.00, inplace=True)
            # fillna --- Fill NA/NaN values using the specified method.
            data_set.equities['发行价格'].fillna(0.00, inplace=True)
            data_set.equities.rename(columns=_rename_equity_cols, inplace=True)
            data_set.equities.dropna(axis=0, how='any', subset=EquityNullFields, inplace=True)
            if not data_set.equities.empty:
                data_set.equities['initial_price'] = data_set.equities['initial_price'].astype(np.float)
                data_set.equities.loc[:, 'asset_type'] = 'equity'
                data_set.equities.fillna('', inplace=True)

        if not data_set.convertibles.empty:
            # convertible
            data_set.convertibles.replace(to_replace='-', value=pd.NA, inplace=True)
            data_set.convertibles.rename(columns=_rename_convertible_cols, inplace=True)
            # 保留已经上市而且转股的可转债
            data_set.convertibles.dropna(axis=0, how='any', subset=ConvertibleNullFields, inplace=True)
            # add and transform maturity date format
            if not data_set.convertibles.empty:
                data_set.convertibles.loc[:, 'last_traded'] = data_set.convertibles['maturity_dt'].apply(
                                                            lambda x: x.replace('-', ''))
                data_set.convertibles.loc[:, 'exchange'] = data_set.convertibles['sid'].apply(
                                                            lambda x: '上海证券交易所' if x.startswith('11') else '深圳证券交易所')
                data_set.convertibles.loc[:, 'asset_type'] = 'convertible'
        if not data_set.funds.empty:
            # fund
            data_set.funds['基金简称'] = data_set.funds['基金简称'].apply(lambda x: x[:-5])
            data_set.funds.rename(columns=_rename_fund_cols, inplace=True)
            data_set.funds.loc[:, 'exchange'] = data_set.funds['sid'].apply(
                                                    lambda x: '上海证券交易所' if x.startswith('5') else '深圳证券交易所')
            print('process fund', data_set.funds.head())
        return data_set

    def write(self, asset_data):
        """Write asset metadata to a sqlite database.
        """
        frame = self._reformat_frame(asset_data)

        self._real_write(
            equity_frame=frame.equities,
            convertible_frame=frame.convertibles,
            fund_frame=frame.funds,
        )
        # update last_traded via equity_status transfer to asset_router
        db.update()

    def write_direct(self,
                     equity_frame,
                     convertible_frame,
                     fund_frame):
        """Write asset metadata to a sqlite database in the format that it is
        stored in the asset db.
        """
        # raise NotImplementedError('not allowed to write metadata into db directly')
        self._real_write(
            equity_frame=equity_frame,
            convertible_frame=convertible_frame,
            fund_frame=fund_frame,
        )


__all__ = ['AssetWriter']
