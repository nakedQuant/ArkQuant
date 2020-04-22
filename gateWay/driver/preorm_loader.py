# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
from sqlalchemy import MetaData, create_engine,Table,Column,Integer,Numeric,String,Index
from sqlalchemy import delete

class DataLayer:
    """
        通用类型：
        sqlalchemy   python             sql
        BigInteger   int                BIGINT
        Boolean      bool               BOOLEAN
        Date         datetime.date      DATE
        DateTime     datetime.datetime  DATETIME
        Enum         str                ENUM
        Float        float              Float
        Integer      int                INTEGER
        Interval     datetime.timedelta INTERVAL
        LargeBinary  byte               BLOB
        Numeric      decimal.Decimal    NUMERIC
        Unicode      unicode            UNICODE
        Text         str                CLOB
        Time         datetime.time      DATETIME
        #autoincrement
        #元数据 Table对象目录，包含与引擎和连接的相关的信息,MetaData.table 对象目录
        #事务
        transaction = connection.begin()
        transaction.commit()
        transaction.rollback()
        #反射单个表
        price = Table('convertible',metadata,autoload = True,autoload_with = engine)
        print(price.columns.keys())
        #反射数据库
        metadata.reflect(bind = engine)
        print(metadata.tables.keys())
    """
    metadata = MetaData()

    engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/spider',
                           pool_size=50, max_overflow=100, pool_timeout=-1)

    kline = Table('asharePrice',metadata,
                    Column('trade_dt',String(10),primary_key= True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('turnover',String(20)),
                    Column('amplitude',String(20)),
                    Column('code',String(10),primary_key = True)
                    )

    description = Table('ashareInfo',metadata,
                    Column('代码',String(10),primary_key= True),
                    Column('上市日期',String(20)),
                    Column('发行价格',String(10)),
                    Column('证券简称更名历史',String(200)),
                    Column('主承销商',String(200)),
                    Column('注册地址',String(200))
                    )

    splits_divdend = Table('splitsDivdend',metadata,
                    Column('代码', String(20),primary_key= True),
                    Column('公告日期',String(20),primary_key= True),
                    Column('送股', String(20)),
                    Column('转增', String(20)),
                    Column('派息', String(20)),
                    Column('进度', String(20)),
                    Column('除权除息日', String(20)),
                    Column('股权登记日', String(20)),
                    Column('红股上市日',String(20))
                           )

    pairwise = Table('ashareIssue', metadata,
                     Column('代码', String(20), primary_key=True),
                     Column('公告日期', String(20),primary_key=True),
                     Column('配股方案', Numeric(5,4)),
                     Column('配股价格', Numeric(10,2)),
                     Column('除权日', String(20)),
                     Column('股权登记日', String(20)),
                     Column('配股上市日', String(20)),
                     )

    equity = Table('ashareEquity',metadata,
                      Column('代码', String(10)),
                      Column('变动日期',String(10)),
                      Column('公告日期',String(10)),
                      Column('总股本', String(20)),
                      Column('流通A股', String(20)),
                      Column('高管股', String(20)),
                      Column('限售A股', String(20)),
                      Column('流通B股', String(20)),
                      Column('限售B股', String(20)),
                      Column('流通H股', String(20)),
                      Column('国家股', String(20)),
                      Column('国有法人股', String(20)),
                      Column('境内发起人股', String(20)),
                      Column('募集法人股', String(20)),
                      Column('一般法人股', String(20)),
                      Column('战略投资者持股', String(20)),
                      Column('基金持股', String(20)),
                      Column('转配股', String(20)),
                      Column('内部职工股', String(20)),
                      Column('优先股', String(20)),
                      )

    mkt_value = Table('ashareValue',metadata,
                      Column('trade_dt', String(10),primary_key=True),
                      Column('code',String(10),primary_key= True),
                      Column('mkt',String(20)),
                      Column('cap', String(20)),
                      Column('strict', String(20)),
                      Column('hk', String(20)),
                      Column('foreign', String(20)),
                      )

    holding = Table('ashareHolding', metadata,
                    Column('代码', String(10)),
                    Column('股东', String(200)),
                    Column('方式', String(10)),
                    Column('变动股本', String(20)),
                    Column('占总流通比例', String(10)),
                    Column('途径', String(20)),
                    Column('总持仓', String(20)),
                    Column('占总股本比例', String(10)),
                    Column('总流通股', String(20)),
                    Column('变动开始日', String(10)),
                    Column('变动截止日', String(10)),
                    Column('公告日', String(10)),
                    )

    index = Table('ashareIndex',metadata,
                    Column('trade_dt',String(10),primary_key= True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('turnover',String(20)),
                    Column('amplitude',String(10)),
                    Column('code',String(10),primary_key= True),
                    Column('name',String(50)),
                    Index('index','trade_dt','code',unique = True)
                    )

    etf = Table('fundPrice',metadata,
                    Column('trade_dt',String(10),primary_key=True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('turnover',String(20)),
                    Column('code', String(10),primary_key = True),
                )

    convertible = Table('convertibleDesc',metadata,
                        Column('bond_id',String(10),nullable = True,primary_key= True),
                        Column('stock_id', String(10),primary_key= True),
                        Column('put_price', String(10)),
                        Column('convert_price', String(10)),
                        Column('convert_dt', String(10)),
                        Column('maturity_dt', String(10)),
                        Column('force_redeem_price', String(10)),
                        Column('put_convert_price', String(10)),
                        Column('guarantor', String(100)),
                        )

    bond_price = Table('convertiblePrice',metadata,
                    Column('trade_dt',String(10),primary_key = True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low', String(10)),
                    Column('volume', String(20)),
                    Column('turnover', String(20)),
                    Column('code',String(10),primary_key= True),
                    )

    hkline = Table('hkPrice',metadata,
                    Column('trade_dt',String(10),primary_key= True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('category',String(20)),
                    Column('h_code', String(10), primary_key=True),
                    Column('code',String(10),primary_key = True),
                    )

    margin = Table('marketMargin', metadata,
                   Column('交易日期', String(10), primary_key=True),
                   Column('融资余额', String(20)),
                   Column('融券余额', String(20)),
                   Column('融资融券总额', String(20)),
                   Column('融资融券差额', String(20)),
                   )

    status = Table('ashareStatus',metadata,
                    Column('code',String(10),primary_key = True),
                    Column('name', String(20)),
                    Column('delist_date',String(10)),
                    Column('status',String(10)),
                    )

    calendar = Table('ashareCalendar',metadata,
                    Column('trade_dt',String(10)),
                    Column('id', Integer,autoincrement= True,primary_key= True),
                     )

    metadata.create_all(bind = engine)

    @classmethod
    def initialize(cls):
        cls.metadata.drop_all(bind=cls.engine)
        cls.metadata.create_all(bind = cls.engine)

    @classmethod
    def drop_table(cls,table_name):
        tbl = getattr(cls,table_name)
        cls.metadata.drop_all(bind = cls.engine,tables = [tbl])
        delattr(cls,table_name)

    def __new__(cls,*args,**kwargs):
        if not hasattr(cls,'_instance'):
             DataLayer._instance = object.__new__(cls)
        return DataLayer._instance

    def db_init(self):
        """针对于每一个线程设立连接，多线程不能共用连接"""
        connection = self.engine.connect()
        return connection

    def enroll(self,tablename,data,conn):
        cls = getattr(self,tablename)
        ins = cls.insert()
        if len(data):
            if isinstance(data,pd.DataFrame):
                _to_dict = data.T.to_dict()
                formatted = list(_to_dict.values())
            elif isinstance(data,pd.Series):
                formatted = data.to_dict()
            elif isinstance(data,dict):
                formatted = data
            else:
                raise ValueError
            if not conn:
                conn = self.db_init()
            conn.execute(ins,formatted)

    def empty_table(self,tbl_name,conn):
        if not conn:
            conn = self.db_init()
        cls = getattr(self,tbl_name)
        ins = delete(cls)
        conn.execute(ins)


import sqlalchemy as sa


# Define a version number for the database generated by these writers
# Increment this version number any time a change is made to the schema of the
# assets database
# NOTE: When upgrading this remember to add a downgrade in:
# .asset_db_migrations
ASSET_DB_VERSION = 7

# A frozenset of the names of all tables in the assets db
# NOTE: When modifying this schema, update the ASSET_DB_VERSION value
asset_db_table_names = frozenset({
    'asset_router',
    'equities',
    'equity_symbol_mappings',
    'equity_supplementary_mappings',
    'futures_contracts',
    'exchanges',
    'futures_root_symbols',
    'version_info',
})

asset_db_table_names = frozenset({
    'asset_price',
    'equity_structure',
    'splits_divdend',
    'equity_issue',
    'holding_event',
    'fund_price',
    'bond_price',
    'benchmark_price',
    'margin',
})



metadata = sa.MetaData()

exchanges = sa.Table(
    'exchanges',
    metadata,
    sa.Column(
        'exchange',
        sa.Text,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column('canonical_name', sa.Text, nullable=False),
    sa.Column('country_code', sa.Text, nullable=False),
)

equities = sa.Table(
    'equities',
    metadata,
    sa.Column(
        'sid',
        sa.Integer,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column('asset_name', sa.Text),
    sa.Column('start_date', sa.Integer, default=0, nullable=False),
    sa.Column('end_date', sa.Integer, nullable=False),
    sa.Column('first_traded', sa.Integer),
    sa.Column('auto_close_date', sa.Integer),
    sa.Column('exchange', sa.Text, sa.ForeignKey(exchanges.c.exchange)),
)

equity_symbol_mappings = sa.Table(
    'equity_symbol_mappings',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.Integer,
        sa.ForeignKey(equities.c.sid),
        nullable=False,
        index=True,
    ),
    sa.Column(
        'symbol',
        sa.Text,
        nullable=False,
    ),
    sa.Column(
        'company_symbol',
        sa.Text,
        index=True,
    ),
    sa.Column(
        'share_class_symbol',
        sa.Text,
    ),
    sa.Column(
        'start_date',
        sa.Integer,
        nullable=False,
    ),
    sa.Column(
        'end_date',
        sa.Integer,
        nullable=False,
    ),
)

equity_supplementary_mappings = sa.Table(
    'equity_supplementary_mappings',
    metadata,
    sa.Column(
        'sid',
        sa.Integer,
        sa.ForeignKey(equities.c.sid),
        nullable=False,
        primary_key=True
    ),
    sa.Column('field', sa.Text, nullable=False, primary_key=True),
    sa.Column('start_date', sa.Integer, nullable=False, primary_key=True),
    sa.Column('end_date', sa.Integer, nullable=False),
    sa.Column('value', sa.Text, nullable=False),
)

futures_root_symbols = sa.Table(
    'futures_root_symbols',
    metadata,
    sa.Column(
        'root_symbol',
        sa.Text,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column('root_symbol_id', sa.Integer),
    sa.Column('sector', sa.Text),
    sa.Column('description', sa.Text),
    sa.Column(
        'exchange',
        sa.Text,
        sa.ForeignKey(exchanges.c.exchange),
    ),
)

futures_contracts = sa.Table(
    'futures_contracts',
    metadata,
    sa.Column(
        'sid',
        sa.Integer,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column('symbol', sa.Text, unique=True, index=True),
    sa.Column(
        'root_symbol',
        sa.Text,
        sa.ForeignKey(futures_root_symbols.c.root_symbol),
        index=True
    ),
    sa.Column('asset_name', sa.Text),
    sa.Column('start_date', sa.Integer, default=0, nullable=False),
    sa.Column('end_date', sa.Integer, nullable=False),
    sa.Column('first_traded', sa.Integer),
    sa.Column(
        'exchange',
        sa.Text,
        sa.ForeignKey(exchanges.c.exchange),
    ),
    sa.Column('notice_date', sa.Integer, nullable=False),
    sa.Column('expiration_date', sa.Integer, nullable=False),
    sa.Column('auto_close_date', sa.Integer, nullable=False),
    sa.Column('multiplier', sa.Float),
    sa.Column('tick_size', sa.Float),
)

asset_router = sa.Table(
    'asset_router',
    metadata,
    sa.Column(
        'sid',
        sa.Integer,
        unique=True,
        nullable=False,
        primary_key=True),
    sa.Column('asset_type', sa.Text),
)

version_info = sa.Table(
    'version_info',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'version',
        sa.Integer,
        unique=True,
        nullable=False,
    ),
    # This constraint ensures a single entry in this table
    sa.CheckConstraint('id <= 1'),
)

#
# Copyright 2015 Quantopian, Inc.
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
from collections import namedtuple
import re

from contextlib2 import ExitStack
import numpy as np
import pandas as pd
import sqlalchemy as sa
from toolz import first

from zipline.errors import AssetDBVersionError
from zipline.assets.asset_db_schema import (
    ASSET_DB_VERSION,
    asset_db_table_names,
    asset_router,
    equities as equities_table,
    equity_symbol_mappings,
    equity_supplementary_mappings as equity_supplementary_mappings_table,
    futures_contracts as futures_contracts_table,
    exchanges as exchanges_table,
    futures_root_symbols,
    metadata,
    version_info,
)

from zipline.utils.preprocess import preprocess
from zipline.utils.range import from_tuple, intersecting_ranges
from zipline.utils.sqlite_utils import coerce_string_to_eng

# Define a namedtuple for use with the load_data and _load_data methods
AssetData = namedtuple(
    'AssetData', (
        'equities',
        'equities_mappings',
        'futures',
        'exchanges',
        'root_symbols',
        'equity_supplementary_mappings',
    ),
)

SQLITE_MAX_VARIABLE_NUMBER = 999

symbol_columns = frozenset({
    'symbol',
    'company_symbol',
    'share_class_symbol',
})
mapping_columns = symbol_columns | {'start_date', 'end_date'}


_index_columns = {
    'equities': 'sid',
    'equity_supplementary_mappings': 'sid',
    'futures': 'sid',
    'exchanges': 'exchange',
    'root_symbols': 'root_symbol',
}


def _normalize_index_columns_in_place(equities,
                                      equity_supplementary_mappings,
                                      futures,
                                      exchanges,
                                      root_symbols):
    """
    Update dataframes in place to set indentifier columns as indices.

    For each input frame, if the frame has a column with the same name as its
    associated index column, set that column as the index.

    Otherwise, assume the index already contains identifiers.

    If frames are passed as None, they're ignored.
    """
    for frame, column_name in ((equities, 'sid'),
                               (equity_supplementary_mappings, 'sid'),
                               (futures, 'sid'),
                               (exchanges, 'exchange'),
                               (root_symbols, 'root_symbol')):
        if frame is not None and column_name in frame:
            frame.set_index(column_name, inplace=True)


def _default_none(df, column):
    return None


def _no_default(df, column):
    if not df.empty:
        raise ValueError('no default value for column %r' % column)


# Default values for the equities DataFrame
_equities_defaults = {
    'symbol': _default_none,
    'asset_name': _default_none,
    'start_date': lambda df, col: 0,
    'end_date': lambda df, col: np.iinfo(np.int64).max,
    'first_traded': _default_none,
    'auto_close_date': _default_none,
    # the full exchange name
    'exchange': _no_default,
}

# the defaults for ``equities`` in ``write_direct``
_direct_equities_defaults = _equities_defaults.copy()
del _direct_equities_defaults['symbol']

# Default values for the futures DataFrame
_futures_defaults = {
    'symbol': _default_none,
    'root_symbol': _default_none,
    'asset_name': _default_none,
    'start_date': lambda df, col: 0,
    'end_date': lambda df, col: np.iinfo(np.int64).max,
    'first_traded': _default_none,
    'exchange': _default_none,
    'notice_date': _default_none,
    'expiration_date': _default_none,
    'auto_close_date': _default_none,
    'tick_size': _default_none,
    'multiplier': lambda df, col: 1,
}

# Default values for the exchanges DataFrame
_exchanges_defaults = {
    'canonical_name': lambda df, col: df.index,
    'country_code': lambda df, col: '??',
}

# Default values for the root_symbols DataFrame
_root_symbols_defaults = {
    'sector': _default_none,
    'description': _default_none,
    'exchange': _default_none,
}

# Default values for the equity_supplementary_mappings DataFrame
_equity_supplementary_mappings_defaults = {
    'value': _default_none,
    'field': _default_none,
    'start_date': lambda df, col: 0,
    'end_date': lambda df, col: np.iinfo(np.int64).max,
}

# Default values for the equity_symbol_mappings DataFrame
_equity_symbol_mappings_defaults = {
    'sid': _no_default,
    'company_symbol': _default_none,
    'share_class_symbol': _default_none,
    'symbol': _default_none,
    'start_date': lambda df, col: 0,
    'end_date': lambda df, col: np.iinfo(np.int64).max,
}

# Fuzzy symbol delimiters that may break up a company symbol and share class
_delimited_symbol_delimiters_regex = re.compile(r'[./\-_]')
_delimited_symbol_default_triggers = frozenset({np.nan, None, ''})


def split_delimited_symbol(symbol):
    """
    Takes in a symbol that may be delimited and splits it in to a company
    symbol and share class symbol. Also returns the fuzzy symbol, which is the
    symbol without any fuzzy characters at all.

    Parameters
    ----------
    symbol : str
        The possibly-delimited symbol to be split

    Returns
    -------
    company_symbol : str
        The company part of the symbol.
    share_class_symbol : str
        The share class part of a symbol.
    """
    # return blank strings for any bad fuzzy symbols, like NaN or None
    if symbol in _delimited_symbol_default_triggers:
        return '', ''

    symbol = symbol.upper()

    split_list = re.split(
        pattern=_delimited_symbol_delimiters_regex,
        string=symbol,
        maxsplit=1,
    )

    # Break the list up in to its two components, the company symbol and the
    # share class symbol
    company_symbol = split_list[0]
    if len(split_list) > 1:
        share_class_symbol = split_list[1]
    else:
        share_class_symbol = ''

    return company_symbol, share_class_symbol


def _generate_output_dataframe(data_subset, defaults):
    """
    Generates an output dataframe from the given subset of user-provided
    data, the given column names, and the given default values.

    Parameters
    ----------
    data_subset : DataFrame
        A DataFrame, usually from an AssetData object,
        that contains the user's input metadata for the asset type being
        processed
    defaults : dict
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
    # The columns provided.
    cols = set(data_subset.columns)
    desired_cols = set(defaults)

    # Drop columns with unrecognised headers.
    data_subset.drop(cols - desired_cols,
                     axis=1,
                     inplace=True)

    # Get those columns which we need but
    # for which no data has been supplied.
    for col in desired_cols - cols:
        # write the default value for any missing columns
        data_subset[col] = defaults[col](data_subset, col)

    return data_subset


def _check_asset_group(group):
    row = group.sort_values('end_date').iloc[-1]
    row.start_date = group.start_date.min()
    row.end_date = group.end_date.max()
    row.drop(list(symbol_columns), inplace=True)
    return row


def _format_range(r):
    return (
        str(pd.Timestamp(r.start, unit='ns')),
        str(pd.Timestamp(r.stop, unit='ns')),
    )


def _check_symbol_mappings(df, exchanges, asset_exchange):
    """Check that there are no cases where multiple symbols resolve to the same
    asset at the same time in the same country.

    Parameters
    ----------
    df : pd.DataFrame
        The equity symbol mappings table.
    exchanges : pd.DataFrame
        The exchanges table.
    asset_exchange : pd.Series
        A series that maps sids to the exchange the asset is in.

    Raises
    ------
    ValueError
        Raised when there are ambiguous symbol mappings.
    """
    mappings = df.set_index('sid')[list(mapping_columns)].copy()
    mappings['country_code'] = exchanges['country_code'][
        asset_exchange.loc[df['sid']]
    ].values
    ambigious = {}

    def check_intersections(persymbol):
        intersections = list(intersecting_ranges(map(
            from_tuple,
            zip(persymbol.start_date, persymbol.end_date),
        )))
        if intersections:
            data = persymbol[
                ['start_date', 'end_date']
            ].astype('datetime64[ns]')
            # indent the dataframe string, also compute this early because
            # ``persymbol`` is a view and ``astype`` doesn't copy the index
            # correctly in pandas 0.22
            msg_component = '\n  '.join(str(data).splitlines())
            ambigious[persymbol.name] = intersections, msg_component

    mappings.groupby(['symbol', 'country_code']).apply(check_intersections)

    if ambigious:
        raise ValueError(
            'Ambiguous ownership for %d symbol%s, multiple assets held the'
            ' following symbols:\n%s' % (
                len(ambigious),
                '' if len(ambigious) == 1 else 's',
                '\n'.join(
                    '%s (%s):\n  intersections: %s\n  %s' % (
                        symbol,
                        country_code,
                        tuple(map(_format_range, intersections)),
                        cs,
                    )
                    for (symbol, country_code), (intersections, cs) in sorted(
                        ambigious.items(),
                        key=first,
                    ),
                ),
            )
        )


def _split_symbol_mappings(df, exchanges):
    """Split out the symbol: sid mappings from the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with multiple rows for each symbol: sid pair.
    exchanges : pd.DataFrame
        The exchanges table.

    Returns
    -------
    asset_info : pd.DataFrame
        The asset info with one row per asset.
    symbol_mappings : pd.DataFrame
        The dataframe of just symbol: sid mappings. The index will be
        the sid, then there will be three columns: symbol, start_date, and
        end_date.
    """
    mappings = df[list(mapping_columns)]
    with pd.option_context('mode.chained_assignment', None):
        mappings['sid'] = mappings.index
    mappings.reset_index(drop=True, inplace=True)

    # take the most recent sid->exchange mapping based on end date
    asset_exchange = df[
        ['exchange', 'end_date']
    ].sort_values('end_date').groupby(level=0)['exchange'].nth(-1)

    _check_symbol_mappings(mappings, exchanges, asset_exchange)
    return (
        df.groupby(level=0).apply(_check_asset_group),
        mappings,
    )


def _dt_to_epoch_ns(dt_series):
    """Convert a timeseries into an Int64Index of nanoseconds since the epoch.

    Parameters
    ----------
    dt_series : pd.Series
        The timeseries to convert.

    Returns
    -------
    idx : pd.Int64Index
        The index converted to nanoseconds since the epoch.
    """
    index = pd.to_datetime(dt_series.values)
    if index.tzinfo is None:
        index = index.tz_localize('UTC')
    else:
        index = index.tz_convert('UTC')
    return index.view(np.int64)


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
        raise AssetDBVersionError(db_version=version_from_table,
                                  expected_version=expected_version)


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


class _empty(object):
    columns = ()


class AssetDBWriter(object):
    """Class used to write data to an assets db.

    Parameters
    ----------
    engine : Engine or str
        An SQLAlchemy engine or path to a SQL database.
    """
    DEFAULT_CHUNK_SIZE = SQLITE_MAX_VARIABLE_NUMBER

    @preprocess(engine=coerce_string_to_eng(require_exists=False))
    def __init__(self, engine):
        self.engine = engine

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

    def _real_write(self,
                    equities,
                    equity_symbol_mappings,
                    equity_supplementary_mappings,
                    futures,
                    exchanges,
                    root_symbols,
                    chunk_size):
        with self.engine.begin() as conn:
            # Create SQL tables if they do not exist.
            self.init_db(conn)

            if exchanges is not None:
                self._write_df_to_table(
                    exchanges_table,
                    exchanges,
                    conn,
                    chunk_size,
                )

            if root_symbols is not None:
                self._write_df_to_table(
                    futures_root_symbols,
                    root_symbols,
                    conn,
                    chunk_size,
                )

            if equity_supplementary_mappings is not None:
                self._write_df_to_table(
                    equity_supplementary_mappings_table,
                    equity_supplementary_mappings,
                    conn,
                    chunk_size,
                )

            if futures is not None:
                self._write_assets(
                    'future',
                    futures,
                    conn,
                    chunk_size,
                )

            if equities is not None:
                self._write_assets(
                    'equity',
                    equities,
                    conn,
                    chunk_size,
                    mapping_data=equity_symbol_mappings,
                )

    def write_direct(self,
                     equities=None,
                     equity_symbol_mappings=None,
                     equity_supplementary_mappings=None,
                     futures=None,
                     exchanges=None,
                     root_symbols=None,
                     chunk_size=DEFAULT_CHUNK_SIZE):
        """Write asset metadata to a sqlite database in the format that it is
        stored in the assets db.

        Parameters
        ----------
        equities : pd.DataFrame, optional
            The equity metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this equity.
              asset_name : str
                  The full name for this asset.
              start_date : datetime
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              auto_close_date : datetime, optional
                  The date on which to close any positions in this asset.
              exchange : str
                  The exchange where this asset is traded.

            The index of this dataframe should contain the sids.
        futures : pd.DataFrame, optional
            The future contract metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this futures contract.
              root_symbol : str
                  The root symbol, or the symbol with the expiration stripped
                  out.
              asset_name : str
                  The full name for this asset.
              start_date : datetime, optional
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              exchange : str
                  The exchange where this asset is traded.
              notice_date : datetime
                  The date when the owner of the contract may be forced
                  to take physical delivery of the contract's asset.
              expiration_date : datetime
                  The date when the contract expires.
              auto_close_date : datetime
                  The date when the broker will automatically close any
                  positions in this contract.
              tick_size : float
                  The minimum price movement of the contract.
              multiplier: float
                  The amount of the underlying asset represented by this
                  contract.
        exchanges : pd.DataFrame, optional
            The exchanges where assets can be traded. The columns of this
            dataframe are:

              exchange : str
                  The full name of the exchange.
              canonical_name : str
                  The canonical name of the exchange.
              country_code : str
                  The ISO 3166 alpha-2 country code of the exchange.
        root_symbols : pd.DataFrame, optional
            The root symbols for the futures contracts. The columns for this
            dataframe are:

              root_symbol : str
                  The root symbol name.
              root_symbol_id : int
                  The unique id for this root symbol.
              sector : string, optional
                  The sector of this root symbol.
              description : string, optional
                  A short description of this root symbol.
              exchange : str
                  The exchange where this root symbol is traded.
        equity_supplementary_mappings : pd.DataFrame, optional
            Additional mappings from values of abitrary type to assets.
        chunk_size : int, optional
            The amount of rows to write to the SQLite table at once.
            This defaults to the default number of bind params in sqlite.
            If you have compiled sqlite3 with more bind or less params you may
            want to pass that value here.

        """
        if equities is not None:
            equities = _generate_output_dataframe(
                equities,
                _direct_equities_defaults,
            )
            if equity_symbol_mappings is None:
                raise ValueError(
                    'equities provided with no symbol mapping data',
                )

            equity_symbol_mappings = _generate_output_dataframe(
                equity_symbol_mappings,
                _equity_symbol_mappings_defaults,
            )
            _check_symbol_mappings(
                equity_symbol_mappings,
                exchanges,
                equities['exchange'],
            )

        if equity_supplementary_mappings is not None:
            equity_supplementary_mappings = _generate_output_dataframe(
                equity_supplementary_mappings,
                _equity_supplementary_mappings_defaults,
            )

        if futures is not None:
            futures = _generate_output_dataframe(_futures_defaults, futures)

        if exchanges is not None:
            exchanges = _generate_output_dataframe(
                exchanges.set_index('exchange'),
                _exchanges_defaults,
            )

        if root_symbols is not None:
            root_symbols = _generate_output_dataframe(
                root_symbols,
                _root_symbols_defaults,
            )

        # Set named identifier columns as indices, if provided.
        _normalize_index_columns_in_place(
            equities=equities,
            equity_supplementary_mappings=equity_supplementary_mappings,
            futures=futures,
            exchanges=exchanges,
            root_symbols=root_symbols,
        )

        self._real_write(
            equities=equities,
            equity_symbol_mappings=equity_symbol_mappings,
            equity_supplementary_mappings=equity_supplementary_mappings,
            futures=futures,
            exchanges=exchanges,
            root_symbols=root_symbols,
            chunk_size=chunk_size,
        )

    def write(self,
              equities=None,
              futures=None,
              exchanges=None,
              root_symbols=None,
              equity_supplementary_mappings=None,
              chunk_size=DEFAULT_CHUNK_SIZE):
        """Write asset metadata to a sqlite database.

        Parameters
        ----------
        equities : pd.DataFrame, optional
            The equity metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this equity.
              asset_name : str
                  The full name for this asset.
              start_date : datetime
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              auto_close_date : datetime, optional
                  The date on which to close any positions in this asset.
              exchange : str
                  The exchange where this asset is traded.

            The index of this dataframe should contain the sids.
        futures : pd.DataFrame, optional
            The future contract metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this futures contract.
              root_symbol : str
                  The root symbol, or the symbol with the expiration stripped
                  out.
              asset_name : str
                  The full name for this asset.
              start_date : datetime, optional
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              exchange : str
                  The exchange where this asset is traded.
              notice_date : datetime
                  The date when the owner of the contract may be forced
                  to take physical delivery of the contract's asset.
              expiration_date : datetime
                  The date when the contract expires.
              auto_close_date : datetime
                  The date when the broker will automatically close any
                  positions in this contract.
              tick_size : float
                  The minimum price movement of the contract.
              multiplier: float
                  The amount of the underlying asset represented by this
                  contract.
        exchanges : pd.DataFrame, optional
            The exchanges where assets can be traded. The columns of this
            dataframe are:

              exchange : str
                  The full name of the exchange.
              canonical_name : str
                  The canonical name of the exchange.
              country_code : str
                  The ISO 3166 alpha-2 country code of the exchange.
        root_symbols : pd.DataFrame, optional
            The root symbols for the futures contracts. The columns for this
            dataframe are:

              root_symbol : str
                  The root symbol name.
              root_symbol_id : int
                  The unique id for this root symbol.
              sector : string, optional
                  The sector of this root symbol.
              description : string, optional
                  A short description of this root symbol.
              exchange : str
                  The exchange where this root symbol is traded.
        equity_supplementary_mappings : pd.DataFrame, optional
            Additional mappings from values of abitrary type to assets.
        chunk_size : int, optional
            The amount of rows to write to the SQLite table at once.
            This defaults to the default number of bind params in sqlite.
            If you have compiled sqlite3 with more bind or less params you may
            want to pass that value here.

        See Also
        --------
        zipline.assets.asset_finder
        """
        if exchanges is None:
            exchange_names = [
                df['exchange']
                for df in (equities, futures, root_symbols)
                if df is not None
            ]
            if exchange_names:
                exchanges = pd.DataFrame({
                    'exchange': pd.concat(exchange_names).unique(),
                })

        data = self._load_data(
            equities if equities is not None else pd.DataFrame(),
            futures if futures is not None else pd.DataFrame(),
            exchanges if exchanges is not None else pd.DataFrame(),
            root_symbols if root_symbols is not None else pd.DataFrame(),
            (
                equity_supplementary_mappings
                if equity_supplementary_mappings is not None
                else pd.DataFrame()
            ),
        )
        self._real_write(
            equities=data.equities,
            equity_symbol_mappings=data.equities_mappings,
            equity_supplementary_mappings=data.equity_supplementary_mappings,
            futures=data.futures,
            root_symbols=data.root_symbols,
            exchanges=data.exchanges,
            chunk_size=chunk_size,
        )

    def _write_df_to_table(self, tbl, df, txn, chunk_size):
        df = df.copy()
        for column, dtype in df.dtypes.iteritems():
            if dtype.kind == 'M':
                df[column] = _dt_to_epoch_ns(df[column])

        df.to_sql(
            tbl.name,
            txn.connection,
            index=True,
            index_label=first(tbl.primary_key.columns).name,
            if_exists='append',
            chunksize=chunk_size,
        )

    def _write_assets(self,
                      asset_type,
                      assets,
                      txn,
                      chunk_size,
                      mapping_data=None):
        if asset_type == 'future':
            tbl = futures_contracts_table
            if mapping_data is not None:
                raise TypeError('no mapping data expected for futures')

        elif asset_type == 'equity':
            tbl = equities_table
            if mapping_data is None:
                raise TypeError('mapping data required for equities')
            # write the symbol mapping data.
            self._write_df_to_table(
                equity_symbol_mappings,
                mapping_data,
                txn,
                chunk_size,
            )

        else:
            raise ValueError(
                "asset_type must be in {'future', 'equity'}, got: %s" %
                asset_type,
            )

        self._write_df_to_table(tbl, assets, txn, chunk_size)

        pd.DataFrame({
            asset_router.c.sid.name: assets.index.values,
            asset_router.c.asset_type.name: asset_type,
        }).to_sql(
            asset_router.name,
            txn.connection,
            if_exists='append',
            index=False,
            chunksize=chunk_size
        )

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

    def _normalize_equities(self, equities, exchanges):
        # HACK: If 'company_name' is provided, map it to asset_name
        if ('company_name' in equities.columns and
                'asset_name' not in equities.columns):
            equities['asset_name'] = equities['company_name']

        # remap 'file_name' to 'symbol' if provided
        if 'file_name' in equities.columns:
            equities['symbol'] = equities['file_name']

        equities_output = _generate_output_dataframe(
            data_subset=equities,
            defaults=_equities_defaults,
        )

        # Split symbols to company_symbols and share_class_symbols
        tuple_series = equities_output['symbol'].apply(split_delimited_symbol)
        split_symbols = pd.DataFrame(
            tuple_series.tolist(),
            columns=['company_symbol', 'share_class_symbol'],
            index=tuple_series.index
        )
        equities_output = pd.concat((equities_output, split_symbols), axis=1)

        # Upper-case all symbol data
        for col in symbol_columns:
            equities_output[col] = equities_output[col].str.upper()

        # Convert date columns to UNIX Epoch integers (nanoseconds)
        for col in ('start_date',
                    'end_date',
                    'first_traded',
                    'auto_close_date'):
            equities_output[col] = _dt_to_epoch_ns(equities_output[col])

        return _split_symbol_mappings(equities_output, exchanges)

    def _normalize_futures(self, futures):
        futures_output = _generate_output_dataframe(
            data_subset=futures,
            defaults=_futures_defaults,
        )
        for col in ('symbol', 'root_symbol'):
            futures_output[col] = futures_output[col].str.upper()

        for col in ('start_date',
                    'end_date',
                    'first_traded',
                    'notice_date',
                    'expiration_date',
                    'auto_close_date'):
            futures_output[col] = _dt_to_epoch_ns(futures_output[col])

        return futures_output

    def _normalize_equity_supplementary_mappings(self, mappings):
        mappings_output = _generate_output_dataframe(
            data_subset=mappings,
            defaults=_equity_supplementary_mappings_defaults,
        )

        for col in ('start_date', 'end_date'):
            mappings_output[col] = _dt_to_epoch_ns(mappings_output[col])

        return mappings_output

    def _load_data(self,
                   equities,
                   futures,
                   exchanges,
                   root_symbols,
                   equity_supplementary_mappings):
        """
        Returns a standard set of pandas.DataFrames:
        equities, futures, exchanges, root_symbols
        """
        # Set named identifier columns as indices, if provided.
        _normalize_index_columns_in_place(
            equities=equities,
            equity_supplementary_mappings=equity_supplementary_mappings,
            futures=futures,
            exchanges=exchanges,
            root_symbols=root_symbols,
        )

        futures_output = self._normalize_futures(futures)

        equity_supplementary_mappings_output = (
            self._normalize_equity_supplementary_mappings(
                equity_supplementary_mappings,
            )
        )

        exchanges_output = _generate_output_dataframe(
            data_subset=exchanges,
            defaults=_exchanges_defaults,
        )

        equities_output, equities_mappings = self._normalize_equities(
            equities,
            exchanges_output,
        )

        root_symbols_output = _generate_output_dataframe(
            data_subset=root_symbols,
            defaults=_root_symbols_defaults,
        )

        return AssetData(
            equities=equities_output,
            equities_mappings=equities_mappings,
            futures=futures_output,
            exchanges=exchanges_output,
            root_symbols=root_symbols_output,
            equity_supplementary_mappings=equity_supplementary_mappings_output,
        )

from alembic.migration import MigrationContext
from alembic.operations import Operations
import sqlalchemy as sa
from toolz.curried import do, operator

from zipline.assets.asset_writer import write_version_info
from zipline.utils.compat import wraps
from zipline.errors import AssetDBImpossibleDowngrade
from zipline.utils.preprocess import preprocess
from zipline.utils.sqlite_utils import coerce_string_to_eng


def alter_columns(op, name, *columns, **kwargs):
    """Alter columns from a table.

    Parameters
    ----------
    name : str
        The name of the table.
    *columns
        The new columns to have.
    selection_string : str, optional
        The string to use in the selection. If not provided, it will select all
        of the new columns from the old table.

    Notes
    -----
    The columns are passed explicitly because this should only be used in a
    downgrade where ``zipline.assets.asset_db_schema`` could change.
    """
    selection_string = kwargs.pop('selection_string', None)
    if kwargs:
        raise TypeError(
            'alter_columns received extra arguments: %r' % sorted(kwargs),
        )
    if selection_string is None:
        selection_string = ', '.join(column.name for column in columns)

    tmp_name = '_alter_columns_' + name
    op.rename_table(name, tmp_name)

    for column in columns:
        # Clear any indices that already exist on this table, otherwise we will
        # fail to create the table because the indices will already be present.
        # When we create the table below, the indices that we want to preserve
        # will just get recreated.
        for table in name, tmp_name:
            try:
                op.drop_index('ix_%s_%s' % (table, column.name))
            except sa.exc.OperationalError:
                pass

    op.create_table(name, *columns)
    op.execute(
        'insert into %s select %s from %s' % (
            name,
            selection_string,
            tmp_name,
        ),
    )
    op.drop_table(tmp_name)


@preprocess(engine=coerce_string_to_eng(require_exists=True))
def downgrade(engine, desired_version):
    """Downgrades the assets db at the given engine to the desired version.

    Parameters
    ----------
    engine : Engine
        An SQLAlchemy engine to the assets database.
    desired_version : int
        The desired resulting version for the assets database.
    """

    # Check the version of the db at the engine
    with engine.begin() as conn:
        metadata = sa.MetaData(conn)
        metadata.reflect()
        version_info_table = metadata.tables['version_info']
        starting_version = sa.select((version_info_table.c.version,)).scalar()

        # Check for accidental upgrade
        if starting_version < desired_version:
            raise AssetDBImpossibleDowngrade(db_version=starting_version,
                                             desired_version=desired_version)

        # Check if the desired version is already the db version
        if starting_version == desired_version:
            # No downgrade needed
            return

        # Create alembic context
        ctx = MigrationContext.configure(conn)
        op = Operations(ctx)

        # Integer keys of downgrades to run
        # E.g.: [5, 4, 3, 2] would downgrade v6 to v2
        downgrade_keys = range(desired_version, starting_version)[::-1]

        # Disable foreign keys until all downgrades are complete
        _pragma_foreign_keys(conn, False)

        # Execute the downgrades in order
        for downgrade_key in downgrade_keys:
            _downgrade_methods[downgrade_key](op, conn, version_info_table)

        # Re-enable foreign keys
        _pragma_foreign_keys(conn, True)


def _pragma_foreign_keys(connection, on):
    """Sets the PRAGMA foreign_keys state of the SQLite database. Disabling
    the pragma allows for batch modification of tables with foreign keys.

    Parameters
    ----------
    connection : Connection
        A SQLAlchemy connection to the db
    on : bool
        If true, PRAGMA foreign_keys will be set to ON. Otherwise, the PRAGMA
        foreign_keys will be set to OFF.
    """
    connection.execute("PRAGMA foreign_keys=%s" % ("ON" if on else "OFF"))


# This dict contains references to downgrade methods that can be applied to an
# assets db. The resulting db's version is the key.
# e.g. The method at key '0' is the downgrade method from v1 to v0
_downgrade_methods = {}


def downgrades(src):
    """Decorator for marking that a method is a downgrade to a version to the
    previous version.

    Parameters
    ----------
    src : int
        The version this downgrades from.

    Returns
    -------
    decorator : callable[(callable) -> callable]
        The decorator to apply.
    """
    def _(f):
        destination = src - 1

        @do(operator.setitem(_downgrade_methods, destination))
        @wraps(f)
        def wrapper(op, conn, version_info_table):
            conn.execute(version_info_table.delete())  # clear the version
            f(op)
            write_version_info(conn, version_info_table, destination)

        return wrapper
    return _


@downgrades(1)
def _downgrade_v1(op):
    """
    Downgrade assets db by removing the 'tick_size' column and renaming the
    'multiplier' column.
    """
    # Drop indices before batch
    # This is to prevent index collision when creating the temp table
    op.drop_index('ix_futures_contracts_root_symbol')
    op.drop_index('ix_futures_contracts_symbol')

    # Execute batch op to allow column modification in SQLite
    with op.batch_alter_table('futures_contracts') as batch_op:

        # Rename 'multiplier'
        batch_op.alter_column(column_name='multiplier',
                              new_column_name='contract_multiplier')

        # Delete 'tick_size'
        batch_op.drop_column('tick_size')

    # Recreate indices after batch
    op.create_index('ix_futures_contracts_root_symbol',
                    table_name='futures_contracts',
                    columns=['root_symbol'])
    op.create_index('ix_futures_contracts_symbol',
                    table_name='futures_contracts',
                    columns=['symbol'],
                    unique=True)


@downgrades(2)
def _downgrade_v2(op):
    """
    Downgrade assets db by removing the 'auto_close_date' column.
    """
    # Drop indices before batch
    # This is to prevent index collision when creating the temp table
    op.drop_index('ix_equities_fuzzy_symbol')
    op.drop_index('ix_equities_company_symbol')

    # Execute batch op to allow column modification in SQLite
    with op.batch_alter_table('equities') as batch_op:
        batch_op.drop_column('auto_close_date')

    # Recreate indices after batch
    op.create_index('ix_equities_fuzzy_symbol',
                    table_name='equities',
                    columns=['fuzzy_symbol'])
    op.create_index('ix_equities_company_symbol',
                    table_name='equities',
                    columns=['company_symbol'])


@downgrades(3)
def _downgrade_v3(op):
    """
    Downgrade assets db by adding a not null constraint on
    ``equities.first_traded``
    """
    op.create_table(
        '_new_equities',
        sa.Column(
            'sid',
            sa.Integer,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('symbol', sa.Text),
        sa.Column('company_symbol', sa.Text),
        sa.Column('share_class_symbol', sa.Text),
        sa.Column('fuzzy_symbol', sa.Text),
        sa.Column('asset_name', sa.Text),
        sa.Column('start_date', sa.Integer, default=0, nullable=False),
        sa.Column('end_date', sa.Integer, nullable=False),
        sa.Column('first_traded', sa.Integer, nullable=False),
        sa.Column('auto_close_date', sa.Integer),
        sa.Column('exchange', sa.Text),
    )
    op.execute(
        """
        insert into _new_equities
        select * from equities
        where equities.first_traded is not null
        """,
    )
    op.drop_table('equities')
    op.rename_table('_new_equities', 'equities')
    # we need to make sure the indices have the proper names after the rename
    op.create_index(
        'ix_equities_company_symbol',
        'equities',
        ['company_symbol'],
    )
    op.create_index(
        'ix_equities_fuzzy_symbol',
        'equities',
        ['fuzzy_symbol'],
    )


@downgrades(4)
def _downgrade_v4(op):
    """
    Downgrades assets db by copying the `exchange_full` column to `exchange`,
    then dropping the `exchange_full` column.
    """
    op.drop_index('ix_equities_fuzzy_symbol')
    op.drop_index('ix_equities_company_symbol')

    op.execute("UPDATE equities SET exchange = exchange_full")

    with op.batch_alter_table('equities') as batch_op:
        batch_op.drop_column('exchange_full')

    op.create_index('ix_equities_fuzzy_symbol',
                    table_name='equities',
                    columns=['fuzzy_symbol'])
    op.create_index('ix_equities_company_symbol',
                    table_name='equities',
                    columns=['company_symbol'])


@downgrades(5)
def _downgrade_v5(op):
    op.create_table(
        '_new_equities',
        sa.Column(
            'sid',
            sa.Integer,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('symbol', sa.Text),
        sa.Column('company_symbol', sa.Text),
        sa.Column('share_class_symbol', sa.Text),
        sa.Column('fuzzy_symbol', sa.Text),
        sa.Column('asset_name', sa.Text),
        sa.Column('start_date', sa.Integer, default=0, nullable=False),
        sa.Column('end_date', sa.Integer, nullable=False),
        sa.Column('first_traded', sa.Integer),
        sa.Column('auto_close_date', sa.Integer),
        sa.Column('exchange', sa.Text),
        sa.Column('exchange_full', sa.Text)
    )

    op.execute(
        """
        insert into _new_equities
        select
            equities.sid as sid,
            sym.symbol as symbol,
            sym.company_symbol as company_symbol,
            sym.share_class_symbol as share_class_symbol,
            sym.company_symbol || sym.share_class_symbol as fuzzy_symbol,
            equities.asset_name as asset_name,
            equities.start_date as start_date,
            equities.end_date as end_date,
            equities.first_traded as first_traded,
            equities.auto_close_date as auto_close_date,
            equities.exchange as exchange,
            equities.exchange_full as exchange_full
        from
            equities
        inner join
            -- Select the last held symbol for each equity sid from the
            -- symbol_mappings table. Selecting max(end_date) causes
            -- SQLite to take the other values from the same row that contained
            -- the max end_date. See https://www.sqlite.org/lang_select.html#resultset.  # noqa
            (select
                 sid, symbol, company_symbol, share_class_symbol, max(end_date)
             from
                 equity_symbol_mappings
             group by sid) as 'sym'
        on
            equities.sid == sym.sid
        """,
    )
    op.drop_table('equity_symbol_mappings')
    op.drop_table('equities')
    op.rename_table('_new_equities', 'equities')
    # we need to make sure the indicies have the proper names after the rename
    op.create_index(
        'ix_equities_company_symbol',
        'equities',
        ['company_symbol'],
    )
    op.create_index(
        'ix_equities_fuzzy_symbol',
        'equities',
        ['fuzzy_symbol'],
    )


@downgrades(6)
def _downgrade_v6(op):
    op.drop_table('equity_supplementary_mappings')


@downgrades(7)
def _downgrade_v7(op):
    tmp_name = '_new_equities'
    op.create_table(
        tmp_name,
        sa.Column(
            'sid',
            sa.Integer,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('asset_name', sa.Text),
        sa.Column('start_date', sa.Integer, default=0, nullable=False),
        sa.Column('end_date', sa.Integer, nullable=False),
        sa.Column('first_traded', sa.Integer),
        sa.Column('auto_close_date', sa.Integer),

        # remove foreign key to exchange
        sa.Column('exchange', sa.Text),

        # add back exchange full column
        sa.Column('exchange_full', sa.Text),
    )
    op.execute(
        """
        insert into
            _new_equities
        select
            eq.sid,
            eq.asset_name,
            eq.start_date,
            eq.end_date,
            eq.first_traded,
            eq.auto_close_date,
            ex.canonical_name,
            ex.exchange
        from
            equities eq
        inner join
            exchanges ex
        on
            eq.exchange == ex.exchange
        where
            ex.country_code in ('US', '??')
        """,
    )
    op.drop_table('equities')
    op.rename_table(tmp_name, 'equities')

    # rebuild all tables without a foreign key to ``exchanges``
    alter_columns(
        op,
        'futures_root_symbols',
        sa.Column(
            'root_symbol',
            sa.Text,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('root_symbol_id', sa.Integer),
        sa.Column('sector', sa.Text),
        sa.Column('description', sa.Text),
        sa.Column('exchange', sa.Text),
    )
    alter_columns(
        op,
        'futures_contracts',
        sa.Column(
            'sid',
            sa.Integer,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('symbol', sa.Text, unique=True, index=True),
        sa.Column('root_symbol', sa.Text, index=True),
        sa.Column('asset_name', sa.Text),
        sa.Column('start_date', sa.Integer, default=0, nullable=False),
        sa.Column('end_date', sa.Integer, nullable=False),
        sa.Column('first_traded', sa.Integer),
        sa.Column('exchange', sa.Text),
        sa.Column('notice_date', sa.Integer, nullable=False),
        sa.Column('expiration_date', sa.Integer, nullable=False),
        sa.Column('auto_close_date', sa.Integer, nullable=False),
        sa.Column('multiplier', sa.Float),
        sa.Column('tick_size', sa.Float),
    )

    # drop the ``country_code`` and ``canonical_name`` columns
    alter_columns(
        op,
        'exchanges',
        sa.Column(
            'exchange',
            sa.Text,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('timezone', sa.Text),
        # Set the timezone to NULL because we don't know what it was before.
        # Nothing in zipline reads the timezone so it doesn't matter.
        selection_string="exchange, NULL",
    )
    op.rename_table('exchanges', 'futures_exchanges')

    # add back the foreign keys that previously existed
    alter_columns(
        op,
        'futures_root_symbols',
        sa.Column(
            'root_symbol',
            sa.Text,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('root_symbol_id', sa.Integer),
        sa.Column('sector', sa.Text),
        sa.Column('description', sa.Text),
        sa.Column(
            'exchange',
            sa.Text,
            sa.ForeignKey('futures_exchanges.exchange'),
        ),
    )
    alter_columns(
        op,
        'futures_contracts',
        sa.Column(
            'sid',
            sa.Integer,
            unique=True,
            nullable=False,
            primary_key=True,
        ),
        sa.Column('symbol', sa.Text, unique=True, index=True),
        sa.Column(
            'root_symbol',
            sa.Text,
            sa.ForeignKey('futures_root_symbols.root_symbol'),
            index=True
        ),
        sa.Column('asset_name', sa.Text),
        sa.Column('start_date', sa.Integer, default=0, nullable=False),
        sa.Column('end_date', sa.Integer, nullable=False),
        sa.Column('first_traded', sa.Integer),
        sa.Column(
            'exchange',
            sa.Text,
            sa.ForeignKey('futures_exchanges.exchange'),
        ),
        sa.Column('notice_date', sa.Integer, nullable=False),
        sa.Column('expiration_date', sa.Integer, nullable=False),
        sa.Column('auto_close_date', sa.Integer, nullable=False),
        sa.Column('multiplier', sa.Float),
        sa.Column('tick_size', sa.Float),
    )

    # Delete equity_symbol_mappings records that no longer refer to valid sids.
    op.execute(
        """
        DELETE FROM
            equity_symbol_mappings
        WHERE
            sid NOT IN (SELECT sid FROM equities);
        """
    )

    # Delete asset_router records that no longer refer to valid sids.
    op.execute(
        """
        DELETE FROM
            asset_router
        WHERE
            sid
            NOT IN (
                SELECT sid FROM equities
                UNION
                SELECT sid FROM futures_contracts
            );
        """
    )
