# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import sqlalchemy as sa
from gateWay.driver import metadata
from gateWay.asset.asset_db_schema import equity_basics, convertible_basics


equity_price = sa.Table(
    'equity_price',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(equity_basics.c.sid),
        nullable=False,
        primary_key=True,
    ),
    sa.Column('trade_dt', sa.String(10), nullable=False),
    sa.Column('open', sa.Numeric(10, 2), nullable=False),
    sa.Column('high', sa.Numeric(10, 2), nullable=False),
    sa.Column('low', sa.Numeric(10, 2), nullable=False),
    sa.Column('close', sa.Numeric(10, 2), nullable=False),
    sa.Column('volume', sa.Numeric(10, 2), nullable = False),
    sa.Column('amount', sa.Numeric(20, 0), nullable=False),
    sa.Column('pct', sa.Numeric(20, 2), nullable=False),

)

convertible_price = sa.Table(
    'bond_price',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(convertible_basics.c.sid),
        nullable=False,
    ),
    sa.Column(
        'swap_code',
        sa.String(10),
        sa.ForeignKey(convertible_basics.c.swap_code),
        nullable=False
    ),
    sa.Column('trade_dt', sa.String(10), primary_key=True, nullable=False),
    sa.Column('open', sa.String(10, 2), nullable=False),
    sa.Column('high', sa.String(10, 2), nullable=False),
    sa.Column('low', sa.String(10, 2), nullable=False),
    sa.Column('close', sa.String(10, 2), nullable=False),
    sa.Column('volume', sa.String(20, 0), nullable=False),
    sa.Column('amount', sa.String(20, 2), nullable=False),
)

fund_price = sa.Table(
    'fund_price',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column('sid',
              sa.String(10),
              nullable=False,
              index=True
              ),
    sa.Column('trade_dt', sa.String(10), primary_key=True, nullable=False),
    sa.Column('open', sa.Numeric(10, 2), nullable=False),
    sa.Column('high', sa.Numeric(10, 2), nullable=False),
    sa.Column('low', sa.Numeric(10, 2), nullable=False),
    sa.Column('close', sa.Numeric(10, 2), nullable=False),
    sa.Column('volume', sa.Numeric(10, 0), nullable=False),
    sa.Column('amount', sa.Numeric(20, 2), nullable=False),
)

# declared_date : 公告日期 ; record_date(ex_date) : 登记日 ; pay_date : 除权除息日 ,effective_date :上市日期
# 股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
# 红股上市日指上市公司所送红股可上市交易（卖出）的日期,上交所证券的红股上市日为股权除权日的下一个交易日；
# 深交所证券的红股上市日为股权登记日后的第3个交易日

equity_splits = sa.Table(
    'equity_splits',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(equity_price.c.sid),
        nullable=False,
    ),
    sa.Column('declared_date', sa.String(10)),
    sa.Column('ex_date', sa.String),
    sa.Column('pay_date', sa.String),
    sa.Column('effective_date', sa.String),
    sa.Column('sid_bonus', sa.Integer),
    sa.Column('sid_transfer', sa.Integer),
    sa.Column('bonus', sa.Numeric(5, 2)),
    sa.Column('progress', sa.Text),
    )

# 配股
equity_rights = sa.Table(
    'equity_rights',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.Integer,
        sa.ForeignKey(equity_price.c.sid),
        nullable=False,
    ),
    sa.Column('declared_date', sa.String(10)),
    sa.Column('ex_date', sa.String(10)),
    sa.Column('pay_date', sa.String(10)),
    sa.Column('effective_date', sa.String),
    sa.Column('rights_bonus', sa.Integer),
    sa.Column('rights_price', sa.Numeric(6, 2)),
)

# 股权结构
# ['变动日期', '公告日期', '股本结构图', '变动原因', '总股本', '流通股', '流通A股', '高管股', '限售A股',
#  '流通B股', '限售B股', '流通H股', '国家股', '国有法人股', '境内法人股', '境内发起人股', '募集法人股',
#  '一般法人股', '战略投资者持股', '基金持股', '转配股', '内部职工股', '优先股'],
# 高管股 属于限售股 --- 限售A股为空 ， 但是高管股不为0 ，总股本 = 流通A股 + 高管股 + 限售A股
ownership = sa.Table(
    'ownership',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(equity_price.c.sid),
        nullable=False,
    ),
    sa.Column('declared_date', sa.String(10)),
    sa.Column('ex_date', sa.String(10)),
    sa.Column('general', sa.Numeric(15, 5)),
    sa.Column('float', sa.Numeric(15, 5)),
    sa.Column('strict', sa.Numeric(15, 5)),
    sa.Column('b_float', sa.Numeric(15, 5)),
    sa.Column('b_strict', sa.Numeric(15, 5)),
    sa.Column('h_float', sa.Numeric(15, 5)),
)

# 股东增减持
holder = sa.Table(
    'holder',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.Integer,
        sa.ForeignKey(equity_price.c.sid),
        nullable=False,
        primary_key=True,
    ),
    sa.Column('股东', sa.Text),
    sa.Column('方式', sa.String(20)),
    sa.Column('变动股本', sa.Numeric(10, 5), nullable=False),
    sa.Column('总持仓', sa.Numeric(10, 5), nullable=False),
    sa.Column('占总股本比', sa.Numeric(10, 5), nullable=False),
    sa.Column('总流通股', sa.Numeric(10, 5), nullable=False),
    sa.Column('占流通比', sa.Numeric(10, 5), nullable=False),
    sa.Column('declared_date', sa.String(10))
)

# 解禁数据
release = sa.Table(
    'release',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.Integer,
        sa.ForeignKey(equity_price.c.sid),
        nullable=False,
        primary_key=True,
    ),
    # sa.Column('release_date', sa.String(10), nullable=False),
    sa.Column('declared_date', sa.String(10), nullable=False),
    sa.Column('release_type', sa.Text, nullable=False),
    sa.Column('cjeltszb', sa.Numeric(10, 5), nullable=False)
)

# 股东大宗交易
massive = sa.Table(
    'massive',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'sid',
        sa.Integer,
        sa.ForeignKey(equity_price.c.sid),
        nullable=False,
    ),
    sa.Column('declared_date', sa.String(10), nullable=False),
    sa.Column('bid_price', sa.Text, nullable=False),
    sa.Column('discount', sa.Text, nullable=False),
    sa.Column('bid_volume', sa.Numeric(10, 5), nullable=False),
    sa.Column('buyer', sa.Text, nullable=False),
    sa.Column('seller', sa.Text, nullable=False),
    sa.Column('cleltszb', sa.Numeric(10, 5), nullable=False),
)

# 版本
version_info = sa.Table(
    'version_info',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        index=True,
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
    sa.CheckConstraint('id <= 1')
)


# 流通市值
# mcap = sa.Table(
#     'mcap',
#     metadata,
#     sa.Column(
#         'sid',
#         sa.Integer,
#         sa.ForeignKey(equity_price.c.sid),
#         unique=True,
#         nullable=False,
#         primary_key=True,
#     ),
#     sa.Column('trade_dt', sa.String(10), nullable=False),
#     sa.Column('mkv', sa.Numeric(15,5), nullable=False),
#     sa.Column('mkv_cap', sa.Numeric(15, 5), nullable=False),
#     sa.Column('mkv_strict', sa.Numeric(15, 5), nullable=False),
#     )


asset_db_table_names = frozenset(['equity_price', 'convertible_price', 'fund_price', 'equity_splits',
                                  'equity_rights', 'equity_structure', 'equity_mcap',
                                  'shareholder', 'release', 'massive', 'version_info'])

__all__ = [metadata, asset_db_table_names]
