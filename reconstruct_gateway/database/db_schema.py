# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import sqlalchemy as sa
from gateway.database import metadata, engine


asset_router = sa.Table(
    'asset_router',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        unique=True,
        nullable=False,
        primary_key=True,
    ),
    # asset_name --- 公司名称
    sa.Column(
        'asset_name',
        sa.Text,
        # unique=True,
        nullable=False
    ),
    sa.Column(
        'asset_type',
        sa.String(20),
        nullable=False
    ),
    sa.Column(
        'exchange',
        # 中文完整的交易所
        sa.String(20)
    ),
    sa.Column(
        'first_traded',
        sa.String(30),
        # nullable=False
    ),
    sa.Column(
        'last_traded',
        sa.String(10),
    ),
    # # null / d / p / st  --- null(normal)
    # sa.Column(
    #     'status',
    #     sa.String(6),
    #     default='null'
    # ),
    sa.Column(
        'country_code',
        sa.String(6),
        default='CH'
    ),
    extend_existing=True,
)

# status table intend to sync asset_router last_traded
equity_status = sa.Table(
    'equity_status',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=True,
        primary_key=True,
    ),
    sa.Column(
        'name',
        sa.String(30),
        nullable=True,
        primary_key=True
    ),
    sa.Column(
        'last_traded',
        sa.String(10)
    ),
    sa.Column(
        'status',
        sa.String(5)
    ),
    extend_existing=True
)


equity_basics = sa.Table(
    'equity_basics',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        # 一对一 或者一对多， 索引一种
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,
        index=True
    ),
    sa.Column(
        'dual_sid',
        sa.String(10),
        # default='null'
    ),
    sa.Column(
        'broker',
        sa.Text,
        nullable=False
    ),
    # district code 可能出现多个
    # e.g. 600286 --- 412200,518000
    sa.Column(
        'district',
        # sa.String(8),
        sa.Text,
        nullable=False
    ),
    sa.Column(
        'initial_price',
        sa.Numeric(10, 2),
        nullable=False
    ),
    sa.Column(
        'business_scope',
        sa.Text,
        nullable=False
    ),
    extend_existing=True
)

#  --- 主要考虑流动性和折价空间, 强赎就是半路就赎回了，转债价格超过130的，大部分都满足强赎条件了， 有的可转债可以没回售条款比如金融行业（中行)
convertible_basics = sa.Table(
    'convertible_basics',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True,
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        unique=True,
        nullable=False,
        primary_key=True,
        index=True
    ),
    sa.Column(
        # 股票可以发行不止一个可转债
        'swap_code',
        sa.String(10),
        nullable=False,
        primary_key=True
    ),
    sa.Column(
        # 可转债发行价格
        'put_price',
        sa.Numeric(10, 3),
        nullable=False
    ),
    sa.Column(
        # 转股日期
        'convert_dt',
        sa.String(10),
        nullable=False
    ),
    sa.Column(
        # 转股价
        'convert_price',
        sa.Numeric(10, 2),
        nullable=False
    ),
    sa.Column(
        # 回售触发价 --- 基于convert_price计算回售触发条款一般为连续30个交易日70%
        'put_convert_price',
        sa.Numeric(10, 2),
        # nullable=False
    ),
    sa.Column(
        # 强制赎回价 --- 基于convert_price计算强制赎回价条款一般为连续30个交易日中不少于15个交易日130%
        'force_redeem_price',
        sa.Numeric(10, 2),
        # nullable=False
    ),
    sa.Column(
        # 一旦触发了强制赎回条款之后以redeem_price赎回未转股的可转债
        'redeem_price',
        sa.Numeric(10, 2),
        # nullable=False
    ),
    sa.Column(
        # 担保人
        'guarantor',
        sa.Text,
        # nullable=False
    ),
    extend_existing=True
)

equity_price = sa.Table(
    'equity_price',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'trade_dt',
        sa.String(10),
        nullable=False,
        primary_key=True,
    ),
    sa.Column('open', sa.Numeric(10, 5), nullable=False),
    sa.Column('high', sa.Numeric(10, 5), nullable=False),
    sa.Column('low', sa.Numeric(10, 5), nullable=False),
    sa.Column('close', sa.Numeric(10, 5), nullable=False),
    sa.Column('volume', sa.Integer, nullable=False),
    sa.Column('amount', sa.Numeric(20, 5), nullable=False),
    sa.Column('pct', sa.Numeric(10, 5), nullable=False),
    extend_existing=True

)

convertible_price = sa.Table(
    'convertible_price',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'trade_dt',
        sa.String(10),
        nullable=False,
        primary_key=True
    ),
    sa.Column('open', sa.Numeric(10, 5), nullable=False),
    sa.Column('high', sa.Numeric(10, 5), nullable=False),
    sa.Column('low', sa.Numeric(10, 5), nullable=False),
    sa.Column('close', sa.Numeric(10, 5), nullable=False),
    sa.Column('volume', sa.Integer, nullable=False),
    sa.Column('amount', sa.Numeric(20, 5), nullable=False),
    extend_existing=True
)

fund_price = sa.Table(
    'fund_price',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column('sid',
              sa.String(10),
              nullable=False,
              primary_key=True,
              ),
    sa.Column(
        'trade_dt',
        sa.String(10),
        nullable=False,
        primary_key=True
    ),
    sa.Column('open', sa.Numeric(10, 5), nullable=False),
    sa.Column('high', sa.Numeric(10, 5), nullable=False),
    sa.Column('low', sa.Numeric(10, 5), nullable=False),
    sa.Column('close', sa.Numeric(10, 5), nullable=False),
    sa.Column('volume', sa.Integer, nullable=False),
    sa.Column('amount', sa.Numeric(20, 5), nullable=False),
    extend_existing=True
)

equity_splits = sa.Table(
    # declared_date : 公告日期 ; record_date(ex_date) : 登记日 ; pay_date : 除权除息日 ,effective_date :上市日期;
    # 股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
    # 红股上市日指上市公司所送红股可上市交易（卖出）的日期,上交所证券的红股上市日为股权除权日的下一个交易日；
    # 深交所证券的红股上市日为股权登记日后的第3个交易日
    'equity_splits',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'declared_date',
        sa.String(10),
        primary_key=True,
    ),
    sa.Column('ex_date', sa.String(10)),
    sa.Column('pay_date', sa.String(10)),
    sa.Column('effective_date', sa.String(10)),
    sa.Column('sid_bonus', sa.Integer),
    sa.Column('sid_transfer', sa.Integer),
    # e.g. 000507 --- 0.42429
    sa.Column('bonus', sa.Numeric(15, 10)),
    # 实施 | 不分配
    sa.Column('progress', sa.Text),
    extend_existing=True
    )

# 配股
equity_rights = sa.Table(
    'equity_rights',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,

    ),
    sa.Column(
        'declared_date',
        sa.String(10),
        primary_key=True
    ),
    sa.Column('ex_date', sa.String(10)),
    sa.Column('pay_date', sa.String(10)),
    sa.Column('effective_date', sa.String(10)),
    sa.Column('rights_bonus', sa.Integer),
    sa.Column('rights_price', sa.Numeric(10, 5)),
    extend_existing=True
)

# 股权结构 --- 由于数据问题存在declared_date ex_date 一致的情况（000012， 000002，600109）增加约束条件流通股和限制股
ownership = sa.Table(
    # ['变动日期', '公告日期', '股本结构图', '变动原因', '总股本', '流通股', '流通A股', '高管股', '限售A股',
    #  '流通B股', '限售B股', '流通H股', '国家股', '国有法人股', '境内法人股', '境内发起人股', '募集法人股',
    #  '一般法人股', '战略投资者持股', '基金持股', '转配股', '内部职工股', '优先股'],
    # 高管股 属于限售股 --- 限售A股为空 ， 但是高管股不为0 ，总股本 = 流通A股 + 高管股 + 限售A股
    'ownership',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,
    ),
    # 变动日为唯一，而公告日会重复 e.g. 000429 (2020-06-22)
    sa.Column(
        'ex_date',
        sa.String(10),
        primary_key=True
    ),
    sa.Column(
        'declared_date',
        sa.String(10),
        primary_key=True,
    ),
    sa.Column('general', sa.Numeric(15, 5)),
    # 存在刚开始非流通
    sa.Column('float',
              sa.String(20),
              primary_key=True),
    # 由于非流通分为高管股以及限制股 所以 -- 表示
    sa.Column('manager', sa.String(20)),
    sa.Column('strict',
              sa.String(20),
              primary_key=True),
    sa.Column('b_float', sa.String(20)),
    sa.Column('b_strict', sa.String(20)),
    sa.Column('h_float', sa.String(20)),
    extend_existing=True
)

# # 流通市值
m_cap = sa.Table(
    'm_cap',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,
    ),
    sa.Column(
        'trade_dt',
        sa.String(10),
        nullable=False,
        primary_key=True
    ),
    sa.Column(
        'mkv',
        sa.Numeric(30, 10),
        nullable=False
    ),
    sa.Column(
        'mkv_cap',
        sa.Numeric(30, 10),
        nullable=False),
    sa.Column(
        'mkv_strict',
        sa.Numeric(30, 10),
        nullable=False
    ),
    extend_existing=True
)

# 股东增减持
holder = sa.Table(
    'holder',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    # 同一天股票可以多次减持
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        # primary_key=True,
    ),
    sa.Column(
        'declared_date',
        sa.String(10),
        # primary_key=True
    ),
    sa.Column('股东', sa.Text),
    sa.Column('方式', sa.String(20)),
    sa.Column('变动股本', sa.Numeric(20, 5), nullable=False),
    sa.Column('总持仓', sa.Numeric(20, 5), nullable=False),
    sa.Column('占总股本比', sa.Numeric(10, 5), nullable=False),
    sa.Column('总流通股', sa.Numeric(20, 5), nullable=False),
    sa.Column('占流通比', sa.Numeric(10, 5), nullable=False),
    extend_existing=True
)

# 股东大宗交易
massive = sa.Table(
    'massive',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        # primary_key=True,

    ),
    sa.Column(
        'declared_date',
        sa.String(10),
        nullable=False,
        # primary_key=True
    ),
    # 折议率
    sa.Column('discount', sa.Text, nullable=False),
    sa.Column('bid_price', sa.Text, nullable=False),
    sa.Column('bid_volume', sa.Numeric(10, 5), nullable=False),
    sa.Column('buyer', sa.Text, nullable=False),
    sa.Column('seller', sa.Text, nullable=False),
    # 成交总额/流通市值
    sa.Column('cjeltszb', sa.Numeric(20, 18), nullable=False),
    extend_existing=True
)

# 解禁数据 release作为mysql关键字 switch to ban_lift
unfreeze = sa.Table(
    'unfreeze',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'sid',
        sa.String(10),
        sa.ForeignKey(asset_router.c.sid),
        nullable=False,
        primary_key=True,
    ),
    # 为了与其他的events 保持一致 --- declared_date
    sa.Column(
        'declared_date',
        sa.String(10),
        nullable=False,
        primary_key=True
    ),
    sa.Column('release_type', sa.Text, nullable=False),
    # 解禁市值占解禁前流动市值比例
    sa.Column('zb', sa.Numeric(20, 18), nullable=False),
    extend_existing=True
)

# 市场融资融券数据
margin = sa.Table(
    'margin',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'declared_date',
        sa.String(10),
        nullable=False,
        primary_key=True,
    ),
    # 融资余额
    sa.Column('rzye', sa.Numeric(15, 8), nullable=False),
    # 融资余额占流通市值比
    sa.Column('rzyezb', sa.Numeric(12, 10), nullable=False),
    # 融券余额
    sa.Column('rqye', sa.Numeric(15, 8), nullable=False),
    extend_existing=True
)


# 版本
version_info = sa.Table(
    'version_info',
    metadata,
    sa.Column(
        'id',
        sa.Integer,
        default=0,
        primary_key=True,
        autoincrement=True,
        index=True
    ),
    sa.Column(
        'version',
        sa.Integer,
        unique=True,
        nullable=False,
    ),
    # This constraint ensures a single entry in this table
    sa.CheckConstraint('id <= 1'),
    extend_existing=True
)

asset_db_table_names = frozenset([
    'asset_router',
    'equity_status',
    'equity_basics',
    'convertible_basics'
])

metadata.create_all(bind=engine)

__all__ = [
           'm_cap',
           'holder',
           'unfreeze',
           'massive',
           'ownership',
           'fund_price',
           'version_info',
           'asset_router',
           'equity_status',
           'equity_basics',
           'equity_price',
           'equity_splits',
           'equity_rights',
           'convertible_basics',
           'convertible_price',
           'asset_db_table_names'
]